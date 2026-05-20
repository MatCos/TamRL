import os
from abc import ABC
from typing import Callable

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import BatchEncoding
from typing_extensions import Self

from src.datasets.tokenizer import HFTokenizerWrapper
from src.models.base import Base
from src.models.utils import get_sinusoidal_positional_encoding, mean_pooling
from src.parser import TransformerConfig


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        pfm_dim,
        transformer_layers,
        vocab_size,
        device,
        n_head,
        dim_feedforward,
        dropout,
    ):
        super().__init__()
        # Layer to normalize the embeddings after adding positional encoding
        self.embedding_layer_norm = nn.LayerNorm(pfm_dim)
        self.device = device

        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=pfm_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pfm_dim,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )

    def forward(self, input_ids, padding_mask) -> torch.Tensor:

        emb = self.embedder(input_ids)
        # Add standard sinusoidal positional encoding
        batch_size, seq_len, d_model = emb.shape
        pe = get_sinusoidal_positional_encoding(seq_len, d_model, emb.device)
        emb = self.embedding_layer_norm(emb + pe.unsqueeze(0))

        # emb: [batch_size, seq_len, input_dim]
        # The padding_mask tells the attention mechanism to ignore padding tokens.
        return self.transformer_encoder(emb, src_key_padding_mask=padding_mask)


class LogitHead(nn.Module):

    def __init__(
        self,
        pfm_dim,
        pfm_layers,
        device,
        use_pointer=True,
        out_features=1,
    ) -> None:
        super().__init__()
        self.device = device
        self.pfm_dim = pfm_dim
        self.pfm_layers = pfm_layers

        self.k_transform = nn.Linear(
            in_features=pfm_dim,
            out_features=pfm_dim,
        )

        self.q_transform = nn.Linear(
            in_features=pfm_dim,
            out_features=pfm_dim,
        )

        self.scorer = nn.Linear(
            in_features=pfm_dim,
            out_features=out_features,
        )

        self.pre_q_transform = nn.Sequential(
            nn.Linear(pfm_dim, pfm_dim),
            nn.ReLU(),
            nn.Linear(pfm_dim, pfm_dim),
        )

        self.use_pointer = use_pointer

        self.to(device)

    def forward(self, encodings, slices) -> torch.Tensor:

        k = self.k_transform(encodings)
        q = self.pre_q_transform(encodings)
        if self.use_pointer:
            # Vectorized implementation of the pointer mechanism
            num_samples = len(slices) - 1
            # 1. Create an index tensor that maps each item to its sample index
            slice_lengths = slices[1:] - slices[:-1]
            sample_indices = (
                torch.arange(num_samples, device=self.device)
                .repeat_interleave(slice_lengths)
                .to(self.device)
            )

            assert len(sample_indices) == len(
                q
            ), f"Index OOB in pointer mechanism. Num pfms {len(sample_indices)}, q len {len(q)}"

            # 2. Perform a segmented sum to get the sum of 'out' for each sample
            q_sums = torch.zeros(
                num_samples, self.pfm_dim, device=self.device, dtype=encodings.dtype
            )
            q_sums.scatter_add_(0, sample_indices.unsqueeze(-1).expand_as(encodings), q)

            # 3. Transform the sums to get the query vectors 'q' for each sample
            q_per_sample = self.q_transform(q_sums)

            # 4. Gather the corresponding 'q' for each item in 'k' and compute logits
            q_broadcast = q_per_sample[sample_indices]
            logits = self.scorer(torch.tanh(k + q_broadcast)).squeeze(-1)
        else:
            logits = self.scorer(torch.tanh(k)).squeeze(-1)

        return logits


class TransformerBaseModel(Base, ABC):

    def __init__(
        self,
        model_config: TransformerConfig,
        vocab_size,
        device,
    ):
        super().__init__()

        self.device = device
        self.pfm_dim = model_config.pfm_dim
        self.transformer_layers = model_config.transformer_layers
        self.pfm_layers = model_config.pfm_layers
        self.vocab_size = vocab_size
        self.n_head = model_config.n_head
        self.dim_feedforward = model_config.dim_feedforward
        self.dropout = model_config.dropout
        self.use_pointer = model_config.use_pointer

    @torch.inference_mode()
    def inference(self, batch, slices, max_final_batch_size=None):
        # TODO max final batch size with different model outputs probably breaks
        assert max_final_batch_size is None
        try:
            if max_final_batch_size is None:
                batch = batch.to(self.device)
                slices = slices.to(self.device)
                out = self(batch, slices)
                return out
            else:
                batches, slices_batches = self.split_batch_slices(
                    batch, slices, max_final_batch_size
                )
                out = []
                for batch_sub, slices_sub in zip(batches, slices_batches):
                    out.append(self(batch_sub, slices_sub))

                assert len(out) > 0, "No outputs from split inference"
                if isinstance(out[0], tuple) and len(out[0]) == 2:
                    return (
                        torch.cat([o[0] for o in out]),
                        torch.cat([o[1] for o in out]),
                    )
                elif isinstance(out[0], torch.Tensor):
                    return torch.cat(out, dim=0)
                else:
                    raise ValueError(
                        f"Unexpected output type from model: {type(out[0])}"
                    )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("| WARNING: ran out of memory during inference")
                padding_len = (
                    batch["input_ids"].shape[1] if "input_ids" in batch else "unknown"
                )
                print(f"Inference OOM. Padding length: {padding_len}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise

    @torch.inference_mode()
    def rank_sample(
        self,
        proofmethods: list[tuple[str, list[dict]]],
        tokenizer: HFTokenizerWrapper,
    ) -> tuple[list[tuple[tuple[int, tuple[str, list[dict]]], float]], list[list[int]]]:
        """
        Rank a list of proof methods based on their scores.
        :param proofmethods: List of proof methods to rank.
        :param tokenizer: Tokenizer to use for encoding the proof methods.
        :return: List of tuples (proof_method, score) sorted by score in descending order.
        """
        scores, tokens = self.predict(proofmethods, tokenizer)
        assert isinstance(
            scores, torch.Tensor
        ), "Scores should be a torch.Tensor for ranking."
        ranked = sorted(
            zip(enumerate(proofmethods), scores.cpu().tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked, tokens

    @torch.inference_mode()
    def predict(
        self,
        proofmethods: list[tuple[str, list[dict]]],
        tokenizer: HFTokenizerWrapper,
    ):
        batch, slices = tokenizer(pfm=proofmethods, return_tensors="pt", padding=True)
        return self.inference(batch, slices), batch.input_ids.tolist()

    @classmethod
    def load_model(cls, path: str, device) -> Self:
        checkpoint = torch.load(path, weights_only=False, map_location=device)
        model_config = TransformerConfig(
            pfm_dim=checkpoint["pfm_dim"],
            transformer_layers=checkpoint["transformer_layers"],
            pfm_layers=checkpoint["pfm_layers"],
            n_head=checkpoint["n_head"],
            dim_feedforward=checkpoint["dim_feedforward"],
            dropout=checkpoint["dropout"],
            use_pointer=checkpoint.get("use_pointer", True),
        )
        self = cls(
            device=device,
            vocab_size=checkpoint["vocab_size"],
            model_config=model_config,
        )
        state = checkpoint["model_state_dict"]
        emb_key = "encoder.embedder.weight"
        if emb_key not in state:
            base = path.removesuffix(".pt")
            emb0 = torch.load(f"{base}_emb0.pt", weights_only=True, map_location=device)
            emb1 = torch.load(f"{base}_emb1.pt", weights_only=True, map_location=device)
            state[emb_key] = torch.cat([emb0, emb1], dim=0)
        self.load_state_dict(state)
        return self

    _MAX_FILE_BYTES = 95 * 1024 * 1024  # 95 MB

    def save_model(self, path: str):
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "pfm_dim": self.pfm_dim,
            "transformer_layers": self.transformer_layers,
            "pfm_layers": self.pfm_layers,
            "vocab_size": self.vocab_size,
            "n_head": self.n_head,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "use_pointer": self.use_pointer,
        }

        import io
        buf = io.BytesIO()
        torch.save(checkpoint, buf)
        if buf.tell() <= self._MAX_FILE_BYTES:
            torch.save(checkpoint, path)
            return

        state = checkpoint["model_state_dict"]
        emb_key = "encoder.embedder.weight"
        emb = state.pop(emb_key)
        mid = emb.shape[0] // 2
        base = path.removesuffix(".pt")
        torch.save(checkpoint, path)
        torch.save(emb[:mid].clone(), f"{base}_emb0.pt")
        torch.save(emb[mid:].clone(), f"{base}_emb1.pt")

    @staticmethod
    def split_batch_slices(batch, slices, max_batch_size):
        # slices: tensor of shape [num_slices+1], e.g. [0, 3, 7, 10]
        # Each slice is samples from slices[i] to slices[i+1]
        diff = torch.diff(slices)

        if sum(diff) <= max_batch_size:
            return [batch], [slices]

        if any(diff > max_batch_size):
            print(
                f"Warning: Some slices are larger ({diff}) than max_batch_size ({max_batch_size}). Not splitting batch in {'inference mode' if not torch.is_grad_enabled() else 'training_mode'}. "
                "This may lead to out of memory errors."
            )
            return [batch], [slices]

        sub = []
        current = []
        current_sum = 0

        for item in diff:
            item_val = item.item()
            if current_sum + item_val > max_batch_size:
                if current:
                    sub.append(current)
                current = [item_val]
                current_sum = item_val
            else:
                current.append(item_val)
                current_sum += item_val

        if current:
            sub.append(current)

        slices_sub = []
        for group in sub:
            slices_sub.append(
                torch.tensor(
                    [0] + list(torch.cumsum(torch.tensor(group), dim=0)),
                    device=slices.device,
                    dtype=torch.long,
                )
            )

        batches = []
        start = 0
        for s in slices_sub:
            end = start + s[-1].item()
            batch_sub = BatchEncoding({k: v[start:end] for k, v in batch.items()})
            batches.append(batch_sub)
            start = end

        return batches, slices_sub


# TODO make abstract to combine with other models
# TODO move training code to trainer class
class TamarinTransformer(TransformerBaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.value_head = LogitHead(
            pfm_dim=self.pfm_dim,
            pfm_layers=self.pfm_layers,
            device=self.device,
            use_pointer=self.use_pointer,
        )

        self.encoder = TransformerEncoder(
            pfm_dim=self.pfm_dim,
            transformer_layers=self.transformer_layers,
            vocab_size=self.vocab_size,
            device=self.device,
            n_head=self.n_head,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )

        self.to(self.device)

    def forward(self, data, slices):

        # data.x: [batch_size, seq_len] (token indices)
        input_ids = data["input_ids"].to(self.device)
        # The tokenizer provides an attention_mask (1 for real tokens, 0 for padding).
        # The transformer expects the inverse (True for padding, False for real).
        padding_mask = (data["attention_mask"] == 0).to(self.device)

        sequence_encoding = self.encoder(input_ids, padding_mask)
        encoding = mean_pooling(sequence_encoding, padding_mask)
        return self.value_head(encoding, slices)

    def _create_targets(self, slices, target_fn) -> torch.Tensor:
        targets = []
        for i, s in enumerate(slices[1:]):
            targets.append(target_fn(s - slices[i]))

        return torch.cat(targets).to(self.device)

    def fit(
        self,
        train_dataloader,
        loss_fn,
        optimizer,
        output_path,
        target_fn,
        epochs: int | None,
        steps: int | None,
        evaluator,  # BaseEvaluator,
        callback_fn: Callable[[dict[str, tuple[float, int]], int, int], None],
        steps_start: int = 0,
        use_wandb: bool = False,
        eval_steps: int = 0,
        max_final_batch_size: int = 800,
        gradient_clipping: float | None = 1.0,
    ):

        if epochs is None and steps is None:
            raise ValueError("Either epochs or steps must be specified.")
        if steps is None:
            assert epochs is not None
            steps = epochs * len(train_dataloader)
        elif epochs is not None:
            raise ValueError("Cannot specify both epochs and steps. Please choose one.")
        print("Training...")

        best: dict[str, float] = {
            "score": float("-inf") if evaluator.greater_is_better else float("inf"),
            "step": -1,
        }

        self.train()
        self.train_loop(
            train_dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            use_wandb=use_wandb,
            steps_start=steps_start,
            evaluator=evaluator,
            output_path=output_path,
            best=best,
            callback_fn=callback_fn,
            eval_steps=eval_steps,
            steps_rem=steps,
            target_fn=target_fn,
            max_final_batch_size=max_final_batch_size,
            gradient_clipping=gradient_clipping,
        )

        self.save_training_checkpoint(
            os.path.join(output_path, "ckp_last.pt"),
            steps,
            optimizer,
        )
        self.save_model(
            os.path.join(output_path, "model_last.pt"),
        )

        print("Training finished.")
        return best["step"]

    def save_training_checkpoint(self, path: str, step, optimizer):
        checkpoint = {
            "step": step + 1,
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def train_loop(
        self,
        train_dataloader,
        loss_fn,
        optimizer,
        use_wandb,
        steps_start: int,
        evaluator,  # BaseEvaluator
        output_path,
        best: dict[str, float],
        callback_fn: Callable[[dict[str, tuple[float, int]], int, int], None],
        eval_steps: int,
        steps_rem: int,
        target_fn,
        max_final_batch_size: int = 800,
        gradient_clipping: float | None = 1.0,
    ):
        pbar = tqdm(
            total=steps_rem + steps_start,
            unit="Step",
            initial=steps_start,
            desc="Training",
        )
        step = steps_start
        if eval_steps == 0:
            eval_steps = len(train_dataloader)

        best_metrics: dict[str, tuple[float, int]] = {}

        optimizer.zero_grad()

        for _ in range(steps_rem // len(train_dataloader) + 1):
            for batch, slices, _ in train_dataloader:
                epoch = step // len(train_dataloader)
                try:
                    batches, slices_batches = self.split_batch_slices(
                        batch, slices, max_final_batch_size
                    )
                    for batch_sub, slices_sub in zip(batches, slices_batches):
                        batch_sub = batch_sub.to(self.device)
                        slices_sub = slices_sub.to(self.device)
                        out = self(batch_sub, slices_sub)
                        targets = self._create_targets(slices_sub, target_fn)
                        loss = loss_fn(out, targets, slices_sub)
                        loss = loss / len(batches)
                        loss.backward()

                    if gradient_clipping is not None:
                        clip_grad_norm_(self.parameters(), max_norm=gradient_clipping)
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "step_loss": loss.item(),
                            "epoch": epoch,
                        }
                    )
                    if use_wandb:
                        import wandb

                        wandb.log(
                            {
                                "step_loss": loss.item(),
                            },
                            step=step,
                        )
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("| WARNING: ran out of memory, skipping batch")
                        padding_len = (
                            batch["input_ids"].shape[1]
                            if "input_ids" in batch
                            else "unknown"
                        )
                        print(
                            f"Batch ran out of memory, skipping. Padding length: {padding_len}"
                        )
                        for p in self.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        pbar.update(1)
                        continue  # skip this batch

                    raise e

                # eval every eval_steps or at the end of the training
                if ((step + 1) % eval_steps == 0) or (step >= steps_rem - 1):
                    evaluation = evaluator(self, step, epoch, None)

                    self.conditional_save(
                        evaluator,
                        evaluation,
                        output_path,
                        epoch,
                        step,
                        optimizer,
                        best,
                    )

                    for k, v in evaluation.items():
                        if (
                            k not in best_metrics
                            or (evaluator.greater_is_better and v > best_metrics[k][0])
                            or (
                                not evaluator.greater_is_better
                                and v < best_metrics[k][0]
                            )
                        ):
                            best_metrics[k] = (v, step)

                    if callback_fn is not None:
                        callback_fn(best_metrics, step, epoch)

                    self.train()
                step += 1
                if step >= steps_rem:
                    return

    def conditional_save(
        self,
        evaluator,  # BaseEvaluator
        evaluation,
        output_path: str,
        epoch: int,
        step: int,
        optimizer,
        best: dict[str, float],
    ):
        if (
            evaluator.greater_is_better
            and evaluation[evaluator.primary_metric] > best["score"]
        ) or (
            not evaluator.greater_is_better
            and evaluation[evaluator.primary_metric] < best["score"]
        ):
            best["score"] = evaluation[evaluator.primary_metric]
            best["step"] = step
            self.save_model(os.path.join(output_path, "model_best.pt"))
            self.save_training_checkpoint(
                os.path.join(output_path, "ckp_best.pt"), step, optimizer
            )
            self.save_model(os.path.join(output_path, f"model_{step}.pt"))
            self.save_training_checkpoint(
                os.path.join(output_path, f"ckp_{step}.pt"),
                step,
                optimizer,
            )
            print(
                f"Best model saved at step {step} (epoch {epoch}) with {evaluator.primary_metric} score {best['score']}"
            )

    @torch.no_grad()
    def evaluate_dataloader(
        self, dataloader, target_fn=None, return_tokens=False
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, list],
        list[list[str]],
        list[torch.Tensor],
    ]:
        self.eval()

        out_all = []
        targets_all = []
        slices_all = torch.tensor([0], device=self.device, dtype=torch.long)
        info_all: dict[str, list] = {}
        proofmethods_all = []
        tokens_all = []

        num_batches = len(dataloader)
        batch_size = getattr(dataloader, "batch_size", None)
        print(f"num_batches: {num_batches}, batch_size: {batch_size}")

        # save real index, proofmethods etc.
        for i, (batch, slices, info) in tqdm(
            enumerate(dataloader), desc="Predicting", total=num_batches, unit="batch"
        ):
            try:
                out = self.inference(batch, slices)
                out_all.append(out)
                if return_tokens:
                    tokens_all.append(batch["input_ids"].cpu())
                slices_all = torch.cat(
                    [
                        slices_all,
                        slices[1:]
                        + (slices_all[-1].item() if len(slices_all) > 0 else 0),
                    ]
                )
                if target_fn is not None:
                    targets_all.append(self._create_targets(slices, target_fn))

                for k in set(info_all.keys()) | set(
                    k
                    for i in info
                    for k in i.keys()
                    if k not in ["processed_idx", "proofmethods_short"]
                ):
                    merged = info_all.get(k, [])
                    for i in info:
                        if k in i:
                            merged.append(i[k])
                    info_all[k] = merged

                proofmethods_all.extend([i["proofmethods_short"] for i in info])

            except RuntimeError as e:
                print(f"Error processing batch {i}: {e}")
                if "out of memory" in str(e):
                    print("| WARNING: ran out of memory, skipping batch")
                    padding_len = (
                        batch["input_ids"].shape[1]
                        if "input_ids" in batch
                        else "unknown"
                    )
                    print(
                        f"Batch {i} ran out of memory, skipping. Padding length: {padding_len}"
                    )
                    for p in self.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue  # skip this batch
                raise e

        out_all_tensor = torch.cat(out_all)

        return (
            out_all_tensor,
            torch.cat(targets_all),
            slices_all,
            info_all,
            proofmethods_all,
            tokens_all,
        )

    def evaluate(
        self,
        evaluator,
        epoch: int | None,
        step: int = -1,
    ) -> dict[str, float]:

        print("Evaluating...")
        self.eval()

        return evaluator(self, step, epoch, None)
