import os
from time import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.datasets.extract_pfm import extract_pfm
from src.datasets.tokenizer import HFTokenizerWrapper
from src.environment.environment import State
from src.models.actor_critic import ActorCriticModel
from src.parser import AgentConfig, MainConfig, OptimizerConfig, TransformerConfig
from src.parser.lemma import LemmaConfig
from src.parser.tokenizer import TokenizerConfig
from src.rl.agent import Agent
from src.rl.replay_buffer import ACTrainingExample, Batch, ReplayBuffer
from src.utils.recorder import Recorder


def _torch_load(path: str, device: torch.device):
    if path.endswith(".gz"):
        import gzip, io
        with gzip.open(path, "rb") as f:
            return torch.load(io.BytesIO(f.read()), map_location=device)
    return torch.load(path, map_location=device)


class ACAgent(Agent):

    def __init__(
        self,
        device: torch.device,
        logger: Recorder,
        tokenizer_config: TokenizerConfig,
        agent_config: AgentConfig,
        optimizer_config: OptimizerConfig,
        model_config: TransformerConfig,
    ) -> None:
        super().__init__()

        self.device = device
        self.logger = logger

        self.warmup_fraction = agent_config.warmup_fraction
        self.underlying_batch_size = agent_config.underlying_batch_size
        self.batch_size = agent_config.batch_size
        self.min_usage_to_update = agent_config.min_usage_to_update

        self.tokenizer = HFTokenizerWrapper(tokenizer_config=tokenizer_config)

        self.model = ActorCriticModel(
            model_config=model_config,
            vocab_size=len(self.tokenizer.tokenizer),
            device=device,
        ).to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=optimizer_config.learning_rate
        )
        self.model_updates = 0

    def select_action(
        self,
        state: State,
    ) -> tuple[int, float]:
        """Selects an action greedily using the policy network only"""
        priors, _ = self.inference(state)
        return int(torch.argmax(priors).item()), float(torch.max(priors).item())

    @torch.inference_mode()
    def inference(
        self,
        state: State,
    ) -> tuple[torch.Tensor, ...]:
        """Performs inference on the given state and returns the model outputs."""
        self.model.eval()
        # Assuming state has a to_tensor method or is compatible with the network
        pfms: list[tuple[str, tuple[dict, ...]]] = extract_pfm(state, None)
        batch, slices = self.tokenizer(pfm=pfms, return_tensors="pt", padding=True)
        priors, values = self.model.inference(batch, slices)
        priors = F.log_softmax(priors, dim=0)
        return priors, values

    @torch.inference_mode()
    def evaluate_test_set(
        self,
        test_set: list[ACTrainingExample],
    ) -> dict[str, float]:
        """Computes loss and value statistics on a fixed test set."""
        self.model.eval()

        batch = Batch.from_items(test_set)
        batches = batch.slice_batch(self.underlying_batch_size)

        total_value_loss = 0.0
        total_policy_loss = 0.0
        all_predicted_values: list[torch.Tensor] = []
        all_target_values: list[torch.Tensor] = []

        for sub_batch in batches:
            states = [item.state for item in sub_batch.items]
            actions = torch.tensor(
                [item.action for item in sub_batch.items],
                device=self.device,
                dtype=torch.long,
            )
            target_values = torch.tensor(
                [item.value for item in sub_batch.items],
                device=self.device,
                dtype=torch.float32,
            )

            pfms: list[list[tuple[str, tuple[dict, ...]]]] = [
                extract_pfm(state, None) for state in states
            ]
            tokenized_states, slices = self.tokenizer(
                pfm=pfms, return_tensors="pt", padding=True
            )

            policy_logits, values = self.model(tokenized_states, slices)

            all_predicted_values.append(values.view(-1))
            all_target_values.append(target_values)

            total_value_loss += F.mse_loss(values.view(-1), target_values).item()

            # Policy loss (segmented log softmax, same as update_model)
            slices_dev = slices.to(self.device)
            lengths = slices_dev[1:] - slices_dev[:-1]
            batch_indices = torch.repeat_interleave(
                torch.arange(len(sub_batch.items), device=self.device), lengths
            )
            max_vals = torch.zeros(
                len(sub_batch.items), device=self.device, dtype=policy_logits.dtype
            )
            max_vals.fill_(float("-inf"))
            max_vals.scatter_reduce_(
                0, batch_indices, policy_logits, reduce="amax", include_self=True
            )
            logits_stable = policy_logits - max_vals[batch_indices]
            exp_logits = logits_stable.exp()
            sum_exp = torch.zeros(
                len(sub_batch.items), device=self.device, dtype=policy_logits.dtype
            )
            sum_exp.scatter_add_(0, batch_indices, exp_logits)
            log_sum_exp = sum_exp.log() + max_vals
            log_probs = policy_logits - log_sum_exp[batch_indices]
            global_action_indices = slices_dev[:-1] + actions
            selected_log_probs = log_probs[global_action_indices]
            total_policy_loss += -selected_log_probs.mean().item()

        n_batches = len(batches)
        predicted_values = torch.cat(all_predicted_values)
        target_values_all = torch.cat(all_target_values)

        return {
            "test_loss": (total_value_loss + total_policy_loss) / n_batches,
            "test_policy_loss": total_policy_loss / n_batches,
            "test_value_loss": total_value_loss / n_batches,
            "test_pred_value_mean": predicted_values.mean().item(),
            "test_pred_value_std": predicted_values.std().item() if len(predicted_values) > 1 else 0.0,
            "test_target_value_mean": target_values_all.mean().item(),
            "test_target_value_std": target_values_all.std().item() if len(target_values_all) > 1 else 0.0,
        }

    def update_model(
        self,
        replay_buffer: ReplayBuffer[ACTrainingExample],
        env_steps: int,
        root_state: dict[LemmaConfig, State],
    ) -> bool:
        self.model.train()

        log = {
            "replay_filled": replay_buffer.filled_ratio,
            "fresh_samples": replay_buffer.fresh_samples,
            "n_updates": self.model_updates,
            "mean_usage_count": replay_buffer.mean_usage_count,
            "min_usage_count": replay_buffer.min_usage_count,
        }
        if (
            len(replay_buffer) < self.batch_size
            or (
                self.warmup_fraction > 0
                and len(replay_buffer) < replay_buffer.capacity * self.warmup_fraction
            )
            or (replay_buffer.min_usage_count >= self.min_usage_to_update)
        ):

            self.logger.log_step(
                {"ModelStep": {**log}},
                env_steps,
            )
            return False

        assert len(root_state) > 0, "Root state must be provided for model update."
        start_time = time()

        samples = replay_buffer.sample(self.batch_size)
        log["sample_mean_usage"] = replay_buffer.last_sample_mean_usage

        batches = samples.slice_batch(self.underlying_batch_size)

        assert len(batches) > 0, "No batches created from slicing."

        self.optimizer.zero_grad()

        value_loss = torch.tensor(0.0, device=self.device)
        policy_loss = torch.tensor(0.0, device=self.device)

        tokenize_time = 0.0
        model_forward_time = 0.0
        model_backward_time = 0.0
        num_pfms = 0
        mem_peak_fwd = 0.0
        mem_pre_fwd = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        for batch in batches:
            states = [item.state for item in batch.items]
            actions = torch.tensor(
                [item.action for item in batch.items],
                device=self.device,
                dtype=torch.long,
            )
            target_values = torch.tensor(
                [item.value for item in batch.items],
                device=self.device,
                dtype=torch.float32,
            )

            tokenize_start_time = time()
            # tokenize states
            pfms: list[list[tuple[str, tuple[dict, ...]]]] = [
                extract_pfm(state, None) for state in states
            ]
            num_pfms += sum(len(pfm) for pfm in pfms)
            tokenized_states, slices = self.tokenizer(
                pfm=pfms, return_tensors="pt", padding=True
            )
            tokenize_time += time() - tokenize_start_time

            model_forward_start_time = time()
            policy_logits, values = self.model(tokenized_states, slices)
            model_forward_time += time() - model_forward_start_time

            if torch.cuda.is_available():
                mem_peak_fwd = max(mem_peak_fwd, torch.cuda.memory_allocated() / 1e9)

            # Value loss (MSE)
            batch_value_loss = F.mse_loss(values.view(-1), target_values)

            # Policy loss (Cross Entropy) Vectorized iteration through proofmethods per sample
            slices_dev = slices.to(self.device)
            lengths = slices_dev[1:] - slices_dev[:-1]
            batch_indices = torch.repeat_interleave(
                torch.arange(len(batch.items), device=self.device), lengths
            )
            # Segmented Max
            max_vals = torch.zeros(
                len(batch.items), device=self.device, dtype=policy_logits.dtype
            )
            max_vals.fill_(float("-inf"))
            max_vals.scatter_reduce_(
                0, batch_indices, policy_logits, reduce="amax", include_self=True
            )
            # Stable LogSumExp
            logits_stable = policy_logits - max_vals[batch_indices]
            exp_logits = logits_stable.exp()
            sum_exp = torch.zeros(
                len(batch.items), device=self.device, dtype=policy_logits.dtype
            )
            sum_exp.scatter_add_(0, batch_indices, exp_logits)
            log_sum_exp = sum_exp.log() + max_vals
            # Log Softmax
            log_probs = policy_logits - log_sum_exp[batch_indices]
            # Gather actions
            global_action_indices = slices_dev[:-1] + actions
            selected_log_probs = log_probs[global_action_indices]

            batch_policy_loss = -selected_log_probs.mean()

            batch_loss = (batch_value_loss + batch_policy_loss) / len(batches)

            model_backward_start_time = time()
            batch_loss.backward()
            model_backward_time += time() - model_backward_start_time

            value_loss += batch_value_loss.detach()
            policy_loss += batch_policy_loss.detach()

        loss = (value_loss + policy_loss) / len(batches)

        model_backward_start_time = time()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        model_backward_time += time() - model_backward_start_time

        mem_after_step = 0.0
        if torch.cuda.is_available():
            mem_after_step = torch.cuda.memory_allocated() / 1e9
        self.model_updates += 1

        _, root_values = zip(*[self.inference(state) for state in root_state.values()])
        root_value = torch.stack(root_values).mean()

        if len(replay_buffer.test_set) == replay_buffer.test_set_size > 0:
            test_metrics = self.evaluate_test_set(replay_buffer.test_set)
            log.update(test_metrics)

        if self.tokenizer.cache is not None:
            log["cache_hit_rate"] = self.tokenizer.cache.hit_rate
            log["cache_usage"] = self.tokenizer.cache.usage

        self.logger.log_step(
            {
                "ModelStep": {
                    **log,
                    **{
                        "loss": loss.item(),
                        "policy_loss": policy_loss.item() / len(batches),
                        "value_loss": value_loss.item() / len(batches),
                        "time": time() - start_time,
                        "optimize_time": model_backward_time,
                        "model_forward_time": model_forward_time,
                        "model_backward_time": model_backward_time,
                        "tokenize_time": tokenize_time,
                        "gradient_accumulation_steps": len(batches),
                        "root_value": root_value.item(),
                        "num_proof_methods": num_pfms / len(samples),
                        "gpu_mem_pre_fwd_gb": mem_pre_fwd,
                        "gpu_mem_peak_fwd_gb": mem_peak_fwd,
                        "gpu_mem_after_step_gb": mem_after_step,
                    },
                }
            },
            env_steps,
        )

        return True

    def save_checkpoint(
        self,
        path: str,
        checkpoint_name: str,
        steps: int,
    ) -> None:
        """Saves a checkpoint of the agent and training progress.

        Args:
            path (str): The directory path to save the checkpoint in.
            checkpoint_name (str): The name for the checkpoint file (e.g., '_step_1').
        """
        os.makedirs(path, exist_ok=True)
        for name in ["_last", checkpoint_name]:
            checkpoint_path = os.path.join(path, f"ckp{name}.pt")
            model_path = os.path.join(path, f"model{name}.pt")

            self.model.save_model(model_path)

            checkpoint = {
                "total_updates": self.model_updates,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "steps": steps,
            }
            torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    @classmethod
    def load_or_create_agent(
        cls,
        main_config: MainConfig,
        agent_config: AgentConfig,
        optimizer_config: OptimizerConfig,
        tokenizer_config: TokenizerConfig,
        model_config: TransformerConfig,
        device: torch.device,
        logger,
    ) -> tuple["ACAgent", int]:
        """Loads an agent from a checkpoint if specified in the main_config,
        otherwise creates a new agent.

        Args:
            main_config (MainConfig): Configuration containing information about loading/resuming runs and checkpoints.
            agent_config (AgentConfig): Configuration for the agent.
            optimizer_config (OptimizerConfig): Configuration for the optimizer.
            tokenizer_config (TokenizerConfig): Configuration for the tokenizer.
            model_config (TransformerConfig): Configuration for the model.
            device (torch.device): The device to load the agent onto.
            logger (_type_): Logger instance for logging.

        Returns:
            ACAgent: The loaded or created agent.
        """
        agent = ACAgent(
            tokenizer_config=tokenizer_config,
            agent_config=agent_config,
            optimizer_config=optimizer_config,
            model_config=model_config,
            device=device,
            logger=logger,
        )

        load_from_run = None

        if main_config.resume_run is not None:
            load_from_run = main_config.resume_run
        if main_config.load_from_run is not None:
            load_from_run = main_config.load_from_run

        if load_from_run is not None:
            checkpoint_path = os.path.join(
                main_config.save_path_prefix,
                load_from_run,
                f"ckp_{main_config.checkpoint}.pt",
            )
            model_path = os.path.join(
                main_config.save_path_prefix,
                load_from_run,
                f"model_{main_config.checkpoint}.pt",
            )

            if not os.path.exists(checkpoint_path) and os.path.exists(checkpoint_path + ".gz"):
                checkpoint_path = checkpoint_path + ".gz"
            assert os.path.exists(checkpoint_path)
            assert os.path.exists(model_path)

            print(f"Loading checkpoint from {checkpoint_path}")
            agent.model = agent.model.load_model(model_path, device)
            checkpoint = _torch_load(checkpoint_path, device)
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            agent.model_updates = checkpoint["total_updates"]
            steps = checkpoint.get("steps", 0)
            print(f"Resuming training from step {agent.model_updates}")
        else:
            steps = 0
            print("Creating a new agent.")

        return agent, steps
