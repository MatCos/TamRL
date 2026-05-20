import os
import tempfile

import torch
from transformers import BatchEncoding

from src.models.actor_critic import ActorCriticModel, ValueHead
from src.models.tamarin_transformer import (
    LogitHead,
    TamarinTransformer,
    TransformerEncoder,
)
from src.models.utils import get_sinusoidal_positional_encoding, mean_pooling
from src.parser.transformer import TransformerConfig

DEVICE = "cpu"

TINY_CONFIG = TransformerConfig(
    pfm_dim=32, transformer_layers=1, pfm_layers=1,
    n_head=4, dim_feedforward=64, dropout=0.0, use_pointer=True,
)
VOCAB_SIZE = 100


def make_batch(batch_size, seq_len, vocab_size=VOCAB_SIZE):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------


class TestUtils:
    def test_positional_encoding(self):
        pe = get_sinusoidal_positional_encoding(10, 32, DEVICE)
        assert pe.shape == (10, 32)
        assert pe.min() >= -1.0
        assert pe.max() <= 1.0

    def test_mean_pooling_no_mask(self):
        out = torch.ones(2, 5, 8)
        result = mean_pooling(out, None)
        assert result.shape == (2, 8)
        assert torch.allclose(result, torch.ones(2, 8))

    def test_mean_pooling_with_mask(self):
        out = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        padding_mask = torch.tensor([[False, False, True]])
        result = mean_pooling(out, padding_mask)
        assert result.shape == (1, 2)
        assert torch.allclose(result, torch.tensor([[2.0, 3.0]]))

    def test_mean_pooling_all_padded(self):
        out = torch.ones(1, 3, 4)
        padding_mask = torch.tensor([[True, True, True]])
        result = mean_pooling(out, padding_mask)
        assert result.shape == (1, 4)
        assert not torch.isnan(result).any()


# ---------------------------------------------------------------------------
# TransformerEncoder
# ---------------------------------------------------------------------------


class TestTransformerEncoder:
    def test_output_shape(self):
        encoder = TransformerEncoder(
            pfm_dim=32, transformer_layers=1, vocab_size=VOCAB_SIZE,
            device=DEVICE, n_head=4, dim_feedforward=64, dropout=0.0,
        )
        input_ids = torch.randint(0, VOCAB_SIZE, (3, 10))
        assert encoder(input_ids, padding_mask=None).shape == (3, 10, 32)

    def test_with_padding_mask(self):
        encoder = TransformerEncoder(
            pfm_dim=32, transformer_layers=1, vocab_size=VOCAB_SIZE,
            device=DEVICE, n_head=4, dim_feedforward=64, dropout=0.0,
        )
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 8))
        padding_mask = torch.zeros(2, 8, dtype=torch.bool)
        padding_mask[0, 5:] = True
        assert encoder(input_ids, padding_mask).shape == (2, 8, 32)


# ---------------------------------------------------------------------------
# LogitHead & ValueHead
# ---------------------------------------------------------------------------


class TestLogitHead:
    def test_with_pointer(self):
        head = LogitHead(pfm_dim=32, pfm_layers=1, device=DEVICE, use_pointer=True)
        logits = head(torch.randn(5, 32), torch.tensor([0, 2, 5]))
        assert logits.shape == (5,)

    def test_without_pointer(self):
        head = LogitHead(pfm_dim=32, pfm_layers=1, device=DEVICE, use_pointer=False)
        logits = head(torch.randn(5, 32), torch.tensor([0, 2, 5]))
        assert logits.shape == (5,)


class TestValueHead:
    def test_output_shape(self):
        head = ValueHead(pfm_dim=32, pfm_layers=1, device=DEVICE, use_pointer=True)
        value = head(torch.randn(5, 32), torch.tensor([0, 2, 5]))
        assert value.shape == (2,)

    def test_single_sample(self):
        head = ValueHead(pfm_dim=32, pfm_layers=1, device=DEVICE)
        value = head(torch.randn(4, 32), torch.tensor([0, 4]))
        assert value.shape == (1,)


# ---------------------------------------------------------------------------
# TamarinTransformer
# ---------------------------------------------------------------------------


class TestTamarinTransformer:
    def test_forward(self):
        model = TamarinTransformer(
            model_config=TINY_CONFIG, vocab_size=VOCAB_SIZE, device=DEVICE,
        )
        model.eval()
        logits = model(make_batch(5, 10), torch.tensor([0, 2, 5]))
        assert logits.shape == (5,)

    def test_save_load_roundtrip(self):
        model = TamarinTransformer(
            model_config=TINY_CONFIG, vocab_size=VOCAB_SIZE, device=DEVICE,
        )
        model.eval()
        batch = make_batch(3, 8)
        slices = torch.tensor([0, 3])

        with torch.inference_mode():
            out_before = model(batch, slices)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            model.save_model(path)
            loaded = TamarinTransformer.load_model(path, DEVICE)
            loaded.eval()
            with torch.inference_mode():
                out_after = loaded(batch, slices)

        assert torch.allclose(out_before, out_after)


# ---------------------------------------------------------------------------
# ActorCriticModel
# ---------------------------------------------------------------------------


class TestActorCriticModel:
    def test_forward_returns_tuple(self):
        model = ActorCriticModel(
            model_config=TINY_CONFIG, vocab_size=VOCAB_SIZE, device=DEVICE,
        )
        model.eval()
        policy, value = model(make_batch(5, 10), torch.tensor([0, 2, 5]))
        assert policy.shape == (5,)
        assert value.shape == (2,)

    def test_save_load_roundtrip(self):
        model = ActorCriticModel(
            model_config=TINY_CONFIG, vocab_size=VOCAB_SIZE, device=DEVICE,
        )
        model.eval()
        batch = make_batch(4, 8)
        slices = torch.tensor([0, 2, 4])

        with torch.inference_mode():
            policy_before, value_before = model(batch, slices)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            model.save_model(path)
            loaded = ActorCriticModel.load_model(path, DEVICE)
            loaded.eval()
            with torch.inference_mode():
                policy_after, value_after = loaded(batch, slices)

        assert torch.allclose(policy_before, policy_after)
        assert torch.allclose(value_before, value_after)


# ---------------------------------------------------------------------------
# split_batch_slices
# ---------------------------------------------------------------------------


class TestSplitBatchSlices:
    def test_no_split_needed(self):
        batch = make_batch(5, 10)
        slices = torch.tensor([0, 2, 5])
        batches, _ = TamarinTransformer.split_batch_slices(batch, slices, 10)
        assert len(batches) == 1
        assert batches[0] is batch

    def test_splits_correctly(self):
        batch = make_batch(10, 8)
        slices = torch.tensor([0, 3, 6, 10])
        batches, slices_list = TamarinTransformer.split_batch_slices(batch, slices, 5)
        total = sum(b["input_ids"].shape[0] for b in batches)
        assert total == 10
        for s in slices_list:
            assert s[0] == 0
            assert s[-1] <= 5

    def test_oversized_slice_not_split(self):
        batch = make_batch(8, 4)
        slices = torch.tensor([0, 7, 8])
        batches, slices_list = TamarinTransformer.split_batch_slices(batch, slices, 5)
        assert len(batches) == 1
        # Returned batch must be the original (no data dropped)
        assert batches[0] is batch
        assert torch.equal(slices_list[0], slices)
