import torch
import torch.nn as nn

from src.models.tamarin_transformer import (
    LogitHead,
    TransformerBaseModel,
    TransformerEncoder,
)
from src.models.utils import mean_pooling


class ValueHead(nn.Module):
    def __init__(self, pfm_dim, pfm_layers, device, use_pointer=True):
        super().__init__()
        self.device = device
        self.inner = LogitHead(
            pfm_dim=pfm_dim,
            pfm_layers=pfm_layers,
            device=device,
            use_pointer=use_pointer,
        )

    def forward(self, encodings, slices):
        scores = self.inner(encodings, slices)

        lengths = slices[1:] - slices[:-1]
        batch_indices = torch.repeat_interleave(
            torch.arange(len(lengths), device=self.device), lengths
        )

        value = torch.zeros(len(lengths), device=self.device, dtype=scores.dtype)
        value.index_add_(0, batch_indices, scores)
        value = value / lengths
        return value


class ActorCriticModel(TransformerBaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Actor (Policy) Head: Outputs logits for each proof method
        # We use LogitHead
        self.policy_head = LogitHead(
            pfm_dim=self.pfm_dim,
            pfm_layers=self.pfm_layers,
            device=self.device,
            use_pointer=self.use_pointer,
        )

        # Critic (Value) Head: Outputs scalar value for the state
        # We use ValueHead to get per-action scores, then pool them
        self.value_head = ValueHead(
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
        slices = slices.to(self.device)
        input_ids = data["input_ids"].to(self.device)
        padding_mask = (data["attention_mask"] == 0).to(self.device)

        sequence_encoding = self.encoder(input_ids, padding_mask)
        encoding = mean_pooling(sequence_encoding, padding_mask)

        policy_logits = self.policy_head(encoding, slices)
        value = self.value_head(encoding, slices)

        return policy_logits, value
