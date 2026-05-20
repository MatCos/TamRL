import torch


def get_sinusoidal_positional_encoding(seq_len, d_model, device):
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float, device=device)
        * -(torch.log(torch.tensor(10000.0)) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def mean_pooling(out, padding_mask):
    # Aggregate sequence output (e.g., mean pooling)
    if padding_mask is None:
        out = out.mean(dim=1)
    else:
        # padding_mask: True for padding tokens -> we want 1 for real tokens
        assert (
            padding_mask.shape[0] == out.shape[0]
        ), "Batch size mismatch in padding mask"
        assert (
            padding_mask.shape[1] == out.shape[1]
        ), "Sequence length mismatch in padding mask"
        mask = (~padding_mask).unsqueeze(-1).to(dtype=out.dtype, device=out.device)
        denom = mask.sum(dim=1).clamp(min=1.0)
        out = (out * mask).sum(dim=1) / denom
    return out
