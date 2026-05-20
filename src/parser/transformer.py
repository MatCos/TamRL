import argparse
from dataclasses import dataclass

DEFAULTS = {
    "pfm_dim": 512,
    "transformer_layers": 4,
    "pfm_layers": 2,
    "dropout": 0.0,
    "n_transformer_head": 8,
    "dim_transformer_feedforward": 1024,
}


def parser_transformer_params(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pfm_dim",
        type=int,
        default=DEFAULTS["pfm_dim"],
        help=f"Proof method embedding dimension (default: {DEFAULTS['pfm_dim']})",
    )
    parser.add_argument(
        "--transformer_layers",
        type=int,
        default=DEFAULTS["transformer_layers"],
        help=f"Number of transformer encoder layers (default: {DEFAULTS['transformer_layers']})",
    )
    parser.add_argument(
        "--pfm_layers",
        type=int,
        default=DEFAULTS["pfm_layers"],
        help=f"Number of head layers. (default: {DEFAULTS['pfm_layers']}).",
    )
    parser.add_argument(
        "--no_pointer",
        action="store_true",
        help="Disable pointer mechanism in the model",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULTS["dropout"],
        help=f"Dropout rate (default: {DEFAULTS['dropout']})",
    )
    parser.add_argument(
        "--n_transformer_head",
        type=int,
        default=DEFAULTS["n_transformer_head"],
        help=f"Number of heads in the transformer (default: {DEFAULTS['n_transformer_head']})",
    )
    parser.add_argument(
        "--dim_transformer_feedforward",
        type=int,
        default=DEFAULTS["dim_transformer_feedforward"],
        help=f"Dimension of the feedforward layer in the transformer (default: {DEFAULTS['dim_transformer_feedforward']})",
    )


@dataclass(frozen=True)
class TransformerConfig:
    pfm_dim: int
    transformer_layers: int
    pfm_layers: int
    n_head: int
    dim_feedforward: int
    dropout: float
    use_pointer: bool


def get_transformer_config(args) -> TransformerConfig:

    return TransformerConfig(
        pfm_dim=getattr(args, "pfm_dim", DEFAULTS["pfm_dim"]),
        transformer_layers=getattr(
            args, "transformer_layers", DEFAULTS["transformer_layers"]
        ),
        pfm_layers=getattr(args, "pfm_layers", DEFAULTS["pfm_layers"]),
        n_head=getattr(args, "n_transformer_head", DEFAULTS["n_transformer_head"]),
        dim_feedforward=getattr(
            args, "dim_transformer_feedforward", DEFAULTS["dim_transformer_feedforward"]
        ),
        dropout=getattr(args, "dropout", DEFAULTS["dropout"]),
        use_pointer=not getattr(args, "no_pointer", False),
    )
