import argparse
from dataclasses import dataclass

DEFAULTS = {
    "learning_rate": 1e-4,
}


def parser_optimizer_params(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULTS["learning_rate"],
        help=f"Learning rate for the optimizer. (default: {DEFAULTS['learning_rate']})",
    )


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float


def get_optimizer_config(args) -> OptimizerConfig:
    return OptimizerConfig(
        learning_rate=getattr(args, "learning_rate", DEFAULTS["learning_rate"]),
    )
