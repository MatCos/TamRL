import argparse
from dataclasses import dataclass

DEFAULTS = {
    "batch_size": 32,
    "max_final_batch_size": 800,
    "warmup_fraction": 0.0,
    "min_usage_to_update": 10,
    "replay_buffer_capacity": 10000,
    "test_set_size": 64,
}


def parser_agent_params(parser: argparse.ArgumentParser) -> None:

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULTS["batch_size"],
        help=f"Batch size for RL training. (default: {DEFAULTS['batch_size']})",
    )
    parser.add_argument(
        "--max_final_batch_size",
        type=int,
        default=DEFAULTS["max_final_batch_size"],
        help=f"Maximum batch size for the final batch in training (default: {DEFAULTS['max_final_batch_size']})",
    )
    parser.add_argument(
        "--warmup_fraction",
        type=float,
        default=DEFAULTS["warmup_fraction"],
        help=f"Fraction of replay buffer to fill before training starts. (default: {DEFAULTS['warmup_fraction']})",
    )
    parser.add_argument(
        "--min_usage_to_update",
        type=int,
        default=DEFAULTS["min_usage_to_update"],
        help=f"Block model updates once every sample in the replay buffer has been used at least this many times. (default: {DEFAULTS['min_usage_to_update']})",
    )
    parser.add_argument(
        "--replay_buffer_capacity",
        type=int,
        default=DEFAULTS["replay_buffer_capacity"],
        help=f"Maximum capacity of the replay buffer. (default: {DEFAULTS['replay_buffer_capacity']})",
    )
    parser.add_argument(
        "--test_set_size",
        type=int,
        default=DEFAULTS["test_set_size"],
        help=f"Number of first replay buffer samples to keep as a fixed test set for tracking loss over time. 0 to disable. (default: {DEFAULTS['test_set_size']})",
    )


@dataclass(frozen=True)
class AgentConfig:
    batch_size: int
    underlying_batch_size: int
    warmup_fraction: float
    min_usage_to_update: int
    replay_buffer_capacity: int
    test_set_size: int


def get_agent_config(args) -> AgentConfig:
    return AgentConfig(
        batch_size=getattr(args, "batch_size", DEFAULTS["batch_size"]),
        underlying_batch_size=getattr(
            args, "max_final_batch_size", DEFAULTS["max_final_batch_size"]
        ),
        warmup_fraction=getattr(args, "warmup_fraction", DEFAULTS["warmup_fraction"]),
        min_usage_to_update=getattr(
            args,
            "min_usage_to_update",
            DEFAULTS["min_usage_to_update"],
        ),
        replay_buffer_capacity=getattr(
            args, "replay_buffer_capacity", DEFAULTS["replay_buffer_capacity"]
        ),
        test_set_size=getattr(args, "test_set_size", DEFAULTS["test_set_size"]),
    )
