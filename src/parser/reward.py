import argparse
from dataclasses import dataclass

DEFAULTS = {
    "branch_penalty": 0.0,
    "time_penalty": 0.0,
    "time_penalty_clip": 90.0,  # 90 seconds
    "timeout_penalty": 3,
}


def parser_reward_params(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--branch_penalty",
        type=float,
        default=DEFAULTS["branch_penalty"],
        help=f"Ratio to punish branches with high branching factor. (default: {DEFAULTS['branch_penalty']})",
    )
    parser.add_argument(
        "--time_penalty",
        type=float,
        default=DEFAULTS["time_penalty"],
        help=f"Penalty for time spent in a node. (default: {DEFAULTS['time_penalty']})",
    )
    parser.add_argument(
        "--time_penalty_clip",
        type=float,
        default=DEFAULTS["time_penalty_clip"],
        help=f"Clip time penalty at this value. Used to normalize penalty. (default: {DEFAULTS['time_penalty_clip']})",
    )
    parser.add_argument(
        "--timeout_penalty",
        type=float,
        default=DEFAULTS["timeout_penalty"],
        help=f"Penalty for timing out in a node (i.e. not returning a result within the time limit). (default: {DEFAULTS['timeout_penalty']})",
    )


@dataclass(frozen=True)
class RewardConfig:
    branch_penalty: float
    time_penalty: float
    time_penalty_clip: float
    timeout_penalty: float


def get_reward_config(args) -> RewardConfig:

    return RewardConfig(
        branch_penalty=getattr(args, "branch_penalty", DEFAULTS["branch_penalty"]),
        time_penalty=getattr(args, "time_penalty", DEFAULTS["time_penalty"]),
        time_penalty_clip=getattr(
            args, "time_penalty_clip", DEFAULTS["time_penalty_clip"]
        ),
        timeout_penalty=getattr(args, "timeout_penalty", DEFAULTS["timeout_penalty"]),
    )
