import argparse
from dataclasses import dataclass

DEFAULTS = {
    "log_avg_window": 100,
    "log_freq": 50,
    "log_avg_window_model_step": 10,
    "log_avg_window_env_step": 10,
}


def parser_recorder_params(parser: argparse.ArgumentParser) -> None:
    """Adds logging-specific hyperparameters to the parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to add arguments to.
    """
    parser.add_argument(
        "--log_avg_window",
        type=int,
        default=DEFAULTS["log_avg_window"],
        help=f"Window size for averaging logged metrics. (default: {DEFAULTS['log_avg_window']})",
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=DEFAULTS["log_freq"],
        help=f"Frequency (in steps) to log training metrics. (default: {DEFAULTS['log_freq']})",
    )
    parser.add_argument(
        "--log_avg_window_model_step",
        type=int,
        default=DEFAULTS["log_avg_window_model_step"],
        help=f"Window size for averaging model step logged metrics. (default: {DEFAULTS['log_avg_window_model_step']})",
    )
    parser.add_argument(
        "--log_avg_window_env_step",
        type=int,
        default=DEFAULTS["log_avg_window_env_step"],
        help=f"Window size for averaging env step logged metrics. (default: {DEFAULTS['log_avg_window_env_step']})",
    )


@dataclass(frozen=True)
class RecorderConfig:
    log_freq: int
    log_avg_window_model_step: int
    log_avg_window_env_step: int
    log_avg_window: int


def get_recorder_config(args) -> RecorderConfig:
    return RecorderConfig(
        log_freq=getattr(args, "log_freq", DEFAULTS["log_freq"]),
        log_avg_window_model_step=getattr(
            args, "log_avg_window_model_step", DEFAULTS["log_avg_window_model_step"]
        ),
        log_avg_window_env_step=getattr(
            args, "log_avg_window_env_step", DEFAULTS["log_avg_window_env_step"]
        ),
        log_avg_window=getattr(args, "log_avg_window", DEFAULTS["log_avg_window"]),
    )
