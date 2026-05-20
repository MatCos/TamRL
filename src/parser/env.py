import argparse
from dataclasses import dataclass

DEFAULTS = {
    "num_envs": 4,
    "stall_frequency": 1,
    "request_timeout": 120,
    "backup_request_timeout": 600,
    "cache_max_memory_mb": 100000.0,
}


def parser_env_params(parser: argparse.ArgumentParser) -> None:
    """Adds Tamarin environment-specific hyperparameters to the parser."""
    parser.add_argument(
        "--num_envs",
        type=int,
        default=DEFAULTS["num_envs"],
        help=f"Number of parallel environments for training. (default: {DEFAULTS['num_envs']})",
    )
    parser.add_argument(
        "--stall_frequency",
        type=int,
        default=DEFAULTS["stall_frequency"],
        help=f"Wait for this many model updates before starting new parallel environments. (default: {DEFAULTS['stall_frequency']})",
    )
    parser.add_argument(
        "--request_timeout",
        type=int,
        default=DEFAULTS["request_timeout"],
        help=f"Number of seconds to wait for a response from the worker before considering backup timeout. (default: {DEFAULTS['request_timeout']})",
    )
    parser.add_argument(
        "--backup_request_timeout",
        type=int,
        default=DEFAULTS["backup_request_timeout"],
        help=f"Number of seconds to wait for a response from the worker before considering sequential request. (default: {DEFAULTS['backup_request_timeout']})",
    )
    parser.add_argument(
        "--cache_max_memory_mb",
        type=float,
        default=DEFAULTS["cache_max_memory_mb"],
        help=f"Maximum memory for the Tamarin state cache in MB. (default: {DEFAULTS['cache_max_memory_mb']})",
    )


@dataclass(frozen=True)
class EnvConfig:
    num_envs: int
    stall_frequency: int
    request_timeout: int
    backup_request_timeout: int
    cache_max_memory_mb: float


def get_env_config(args) -> EnvConfig:
    return EnvConfig(
        num_envs=getattr(args, "num_envs", DEFAULTS["num_envs"]),
        stall_frequency=getattr(args, "stall_frequency", DEFAULTS["stall_frequency"]),
        request_timeout=getattr(args, "request_timeout", DEFAULTS["request_timeout"]),
        backup_request_timeout=getattr(
            args, "backup_request_timeout", DEFAULTS["backup_request_timeout"]
        ),
        cache_max_memory_mb=getattr(args, "cache_max_memory_mb", DEFAULTS["cache_max_memory_mb"]),
    )
