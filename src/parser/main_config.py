import argparse
import os
import random
import time
from dataclasses import dataclass

from src.utils.utils import get_slurm_id

DEFAULTS = {
    "resume_run": None,
    "load_from_run": None,
    "load_from_checkpoint": "last",
    "save_path_prefix": "output",
    "seed": 42,
    "wandb_suffix": "",
}


def parser_main_params(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resume_run",
        type=str,
        default=DEFAULTS["resume_run"],
        help=f"run name for wandb and model save path. If given and already existing, resume run. (default: {DEFAULTS['resume_run']})",
    )
    parser.add_argument(
        "--load_from_run",
        type=str,
        default=DEFAULTS["load_from_run"],
        help=f"run name to load model and config from for evaluation (default: {DEFAULTS['load_from_run']})",
    )
    parser.add_argument(
        "--load_from_checkpoint",
        type=str,
        default=DEFAULTS["load_from_checkpoint"],
        help=f"the model checkpoint to load (e.g. last, or specific chkp) (default: {DEFAULTS['load_from_checkpoint']})",
    )
    parser.add_argument(
        "--save_path_prefix",
        type=str,
        default=DEFAULTS["save_path_prefix"],
        help=f"path prefix for model save path (default: {DEFAULTS['save_path_prefix']})",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force using CPU even if CUDA/MPS is available",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULTS["seed"],
        help=f"Random seed for reproducibility (default: {DEFAULTS['seed']})",
    )
    parser.add_argument(
        "--wandb_suffix",
        type=str,
        default=DEFAULTS["wandb_suffix"],
        help=f"Suffix for the Weights & Biases run name (default: {DEFAULTS['wandb_suffix']})",
    )


@dataclass(frozen=True)
class MainConfig:
    run_name: str
    resume_run: str | None
    load_from_run: str | None
    run_path: str
    checkpoint: str
    save_path_prefix: str
    use_wandb: bool
    force_cpu: bool
    seed: int
    wandb_suffix: str
    slurm_id: str | None


def get_main_config(args) -> MainConfig:
    run_path, run_name = create_run_name(args)

    return MainConfig(
        run_name=run_name,
        run_path=run_path,
        load_from_run=getattr(args, "load_from_run", DEFAULTS["load_from_run"]),
        resume_run=getattr(args, "resume_run", DEFAULTS["resume_run"]),
        checkpoint=getattr(
            args, "load_from_checkpoint", DEFAULTS["load_from_checkpoint"]
        ),
        save_path_prefix=getattr(
            args, "save_path_prefix", DEFAULTS["save_path_prefix"]
        ),
        use_wandb=getattr(args, "use_wandb", False),
        force_cpu=getattr(args, "force_cpu", False),
        seed=getattr(args, "seed", DEFAULTS["seed"]),
        wandb_suffix=getattr(args, "wandb_suffix", DEFAULTS["wandb_suffix"]),
        slurm_id=get_slurm_id(),
    )


def create_run_name(args) -> tuple[str, str]:

    random.seed(time.time())

    arg_resume_run = getattr(args, "resume_run", DEFAULTS["resume_run"])
    arg_seed = getattr(args, "seed", DEFAULTS["seed"])
    arg_save_path_prefix = getattr(
        args, "save_path_prefix", DEFAULTS["save_path_prefix"]
    )

    if arg_resume_run is None:
        run_name = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))
        print(f"Creating new run {run_name}")
    else:
        run_name = arg_resume_run
        print(f"Resuming run {run_name}")

    path = os.path.join(arg_save_path_prefix, run_name) + "/"

    random.seed(arg_seed)

    os.makedirs(path, exist_ok=True)

    return path, run_name
