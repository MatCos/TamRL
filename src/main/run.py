import json
import os
import random

import numpy as np
import torch

from src.parser.parser_rl import parse_args_rl
from src.training.trainer import Trainer
from src.utils.utils import dataclass_to_dict


def run() -> None:
    """Parses arguments and runs the training process."""
    config = parse_args_rl()
    torch.manual_seed(config.main.seed)
    np.random.seed(config.main.seed)
    random.seed(config.main.seed)

    if torch.cuda.is_available() and config.env.num_envs == 1:
        torch.cuda.manual_seed_all(config.main.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=False)

    trainer = Trainer(config)
    trainer()
    with open(
        os.path.join(config.main.run_path, "args_train.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(dataclass_to_dict(config), f, indent=2)


if __name__ == "__main__":
    # Set the start method to 'spawn'
    # This is required for sharing CUDA tensors and avoids fork-related issues
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # This handles cases where the start method might have already been set
        pass
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    run()
