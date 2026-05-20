import os
import random
import time
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

import torch

THEORY_INDEX = "theoryIndex"
THEORY_KIND = "theoryKind"
THEORY_NAME = "theoryName"
THEORY_RAW = "theoryRaw"
LEMMAS = "lemmas"
LEMMA_NAME = "name"
PROOF_STATE = "proofState"
QUANTIFIER = "quantifier"
CASES = "cases"
CHOSEN_PROOF_METHOD = "chosenProofMethod"
PROOF_STATUS = "proofStatus"
STATUS = "status"
ENCODED_METHODS = "encodedMethods"
ENCODED_SYS = "encodedSys"
CONSTRAINT_SYSTEM = "constraintSystem"
PROOF_METHODS = "proofMethods"
NEW_THEORY_INDEX = "newTheoryIndex"
NEXT_PROOF_PATH = "nextProofpath"
PROOF_METHOD_NAME = "name"
SIDE = "side"
COMPLETE_PROOF = "CompleteProof"
INCOMPLETE_PROOF = "IncompleteProof"
UNFINISHABLE_PROOF = "UnfinishableProof"
TRACE_FOUND = "TraceFound"
SOLVED = "Solved"
CONTRADICTORY = "Contradictory"
UNDETERMINED = "Undetermined"
CASE_NAME = "caseName"
QUEUE_TIMEOUT = 10800  # 3 hours, to avoid hanging indefinitely if a worker dies without sending a kill signal


def create_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def create_run_name_legacy(args) -> str:

    random.seed(time.time())

    if args.run_name is None:
        args.run_name = "".join(
            random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8)
        )

    path = os.path.join(args.save_path_prefix, args.run_name) + "/"

    if os.path.exists(path) and args.resume:
        print(f"Resuming run {args.run_name}")
        args.base_model = args.run_name
    elif not os.path.exists(path):
        print(f"Creating new run {args.run_name}")
        args.base_model = None
    else:
        # os.path.exists(path) and not args.resume:
        args.base_model = args.run_name
        args.run_name = "".join(
            random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8)
        )
        path = os.path.join(args.save_path_prefix, args.run_name) + "/"
        print(
            f"Not resuming run. Using base model {args.base_model} and run name {args.run_name}"
        )

    random.seed(args.seed)

    os.makedirs(path, exist_ok=True)

    return path


def print_banner(msg):
    msg_len = len(msg)
    filler_len = (80 - msg_len) // 2
    header = "=" * filler_len + msg + "=" * filler_len
    print(header + "\n")
    print("=" * 80 + "\n", flush=True)


def eval_num(path):
    return len(
        [
            d
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d)) and d.startswith("eval")
        ]
    )


def get_slurm_id() -> str | None:
    array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
    task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if array_job_id and task_id:
        return f"{array_job_id}_{task_id}"
    return os.getenv("SLURM_JOB_ID")


def make_hashable(obj: Any) -> Any:
    """Recursively convert common container types into hashable equivalents."""
    if isinstance(obj, dict):
        # sort items to ensure deterministic ordering
        return tuple((k, make_hashable(v)) for k, v in sorted(obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(make_hashable(v) for v in obj)
    if isinstance(obj, (set, frozenset)):
        return tuple(sorted(make_hashable(v) for v in obj))
    try:
        hash(obj)
        return obj
    except TypeError:
        return repr(obj)


def dataclass_to_dict(instance, separator="/"):
    flat_dict = {}

    def _recurse(instance, prefix=""):
        # Iterate over fields of the current dataclass instance
        for field in fields(instance):
            value = getattr(instance, field.name)

            # Create the key using the FIELD name
            current_key = f"{prefix}{field.name}"

            if is_dataclass(value):
                # Recurse deeper, adding the separator to the prefix
                _recurse(value, prefix=f"{current_key}{separator}")
            elif isinstance(value, (list, tuple)):
                # Handle lists/tuples by indexing each element
                for i, item in enumerate(value):
                    item_key = f"{current_key}{separator}{i}"
                    if is_dataclass(item):
                        _recurse(item, prefix=f"{item_key}{separator}")
                    else:
                        flat_dict[item_key] = item
            elif isinstance(value, Enum):
                flat_dict[current_key] = str(value)
            else:
                # Assign the value to the dictionary
                flat_dict[current_key] = value

    _recurse(instance)

    return flat_dict


class TamarinCallTimeoutError(Exception):
    pass


class QueueWaitTimeoutError(Exception):
    pass
