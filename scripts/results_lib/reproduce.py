from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from .utils import normalize, is_completed
from .tables import PROTOCOL_ORDER

_WANDB_TO_CLI: dict[str, str] = {
    "search/budget": "search_budget",
    "search/budget_increase_factor": "search_budget_increase_factor",
    "tokenizer/cache_size": "tokenizer_cache_size",
    "tokenizer/max_length": "tokenizer_max_length",
    "tokenizer/model_name": "tokenizer",
    "transformer/n_head": "n_transformer_head",
    "transformer/dim_feedforward": "dim_transformer_feedforward",
}

_CLI_SKIP_KEYS = {
    "main/run_name", "main/run_path", "main/slurm_id", "main/use_wandb",
    "main/wandb_suffix", "main/checkpoint", "main/resume_run",
    "main/load_from_run", "main/save_path_prefix", "main/force_cpu",
    "tokenizer/key", "tokenizer/id_range", "tokenizer/replace_var",
    "ucb/deactivate_model", "transformer/use_pointer",
    "recorder/log_freq", "recorder/log_avg_window",
    "recorder/log_avg_window_env_step", "recorder/log_avg_window_model_step",
    "agent/test_set_size", "agent/underlying_batch_size",
    "main/seed",
}

_BEST_RUNS_EXCLUDE_KEYS = {
    "config_group",
    "main/run_name", "main/run_path", "main/slurm_id",
    "main/force_cpu", "main/use_wandb", "main/checkpoint", "main/resume_run",
    "main/wandb_suffix", "main/load_from_run", "main/save_path_prefix",
    "tokenizer/key", "tokenizer/id_range", "tokenizer/replace_var",
    "agent/test_set_size", "agent/underlying_batch_size",
    "ucb/deactivate_model",
    "recorder/log_freq", "recorder/log_avg_window",
    "recorder/log_avg_window_env_step", "recorder/log_avg_window_model_step",
    "transformer/use_pointer",
}


def build_reproduce_command(full_config: dict) -> str:
    defaults: dict[str, object] = {}
    try:
        from src.parser.ac_agent import DEFAULTS as agent_d
        from src.parser.env import DEFAULTS as env_d
        from src.parser.optimizer import DEFAULTS as opt_d
        from src.parser.reward import DEFAULTS as rew_d
        from src.parser.search import DEFAULTS as search_d
        from src.parser.tokenizer import DEFAULTS as tok_d
        from src.parser.transformer import DEFAULTS as trans_d
        from src.parser.ucb import DEFAULTS as ucb_d

        for d in (agent_d, env_d, opt_d, rew_d, search_d, tok_d, trans_d, ucb_d):
            defaults.update(d)
    except ImportError:
        pass

    protocol_path = full_config.get("lemmas/0/theory_path", "")
    if protocol_path:
        protocol_path = os.path.dirname(protocol_path)
    heuristic = full_config.get("lemmas/0/heuristic", "s")

    parts = ["python -m src.main.run train"]
    if protocol_path:
        parts.append(f"--protocol={protocol_path}")
    if heuristic != "f":
        parts.append(f"--heuristic={heuristic}")

    for wandb_key, value in sorted(full_config.items()):
        if wandb_key.startswith("lemmas/") or wandb_key.startswith("_"):
            continue
        if wandb_key in _CLI_SKIP_KEYS:
            continue

        cli_name = _WANDB_TO_CLI.get(wandb_key)
        if cli_name is None:
            cli_name = wandb_key.split("/", 1)[-1]

        if value is None or str(value) in ("None", ""):
            continue

        default = defaults.get(cli_name)
        if default is not None:
            try:
                if type(default)(value) == default:
                    continue
            except (ValueError, TypeError):
                pass

        if isinstance(value, bool):
            if value:
                parts.append(f"--{cli_name}")
        else:
            parts.append(f"--{cli_name}={value}")

    return " ".join(parts)


def generate_hyperparameters_csv(
    all_runs: dict[str, dict], output_path: str | Path,
    best_only: bool = True,
) -> None:
    """Write a hyperparameters CSV.

    When *best_only* is True (default), only the best run per protocol
    plus any extra runs that solve additional lemmas are included.
    When False, all runs are included.
    """
    protocols: dict[str, list[tuple[str, dict]]] = {}
    for run_id, run_data in all_runs.items():
        for key in run_data:
            if key.startswith("_"):
                continue
            protocols.setdefault(key, []).append((run_id, run_data[key]))

    rows: list[dict] = []
    for proto, runs in sorted(protocols.items()):
        if not best_only:
            for run_id, lemmas in runs:
                completed = sum(1 for m in lemmas.values() if is_completed(m))
                full_config = all_runs[run_id].get("_meta", {}).get("config", {})
                config = {
                    k: v for k, v in full_config.items()
                    if not k.startswith("lemmas/") and k not in _BEST_RUNS_EXCLUDE_KEYS
                }
                rows.append({
                    "protocol": proto,
                    "run_id": run_id,
                    "completed_lemmas": completed,
                    "command": build_reproduce_command(full_config),
                    **config,
                })
            continue

        best_id, best_count = None, -1
        for run_id, lemmas in runs:
            completed = sum(1 for m in lemmas.values() if is_completed(m))
            if completed > best_count:
                best_count = completed
                best_id = run_id

        if best_id is None:
            continue

        full_config = all_runs[best_id].get("_meta", {}).get("config", {})
        config = {
            k: v for k, v in full_config.items()
            if not k.startswith("lemmas/") and k not in _BEST_RUNS_EXCLUDE_KEYS
        }
        rows.append({
            "protocol": proto,
            "run_id": best_id,
            "completed_lemmas": best_count,
            "command": build_reproduce_command(full_config),
            **config,
        })

        best_solved = {
            l for l, m in all_runs[best_id][proto].items()
            if is_completed(m)
        }
        for run_id, lemmas in runs:
            if run_id == best_id:
                continue
            solved = {
                l for l, m in lemmas.items() if is_completed(m)
            }
            if solved - best_solved:
                extra_full_config = all_runs[run_id].get("_meta", {}).get("config", {})
                cfg = {
                    k: v for k, v in extra_full_config.items()
                    if not k.startswith("lemmas/") and k not in _BEST_RUNS_EXCLUDE_KEYS
                }
                extra_count = sum(
                    1 for m in lemmas.values() if is_completed(m)
                )
                rows.append({
                    "protocol": proto,
                    "run_id": run_id,
                    "completed_lemmas": extra_count,
                    "command": build_reproduce_command(extra_full_config),
                    **cfg,
                })

    if not rows:
        return

    all_keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    output_path = Path(output_path)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {output_path} ({len(rows)} runs, {len(all_keys)} columns)")


def generate_best_runs_csv(results_dir: str, output_path: str | Path) -> None:
    """Legacy wrapper: aggregate all raw.json files and write a single CSV."""
    all_runs: dict[str, dict] = {}
    for raw_path in sorted(Path(results_dir).glob("*/raw.json")):
        if raw_path.parent.name == "all":
            continue
        with open(raw_path) as f:
            data = json.load(f)
        for run_id, run_data in data.items():
            all_runs.setdefault(run_id, {}).update(run_data)
    generate_hyperparameters_csv(all_runs, output_path)
