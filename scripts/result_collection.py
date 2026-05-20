#!/usr/bin/env python3
from __future__ import annotations

"""Result collection script — equivalent of scripts/result_collection.ipynb.

Run from the project root. Steps are presented interactively; use -y / --yes
to accept all of them automatically.

Usage:
    python scripts/result_collection.py          # interactive
    python scripts/result_collection.py -y       # auto-yes all steps
    python scripts/result_collection.py -y --no-cache  # skip cached raw.json
"""

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


try:
    import wandb
except ImportError:
    wandb = None

from results_lib.aggregation import aggregate_runs, format_summary
from results_lib.baselines import (
    DummyRun,
    build_folder_to_theory_mapping,
    extract_theory_name,
    load_csv_data,
    load_csv_results,
)
from results_lib.markdown import (
    generate_lemma_completion_by_type,
    generate_lemma_completion_overview,
    generate_proofsize_overview,
)
from results_lib.plots import plot_heatmap
from results_lib.reproduce import generate_hyperparameters_csv
from results_lib.tables import (
    OUTPUT_FILES,
    generate_paper_tables,
    write_preserving_caption,
)
from results_lib.utils import is_completed, parse_time_to_seconds

# ---------------------------------------------------------------------------
# Step helper
# ---------------------------------------------------------------------------


def ask(prompt: str, yes_all: bool) -> bool:
    if yes_all:
        print(f"[auto] {prompt}")
        return True
    return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")


# ---------------------------------------------------------------------------
# Step 1 — fetch / refresh run IDs
# ---------------------------------------------------------------------------


def fetch_run_ids(
    api: wandb.Api, regex: str = r"4\.0", existing_ids: set[str] | None = None
) -> pd.DataFrame:
    pattern = re.compile(regex, re.IGNORECASE)
    runs = []
    recent_runs = api.runs("matcos/tamarin-rl", order="-created_at", per_page=50)
    for i, r in enumerate(tqdm(recent_runs, total=200, desc="Scanning recent runs")):
        if i >= 200:
            break
        if pattern.search(r.display_name or r.name):
            runs.append(r)
    print(f"  Scanned {min(i + 1, 200)} recent runs, {len(runs)} match /{regex}/i")
    known = {r.id for r in runs}
    missed = (existing_ids or set()) - known
    if missed:
        print(f"  Checking {len(missed)} previously known IDs not in recent batch...")
        for rid in missed:
            try:
                r = api.run(f"matcos/tamarin-rl/{rid}")
                if pattern.search(r.display_name or r.name):
                    runs.append(r)
            except Exception:
                pass
    print(f"  Found {len(runs)} runs matching /{regex}/i")

    protocols_with_base_s = {
        p.parent.name for p in Path("eval").glob("*/base_s") if p.is_dir()
    }

    rows = []
    for run in runs:
        config = run.config

        theory_paths = [
            v
            for k, v in config.items()
            if k.endswith("/theory_path") and isinstance(v, str)
        ]
        subfolders: set[str] = set()
        variants: set[str] = set()
        for tp in theory_paths:
            parts = tp.split("/")
            for i, p in enumerate(parts):
                if p == "eval" and i + 1 < len(parts):
                    subfolders.add(parts[i + 1])
                    if i + 2 < len(parts):
                        variants.add(parts[i + 2])
                    break
            else:
                if len(parts) > 1:
                    subfolders.add(parts[1])
                if len(parts) > 2:
                    variants.add(parts[2])

        subfolder = ", ".join(sorted(subfolders)) if subfolders else "N/A"

        heuristic_weight = config.get("ucb/heuristic_weight", 0.0)
        has_heuristic = heuristic_weight is not None and heuristic_weight > 0

        if has_heuristic and variants:
            renamed: set[str] = set()
            for v in variants:
                if v == "original" and not subfolders & protocols_with_base_s:
                    renamed.add("base_s")
                else:
                    renamed.add(v)
            heuristic = ", ".join(sorted(renamed))
        else:
            heuristic = "no"

        deactivate_model = config.get("ucb/deactivate_model", False)

        slurm_job_id = config.get("main/slurm_id")

        rows.append(
            {
                "run_name": run.name,
                "run_id": run.id,
                "state": run.state,
                "protocol": subfolder,
                "heuristic": heuristic,
                "model_deactivated": deactivate_model,
                "slurm_job_id": slurm_job_id,
            }
        )

    return pd.DataFrame(rows)


def step_refresh_run_ids(
    api: wandb.Api, regex: str, results_dir: str, yes: bool
) -> pd.DataFrame:
    RUN_IDS_PATH = f"{results_dir}/run_ids.csv"

    if os.path.exists(RUN_IDS_PATH):
        df = pd.read_csv(RUN_IDS_PATH)
        print(f"Loaded {len(df)} runs from {RUN_IDS_PATH}")
    else:
        df = pd.DataFrame()
        print(f"No existing file at {RUN_IDS_PATH}, starting fresh.")

    if ask("Refresh run IDs from W&B?", yes):
        existing_ids = set(df["run_id"]) if not df.empty else None
        df_fresh = fetch_run_ids(api, regex, existing_ids=existing_ids)
        print(f"Fetched {len(df_fresh)} runs from W&B")
        if not df.empty:
            df = (
                pd.concat([df, df_fresh])
                .drop_duplicates(subset="run_id", keep="last")
                .reset_index(drop=True)
            )
        else:
            df = df_fresh
        os.makedirs(os.path.dirname(RUN_IDS_PATH), exist_ok=True)
        df.to_csv(RUN_IDS_PATH, index=False)
        print(f"Saved {len(df)} runs to {RUN_IDS_PATH}")

    print(df.to_string())
    return df


# ---------------------------------------------------------------------------
# Step 2 — load run objects
# ---------------------------------------------------------------------------


def step_load_runs(
    api: wandb.Api, df_run_ids: pd.DataFrame, yes: bool
) -> tuple[dict[str, dict], set[str]]:
    runs: dict[str, dict] = {}
    failed_ids: set[str] = set()
    if not ask("Load run objects from W&B?", yes):
        return runs, failed_ids

    for (heuristic, model_deactivated), group in df_run_ids.groupby(
        ["heuristic", "model_deactivated"]
    ):
        config_name = f"{heuristic}_{'no_model' if model_deactivated else 'model'}"
        runs[config_name] = {}
        for _, row in group.iterrows():
            try:
                run = api.run(f"matcos/tamarin-rl/{row['run_id']}")
                if not run.name:
                    print(f"  Skipping run {row['run_id']} (not found)")
                    failed_ids.add(row["run_id"])
                    continue
                print(
                    f"  Loaded run: {run.name} (ID: {run.id})"
                    f" -> config: {config_name}"
                )
                runs[config_name][run.id] = run
            except Exception as e:
                print(f"  Skipping run {row['run_id']}: {e}")
                failed_ids.add(row["run_id"])

    for config_name, config_runs in runs.items():
        print(f"  {config_name}: {len(config_runs)} runs")
    total = sum(len(v) for v in runs.values())
    print(f"Total configs: {len(runs)}, total runs: {total}")
    return runs, failed_ids


# ---------------------------------------------------------------------------
# Step 4 — refresh metrics, aggregate, save summaries
# ---------------------------------------------------------------------------


def get_from_history(
    run, key_completes: str, key_tree: str, key_steps: str | None = None
) -> tuple[float, int, int | None]:
    scan_keys = ["_step", "_runtime", key_completes, key_tree]
    if key_steps:
        scan_keys.append(key_steps)
    history = pd.DataFrame(
        list(run.scan_history(keys=scan_keys, page_size=20000))
    )
    assert not history.empty, f"History empty for run {run.id}"
    assert "_step" in history.columns and "_runtime" in history.columns
    history = history.sort_values("_step")
    history["_runtime"] = history["_runtime"].ffill().bfill()

    col_completes_vals = pd.to_numeric(history[key_completes], errors="coerce")
    rows = history.loc[~col_completes_vals.isna() & (col_completes_vals > 0)]
    assert not rows.empty, f"No completed lemmas for run {run.id}"
    rt = rows.iloc[0]["_runtime"]
    assert pd.notna(rt), f"Runtime is NaN for run {run.id}"

    first_solved = round(float(rt), 1)
    col_tree_vals = pd.to_numeric(history[key_tree], errors="coerce")
    min_tree_size = int(col_tree_vals.min())

    first_solve_steps = None
    if key_steps and key_steps in history.columns:
        step_val = rows.iloc[0].get(key_steps)
        if pd.notna(step_val):
            first_solve_steps = int(step_val)

    return first_solved, min_tree_size, first_solve_steps


def _needs_refresh(raw_data: dict, run_id: str) -> bool:
    if run_id not in raw_data:
        return True
    meta = raw_data[run_id].get("_meta")
    if meta is None:
        return True
    return meta.get("state") in ["running", "starting"]


def refresh(
    run_iterable,
    raw_data: dict[str, dict[str, dict[str, dict[str, float | int | str]]]],
) -> dict[str, Any]:
    run_map: dict[str, Any] = {}

    for run in run_iterable:
        run_map[run.id] = run

        if not _needs_refresh(raw_data, run.id):
            continue

        raw_data[run.id] = {
            "_meta": {
                "name": run.name,
                "state": run.state,
                "config": dict(run.config),
            }
        }
        search_metrics = {
            k.split("Search/", 1)[1]: v
            for k, v in run.summary.items()
            if k.startswith("Search/")
        }
        theories = set(k.split("/")[0] for k in search_metrics.keys())

        for theory in theories:
            raw_data[run.id][theory] = {}
            theory_indices = [
                k.split("/")[1]
                for k, v in run.config.items()
                if k.startswith("lemmas/")
                and k.endswith("/theory_name")
                and v == theory
            ]
            completes = {
                k.split("/")[2]: v
                for k, v in run.summary.items()
                if k.startswith("Search/")
                and k.endswith("/lemma_completes")
                and k.split("/")[1] == theory
            }

            for lemma, count_completes in tqdm(
                completes.items(),
                desc=(
                    f"Refreshing run: {run.name}"
                    f" (ID: {run.id}) with state: {run.state}"
                ),
                unit="lemma",
            ):
                raw_data[run.id][theory][lemma] = {"completes": count_completes}
                if count_completes > 0:
                    col_completes = f"Search/{theory}/{lemma}/lemma_completes"
                    col_tree = f"Search/{theory}/{lemma}/tree_size"
                    col_steps = f"Search/{theory}/{lemma}/steps"
                    first_solved, min_tree_size, first_solve_steps = (
                        get_from_history(
                            run, col_completes, col_tree, col_steps
                        )
                    )
                    raw_data[run.id][theory][lemma][
                        "first_completion_time"
                    ] = first_solved
                    raw_data[run.id][theory][lemma]["tree_size"] = min_tree_size
                    if first_solve_steps is not None:
                        raw_data[run.id][theory][lemma][
                            "first_solve_steps"
                        ] = first_solve_steps

                indices = [
                    k.split("/")[1]
                    for k, v in run.config.items()
                    if k.startswith("lemmas/")
                    and k.endswith("/lemma_name")
                    and v == lemma
                    and k.split("/")[1] in theory_indices
                ]
                assert len(indices) == 1
                raw_data[run.id][theory][lemma]["type"] = run.config[
                    f"lemmas/{indices[0]}/lemma_type"
                ]

    return run_map


def write_back(raw_data: dict, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(raw_data, f, indent=2, default=int)


def step_refresh_metrics(
    runs: dict[str, dict], results_dir: str, load_cached: bool, prune: bool, yes: bool
) -> None:
    if not runs or not ask("Refresh metrics from W&B?", yes):
        return

    for config, config_runs in runs.items():
        filename = f"{results_dir}/{config}/raw.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if os.path.exists(filename) and load_cached:
            print(f"Loading cached raw data for {config} from {filename}")
            with open(filename, "r") as f:
                raw_data = json.load(f)
        else:
            print(f"Starting fresh raw data for {config}.")
            raw_data = {}

        run_map = refresh(config_runs.values(), raw_data)

        if prune:
            stale = [rid for rid in raw_data if rid not in run_map]
            for rid in stale:
                print(f"  Pruning stale run {rid} from {filename}")
                del raw_data[rid]

        write_back(raw_data, filename)
        print(f"  Updated {filename}")


def _load_and_aggregate(
    results_dir: str,
    baseline_data: dict | None = None,
    baseline_runs: dict | None = None,
) -> tuple[dict, dict, dict[str, dict]]:
    """Load raw.json files and aggregate metrics. Does not write anything.

    Returns (summaries, heatmaps, loaded_raw) where loaded_raw maps
    config name -> parsed raw.json dict.
    """
    if baseline_data is None or baseline_runs is None:
        baseline_data, baseline_runs = load_csv_data("results/tamarin")

    raw_paths = sorted(Path(results_dir).glob("*/raw.json"))
    if not raw_paths:
        print(f"  No raw.json files found in {results_dir}.")
        return {}, {}, {}

    summaries: dict = {"all": {}}
    heatmaps: dict = {"all": {}}
    loaded_raw: dict[str, dict] = {}

    for raw_path in raw_paths:
        config = raw_path.parent.name
        with open(raw_path) as f:
            raw_data: dict = json.load(f)
        loaded_raw[config] = raw_data

        augmented = dict(raw_data)
        run_map: dict = {}
        for rid, rdata in raw_data.items():
            meta = rdata.get("_meta")
            if meta:
                run_map[rid] = DummyRun(meta["name"])

        for rid, rdata in baseline_data.items():
            augmented[rid] = rdata
            run_map[rid] = baseline_runs[rid]

        summaries[config] = {}
        heatmaps[config] = {}

        if "original" in config or "partial_" in config or config == "single":
            summary_targets = [summaries[config]]
            heatmap_targets = [heatmaps[config]]
        else:
            summary_targets = [summaries[config], summaries["all"]]
            heatmap_targets = [heatmaps[config], heatmaps["all"]]

        aggregate_runs(augmented, summary_targets, heatmap_targets, run_map)
        summaries[config] = format_summary(summaries[config])

    summaries["all"] = format_summary(summaries["all"])

    return summaries, heatmaps, loaded_raw


def _write_summaries_and_csvs(
    results_dir: str,
    summaries: dict,
    loaded_raw: dict[str, dict],
) -> None:
    """Write summary.yaml and hyperparameters.csv files."""
    for config, summary_data in summaries.items():
        filename = f"{results_dir}/{config}/summary.yaml"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            yaml.dump(
                summary_data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        print(f"  Saved {filename}")

    all_runs_combined: dict = {}
    for config, raw_data in loaded_raw.items():
        generate_hyperparameters_csv(
            raw_data,
            f"{results_dir}/{config}/hyperparameters.csv",
            best_only=False,
        )
        for run_id, run_data in raw_data.items():
            all_runs_combined.setdefault(run_id, {}).update(run_data)
    if all_runs_combined:
        generate_hyperparameters_csv(
            all_runs_combined,
            f"{results_dir}/all/hyperparameters.csv",
            best_only=False,
        )


def step_aggregate_and_split(
    results_dir: str,
    output_dir: str,
    config_names: list[str],
    yes: bool,
) -> tuple[dict, dict]:
    if not ask("Aggregate, split best/others, and save summaries?", yes):
        return {}, {}

    print("  Loading input folder...")
    baseline_data, baseline_runs = load_csv_data("results/tamarin")
    summaries, heatmaps, loaded_raw = _load_and_aggregate(
        results_dir,
        baseline_data,
        baseline_runs,
    )

    print("  Splitting best/others...")
    pooled: dict[str, dict] = {}
    for config_name in config_names:
        if config_name not in loaded_raw:
            print(f"  Skipping {config_name}: raw.json not found")
            continue
        for run_id, run_data in loaded_raw[config_name].items():
            name = run_data.get("_meta", {}).get("name", "")
            if "single" in name:
                continue
            pooled.setdefault(run_id, {}).update(run_data)

    if not pooled:
        print("  No runs found in specified configs.")
        return summaries, heatmaps

    best_per_proto = _find_best_per_protocol(pooled)
    best_run_ids: set[str] = set()
    extra_run_ids: set[str] = set()

    for proto, (best_id, _, _) in best_per_proto.items():
        best_run_ids.add(best_id)
        best_solved = {
            l for l, m in pooled[best_id].get(proto, {}).items() if is_completed(m)
        }
        for run_id, run_data in pooled.items():
            if run_id == best_id or proto not in run_data:
                continue
            solved = {l for l, m in run_data[proto].items() if is_completed(m)}
            if solved - best_solved:
                extra_run_ids.add(run_id)

    # Find runs that achieve the best time or tree size for any lemma
    best_at_ids: set[str] = set()
    all_ids = set(pooled.keys())
    all_protos = {p for rid in all_ids for p in pooled[rid] if p != "_meta"}
    for proto in all_protos:
        lemma_best_time: dict[str, tuple[float, str]] = {}
        lemma_best_size: dict[str, tuple[int, str]] = {}
        for run_id in all_ids:
            if proto not in pooled[run_id]:
                continue
            for lemma, m in pooled[run_id][proto].items():
                if not is_completed(m):
                    continue
                t = m.get("first_completion_time", float("inf"))
                if lemma not in lemma_best_time or t < lemma_best_time[lemma][0]:
                    lemma_best_time[lemma] = (t, run_id)
                s = m.get("tree_size")
                if s is not None and (
                    lemma not in lemma_best_size or s < lemma_best_size[lemma][0]
                ):
                    lemma_best_size[lemma] = (s, run_id)
        for _, rid in lemma_best_time.values():
            best_at_ids.add(rid)
        for _, rid in lemma_best_size.values():
            best_at_ids.add(rid)

    keep_ids = best_run_ids | extra_run_ids | best_at_ids
    subsumed_ids: set[str] = set()

    for run_id in all_ids - keep_ids:
        for proto in pooled[run_id]:
            if proto == "_meta":
                continue
            run_lemmas = pooled[run_id][proto]
            for other_id in all_ids:
                if other_id == run_id:
                    continue
                if proto not in pooled[other_id]:
                    continue
                if _is_subsumed(run_lemmas, pooled[other_id][proto]):
                    subsumed_ids.add(run_id)
                    break
            if run_id in subsumed_ids:
                break

    others_ids = (
        extra_run_ids | best_at_ids | (all_ids - best_run_ids - subsumed_ids)
    ) - best_run_ids

    best_data = {rid: pooled[rid] for rid in best_run_ids if rid in pooled}
    others_data = {rid: pooled[rid] for rid in others_ids if rid in pooled}

    for subdir, data in [("best", best_data), ("others", others_data)]:
        out = Path(output_dir) / subdir
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "raw.json", "w") as f:
            json.dump(data, f, indent=2, default=int)
        print(f"  Wrote {out / 'raw.json'} ({len(data)} runs)")

    only_best_at = best_at_ids - best_run_ids - extra_run_ids
    print(
        f"  Best: {len(best_run_ids)} runs, extra: {len(extra_run_ids)},"
        f" best-at: {len(only_best_at)},"
        f" subsumed: {len(subsumed_ids)}, others: {len(others_ids)}"
    )

    return summaries, heatmaps


def step_aggregate_folder(folder: str, yes: bool) -> None:
    if not ask("Aggregate folder (summaries + hyperparameters)?", yes):
        return
    summaries, _, loaded_raw = _load_and_aggregate(folder)
    _write_summaries_and_csvs(folder, summaries, loaded_raw)


def _find_best_per_protocol(
    pooled: dict[str, dict],
) -> dict[str, tuple[str, int, float]]:
    """Return {protocol: (run_id, completed_count, max_time)} for the best run."""
    protocols: dict[str, list[tuple[str, dict]]] = {}
    for run_id, run_data in pooled.items():
        for key in run_data:
            if key == "_meta":
                continue
            protocols.setdefault(key, []).append((run_id, run_data[key]))

    best: dict[str, tuple[str, int, float]] = {}
    for proto, runs in protocols.items():
        best_id, best_count, best_time = None, -1, 0.0
        for run_id, lemmas in runs:
            completed = sum(1 for m in lemmas.values() if is_completed(m))
            max_time = max(
                (
                    m.get("first_completion_time", 0.0)
                    for m in lemmas.values()
                    if is_completed(m)
                ),
                default=0.0,
            )
            if completed > best_count or (
                completed == best_count and max_time < best_time
            ):
                best_id, best_count, best_time = run_id, completed, max_time
        if best_id is not None:
            best[proto] = (best_id, best_count, best_time)
    return best


def _is_subsumed(run_lemmas: dict, other_lemmas: dict) -> bool:
    """Return True if run_lemmas is strictly dominated by other_lemmas."""
    run_solved = {l for l, m in run_lemmas.items() if is_completed(m)}
    other_solved = {l for l, m in other_lemmas.items() if is_completed(m)}
    if not run_solved.issubset(other_solved):
        return False
    if run_solved == other_solved:
        strictly_worse = False
        for l in run_solved:
            rt = run_lemmas[l].get("first_completion_time", 0.0)
            ot = other_lemmas[l].get("first_completion_time", 0.0)
            rs = run_lemmas[l].get("tree_size", 0)
            os_ = other_lemmas[l].get("tree_size", 0)
            if rt < ot or (rs is not None and os_ is not None and rs < os_):
                return False
            if rt > ot or (rs is not None and os_ is not None and rs > os_):
                strictly_worse = True
        return strictly_worse
    return True


# ---------------------------------------------------------------------------
# Step 5 — fetch proof trees + checkpoints from remote
# ---------------------------------------------------------------------------


_SCP_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=60"]


def step_fetch_artifacts(
    output_dir: str,
    login: str,
    remote_base: str,
    no_cache: bool,
    yes: bool,
    model_run: str | None = None,
) -> None:
    if not ask("Fetch proof trees + checkpoints from cluster?", yes):
        return

    for subdir_name in ("best", "others"):
        raw_path = Path(output_dir) / subdir_name / "raw.json"
        if not raw_path.exists():
            print(f"  {raw_path} not found.")
            continue
        with open(raw_path) as f:
            raw_data = json.load(f)

        run_ids = [rid for rid in raw_data if not rid.startswith("_")]
        print(f"  [{subdir_name}] {len(run_ids)} runs to fetch")

        for run_id in run_ids:
            local_dir = Path(output_dir) / subdir_name / "artifacts" / run_id
            local_dir.mkdir(parents=True, exist_ok=True)

            meta = raw_data[run_id].get("_meta", {})
            still_running = meta.get("state") in ("running", "starting")

            fetch_model = model_run is None or run_id == model_run

            items_to_fetch = []
            if not (local_dir / "proof_trees").is_dir() or no_cache or still_running:
                items_to_fetch.append(("proof_trees", True))
            if fetch_model and (not (local_dir / "ckp_last.pt").exists() or no_cache or still_running):
                items_to_fetch.append(("ckp_last.pt", False))
            if fetch_model and (not (local_dir / "model_last.pt").exists() or no_cache or still_running):
                items_to_fetch.append(("model_last.pt", False))

            if not items_to_fetch:
                print(f"    {run_id}: already fetched, skipping")
                continue

            fetched = []
            for item, is_dir in items_to_fetch:
                cmd = ["scp", *_SCP_OPTS]
                if is_dir:
                    cmd.append("-r")
                cmd.extend(
                    [
                        f"{login}:{remote_base}/{run_id}/{item}",
                        str(local_dir / item),
                    ]
                )
                r = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if r.returncode == 0:
                    print(f"      {run_id}: OK {item}")
                    fetched.append(item)
                else:
                    err = (
                        r.stderr.strip().split("\n")[-1]
                        if r.stderr
                        else "unknown error"
                    )
                    print(f"      {run_id}: FAIL {item} — {err}")

            if not fetched:
                print(f"    {run_id}: not found on remote")
                continue

            print(f"    {run_id}: fetched {', '.join(fetched)}")


# ---------------------------------------------------------------------------
# Step 6 — generate all tables (md + tex)
# ---------------------------------------------------------------------------


def _merge_proto_yaml(dst: dict, src: dict) -> None:
    """Merge per-protocol yaml data, keeping best min_time/min_tree_size per lemma."""
    src_lemmas = src.get("lemmas", {})
    dst_lemmas = dst.setdefault("lemmas", {})
    for lemma, sdata in src_lemmas.items():
        if lemma not in dst_lemmas:
            dst_lemmas[lemma] = sdata
            continue
        ddata = dst_lemmas[lemma]
        st = parse_time_to_seconds(sdata.get("min_time"))
        dt = parse_time_to_seconds(ddata.get("min_time"))
        if st is not None and (dt is None or st < dt):
            ddata["min_time"] = sdata["min_time"]
        ss = sdata.get("min_tree_size")
        ds = ddata.get("min_tree_size")
        if ss is not None and (ds is None or ss < ds):
            ddata["min_tree_size"] = ss
        ddata["count"] = ddata.get("count", 0) + sdata.get("count", 0)


def _generate_all_tables(results_dir: str, baselines_dir: str) -> None:
    """Generate per-config markdown tables and paper tables for a results folder."""
    csv_data = load_csv_results(baselines_dir)
    if not csv_data:
        print("  Warning: no CSV baseline data found.")

    configs = [p.parent.name for p in Path(results_dir).glob("*/summary.yaml")]
    if not configs:
        print(f"  No summary.yaml files found in {results_dir}.")
        return

    all_yaml: dict = {}
    all_raw: dict = {}
    best_raw: dict = {}

    for config in sorted(configs):
        summary_path = f"{results_dir}/{config}/summary.yaml"
        with open(summary_path) as f:
            yaml_data = yaml.safe_load(f) or {}

        raw_path = f"{results_dir}/{config}/raw.json"
        if os.path.exists(raw_path):
            with open(raw_path) as f:
                raw_data = json.load(f)
        else:
            raw_data = {}
            for rp in Path(results_dir).glob("*/raw.json"):
                sibling = rp.parent.name
                if "original" in sibling or "partial_" in sibling:
                    continue
                with open(rp) as f:
                    raw_data.update(json.load(f))

        if config in ("best", "others"):
            best_raw.update(raw_data)

        if config != "all":
            for proto, proto_data in yaml_data.items():
                if proto not in all_yaml:
                    all_yaml[proto] = proto_data
                else:
                    _merge_proto_yaml(all_yaml[proto], proto_data)
            for run_id, run_data in raw_data.items():
                all_raw[run_id] = run_data

        tables_dir = Path(f"{results_dir}/{config}/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)

        generate_lemma_completion_overview(
            yaml_data, csv_data, tables_dir / "lemma_completion_overview.md"
        )
        generate_lemma_completion_by_type(
            yaml_data, csv_data, raw_data, tables_dir / "lemma_completion_by_type.md"
        )
        generate_proofsize_overview(
            yaml_data, csv_data, tables_dir / "proofsize_overview.md"
        )
        generate_paper_tables(yaml_data, raw_data, csv_data, tables_dir)
        print(f"  [{config}] tables -> {tables_dir}")

    all_tables = Path(results_dir) / "all" / "tables"
    all_tables.mkdir(parents=True, exist_ok=True)
    generate_paper_tables(
        all_yaml, all_raw, csv_data, all_tables, best_raw=best_raw or None
    )


# ---------------------------------------------------------------------------
# Step 7 — insert tex files into paper folder
# ---------------------------------------------------------------------------


def step_insert_paper_tables(
    output_dir: str,
    paper_dir: str,
    yes: bool,
) -> None:
    if not ask("Insert fresh tex files into paper folder?", yes):
        return

    src = Path(output_dir) / "all" / "tables"
    dst = Path(paper_dir)
    dst.mkdir(parents=True, exist_ok=True)

    paper_files = set(OUTPUT_FILES.values())
    tex_files = [f for f in sorted(src.glob("*.tex")) if f.name in paper_files]
    if not tex_files:
        print(f"  No paper .tex files found in {src}")
        return

    for tex_file in tex_files:
        target = dst / tex_file.name
        content = tex_file.read_text()
        if target.exists():
            write_preserving_caption(content, target)
        else:
            target.write_text(content)
        print(f"  {tex_file.name} -> {target}")


# ---------------------------------------------------------------------------
# Steps 8–9 — plot heatmaps
# ---------------------------------------------------------------------------


def _load_heatmaps(
    results_dir: str,
    baseline_data: dict | None = None,
    baseline_runs: dict | None = None,
) -> dict:
    """Reconstruct heatmaps from saved raw.json files (no W&B needed)."""
    if baseline_data is None or baseline_runs is None:
        baseline_data, baseline_runs = load_csv_data("results/tamarin")

    raw_paths = sorted(Path(results_dir).glob("*/raw.json"))
    if not raw_paths:
        print(f"  No raw.json files found in {results_dir}.")
        return {}

    heatmaps: dict = {"all": {}}
    for raw_path in raw_paths:
        config = raw_path.parent.name
        with open(raw_path) as f:
            raw_data: dict = json.load(f)

        run_map: dict = {}
        for rid, rdata in raw_data.items():
            meta = rdata.get("_meta")
            if meta:
                run_map[rid] = DummyRun(meta["name"])
        for rid, rdata in baseline_data.items():
            raw_data[rid] = rdata
            run_map[rid] = baseline_runs[rid]

        heatmaps[config] = {}
        if "original" in config or "partial_" in config or config == "single":
            targets = [heatmaps[config]]
        else:
            targets = [heatmaps[config], heatmaps["all"]]

        aggregate_runs(raw_data, [{}], targets, run_map)

    return heatmaps


def _plot_heatmaps(heatmaps: dict, results_dir: str) -> None:
    for config, heatmap_data in heatmaps.items():
        plots_dir = f"{results_dir}/{config}/plots"
        os.makedirs(plots_dir, exist_ok=True)
        for name, data in heatmap_data.items():
            if all("baseline" in k[1] for k in data):
                continue
            plot_heatmap(
                data,
                title=f"{name} ({config}) - Completes",
                file=f"{plots_dir}/heatmap_{name}.png",
                show=False,
            )
            plot_heatmap(
                data,
                title=f"{name} ({config}) - Time to Completion",
                file=f"{plots_dir}/heatmap_{name}_time.png",
                show=False,
                key="first_completion_time",
                zmin=0,
                zmax=None,
                zero_red=False,
            )
            plot_heatmap(
                data,
                title=f"{name} ({config}) - Tree Size",
                file=f"{plots_dir}/heatmap_{name}_tree.png",
                show=False,
                key="tree_size",
                zmin=0,
                zmax=300,
                zero_red=False,
            )
            print(f"  Saved heatmaps for {name} ({config})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


WANDB_STEPS = {1, 2}

SPLIT_CONFIGS = ["base_s_model", "no_model"]

STEPS = {
    1: "[wandb] Fetch / refresh run IDs",
    2: "[wandb] Sync runs, refresh metrics",
    3: "[local] Split best/others from input to folder",
    4: "[local] Aggregate folder (summaries + hyperparameters)",
    5: "[ssh]   Fetch proof trees + checkpoints",
    6: "[local] Generate tables",
    7: "[local] Insert fresh tex into paper folder",
    8: "[local] Generate plots",
}


def _parse_step_range(spec: str, valid: set[int]) -> set[int]:
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    invalid = result - valid
    if invalid:
        raise SystemExit(f"Invalid step(s): {sorted(invalid)}. Valid: {sorted(valid)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect and process W&B training results."
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Auto-yes all steps")
    parser.add_argument(
        "--filter", default=None, help="W&B run-name regex filter (e.g. '4\\.0')"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Results folder (e.g. results)",
    )
    parser.add_argument(
        "--paper-dir",
        type=str,
        default=None,
        help="Paper folder for tex insertion (e.g. ../paper/tables)",
    )
    parser.add_argument(
        "--remote",
        type=str,
        default=None,
        help="SSH remote for artifact fetch (e.g. user@host:path/to/output)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached raw.json / artifacts and re-fetch",
    )
    parser.add_argument(
        "--model-run",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="Only download model/checkpoint for this run ID (proof trees for all)",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Remove raw.json entries for runs no longer returned by W&B",
    )
    steps_help = ", ".join(f"{k}={v}" for k, v in STEPS.items())
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        metavar="RANGE",
        help=(
            f"Steps to run (e.g. '3-6', '3,5', '3-4,6'). "
            f"Default: all. Available: {steps_help}"
        ),
    )
    args = parser.parse_args()

    active = set(STEPS.keys())
    if args.steps:
        active = _parse_step_range(args.steps, set(STEPS.keys()))

    def run(n: int) -> bool:
        return n in active

    needs_filter = active & {1, 2, 3}
    if needs_filter and not args.filter:
        raise SystemExit(
            f"--filter is required for step(s) {sorted(needs_filter)}"
        )

    results_dir = None
    if args.filter:
        filter_label = args.filter.replace("\\", "")
        results_dir = f"results-{filter_label}"

    needs_folder = active & {3, 4, 5, 6, 7, 8}
    if needs_folder and not args.folder:
        raise SystemExit(f"--folder is required for step(s) {sorted(needs_folder)}")
    if run(7) and not args.paper_dir:
        raise SystemExit("--paper-dir is required for step 7")
    if run(5) and not args.remote:
        raise SystemExit("--remote is required for step 5")

    if active & WANDB_STEPS:
        if pd is None:
            raise SystemExit("pandas is required for steps 1-2")
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 2000)
        pd.set_option("display.max_colwidth", None)

    api = None
    if active & WANDB_STEPS:
        if wandb is None:
            raise SystemExit(
                "wandb is required for steps 1-2. Install with: pip install wandb"
            )
        api = wandb.Api(timeout=60)

    df_run_ids = None
    if run(1) and api is not None:
        df_run_ids = step_refresh_run_ids(api, args.filter, results_dir, args.yes)

    if run(2) and api is not None:
        if df_run_ids is None:
            path = f"{results_dir}/run_ids.csv"
            if not os.path.exists(path):
                print(f"  {path} not found — run step 1 first.")
            else:
                df_run_ids = pd.read_csv(path)
        if df_run_ids is not None:
            runs, failed_ids = step_load_runs(api, df_run_ids, args.yes)
            if args.prune and failed_ids and df_run_ids is not None:
                before = len(df_run_ids)
                df_run_ids = df_run_ids[~df_run_ids["run_id"].isin(failed_ids)]
                run_ids_path = f"{results_dir}/run_ids.csv"
                df_run_ids.to_csv(run_ids_path, index=False)
                print(
                    f"  Pruned {before - len(df_run_ids)} stale entries"
                    f" from {run_ids_path}"
                )
            if runs:
                step_refresh_metrics(
                    runs,
                    results_dir,
                    load_cached=not args.no_cache,
                    prune=args.prune,
                    yes=args.yes,
                )

    if run(3):
        step_aggregate_and_split(
            results_dir,
            args.folder,
            SPLIT_CONFIGS,
            args.yes,
        )

    if run(4):
        step_aggregate_folder(args.folder, args.yes)

    if run(5):
        if ":" not in args.remote:
            raise SystemExit(
                "--remote must be in the format user@host:path/to/output"
            )
        remote_login, remote_base = args.remote.split(":", 1)
        step_fetch_artifacts(
            args.folder,
            remote_login,
            remote_base,
            args.no_cache,
            args.yes,
            model_run=args.model_run,
        )

    if run(6) and ask("Generate tables?", args.yes):
        _generate_all_tables(args.folder, "results/tamarin")

    if run(7):
        step_insert_paper_tables(args.folder, args.paper_dir, args.yes)

    if run(8) and ask("Generate plots?", args.yes):
        heatmaps = _load_heatmaps(args.folder)
        if heatmaps:
            _plot_heatmaps(heatmaps, args.folder)


if __name__ == "__main__":
    main()
