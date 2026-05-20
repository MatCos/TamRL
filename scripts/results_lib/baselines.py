from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from pathlib import Path


class DummyRun:
    def __init__(self, name: str) -> None:
        self.name = name
        self.state = "finished"
        self.id = name


def extract_theory_name(spthy_file: Path) -> str | None:
    try:
        with open(spthy_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                match = re.match(r"^\s*theory\s+(\S+)(?:\s+begin|\s*$)", line)
                if match:
                    return match.group(1)
    except Exception as e:
        print(f"Warning: Could not read {spthy_file}: {e}")
    return None


def build_folder_to_theory_mapping(root_dir: str) -> dict[Path, str]:
    mapping: dict[Path, str] = {}
    for spthy_file in Path(root_dir).rglob("*.spthy"):
        if "results" in spthy_file.parts or "artifacts" in spthy_file.parts:
            continue
        try:
            theory_name = extract_theory_name(spthy_file)
            if theory_name:
                mapping[spthy_file.parent] = theory_name
        except Exception as e:
            print(f"Warning: Error processing {spthy_file}: {e}")
    return mapping


def load_csv_data(root_dir: str, models_dir: str = "eval") -> tuple[dict, dict]:
    folder_to_theory = build_folder_to_theory_mapping(models_dir)
    baseline_data: dict = {}
    baseline_runs: dict = {}
    version_map = {
        "base_c": "baseline_c",
        "base_s": "baseline_s",
        "original": "baseline_original",
    }

    for folder, theory_name in folder_to_theory.items():
        try:
            rel = folder.relative_to(models_dir)
        except ValueError:
            rel = Path(folder.name)
        csv_search_dir = Path(root_dir) / rel
        csv_files = list(csv_search_dir.rglob("*.csv")) if csv_search_dir.exists() else []
        if not csv_files:
            print(
                f"Warning: No CSV files found in folder {folder}"
                f" for theory {theory_name}. Skipping."
            )
            continue
        if len(csv_files) > 1:
            print(
                f"Warning: Multiple CSV files found in folder {folder}"
                f" for theory {theory_name}. Using the first one found."
            )
        csv_file = csv_files[0]

        version = None
        for v in version_map:
            if v in csv_file.parts:
                version = v
                break
        if not version:
            print(f"Warning: No version found in path {csv_file}. Skipping.")
            continue

        run_id = version_map[version]
        if run_id not in baseline_data:
            baseline_data[run_id] = {}
            baseline_runs[run_id] = DummyRun(run_id)
        if theory_name not in baseline_data[run_id]:
            baseline_data[run_id][theory_name] = {}

        try:
            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 3:
                        continue
                    lemma_name = row[0].strip()
                    result = row[1].strip().lower()
                    try:
                        time_val = float(row[2].strip())
                    except ValueError:
                        time_val = None

                    tree_size = None
                    if len(row) >= 4:
                        try:
                            tree_size = int(row[3].strip())
                        except ValueError:
                            pass

                    completes = (
                        1 if result in {"true", "false", "verified", "falsified"} else 0
                    )
                    entry: dict[str, int | float | str] = {"completes": completes}
                    if completes:
                        if time_val is not None:
                            entry["first_completion_time"] = time_val
                        if tree_size is not None:
                            entry["tree_size"] = tree_size
                    baseline_data[run_id][theory_name][lemma_name] = entry
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if "baseline_s" in baseline_data and "baseline_original" in baseline_data:
        for theory_name in baseline_data["baseline_original"]:
            if theory_name not in baseline_data["baseline_s"]:
                baseline_data["baseline_s"][theory_name] = baseline_data[
                    "baseline_original"
                ][theory_name]
    elif "baseline_original" in baseline_data and "baseline_s" not in baseline_data:
        baseline_data["baseline_s"] = dict(baseline_data["baseline_original"])
        baseline_runs["baseline_s"] = DummyRun("baseline_s")

    return baseline_data, baseline_runs


def load_csv_results(
    root_dir: str, models_dir: str = "eval"
) -> dict[str, dict[str, dict[str, dict]]]:
    """Load all baseline CSV files.

    Returns: {theory_name: {version: {lemma: {result, time, tree_size, csv_file}}}}
    """
    csv_results: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    folder_to_theory: dict[str, str] = {}
    for path_key, theory in build_folder_to_theory_mapping(models_dir).items():
        try:
            rel = path_key.relative_to(root_dir)
            if rel.parts:
                folder_to_theory.setdefault(rel.parts[0], theory)
        except ValueError:
            pass

    for csv_file in Path(root_dir).rglob("*.csv"):
        parts = csv_file.parts
        folder_name = version = None
        for i, part in enumerate(parts):
            if part == os.path.basename(root_dir):
                if i + 1 < len(parts):
                    folder_name = parts[i + 1]
                if i + 2 < len(parts):
                    version = parts[i + 2]
                break
        if not folder_name or not version:
            continue
        theory_name = folder_to_theory.get(folder_name, folder_name)
        try:
            with open(csv_file, encoding="utf-8") as f:
                for row in csv.reader(f):
                    if len(row) < 3:
                        continue
                    lemma = row[0].strip()
                    result = row[1].strip()
                    try:
                        time_val = float(row[2].strip())
                    except ValueError:
                        continue
                    tree_size = None
                    if len(row) >= 4:
                        try:
                            tree_size = int(row[3].strip())
                        except ValueError:
                            pass
                    csv_results[theory_name][version][lemma] = {
                        "result": result,
                        "time": time_val,
                        "tree_size": tree_size,
                        "csv_file": str(csv_file.relative_to(root_dir)),
                    }
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    result = {
        theory: {version: dict(lemmas) for version, lemmas in versions.items()}
        for theory, versions in csv_results.items()
    }

    for theory, versions in result.items():
        if "base_s" not in versions and "original" in versions:
            versions["base_s"] = versions["original"]

    return result
