from __future__ import annotations

from pathlib import Path

from .utils import is_solved, is_completed, avg
from .tables import (
    _all_models,
    build_lemma_completion_by_type_rows,
    build_proofsize_overview_rows,
)


# ---------------------------------------------------------------------------
# Public markdown generators
# ---------------------------------------------------------------------------


def generate_lemma_completion_overview(
    yaml_data: dict, csv_data: dict, output_path: str | Path
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Lemma Completion Overview\n\n")
        f.write(
            "| Protocol | Total | ML (any) | ML (best run)"
            " | Tamarin (c) | Tamarin (s) | Original |\n"
        )
        f.write(
            "|----------|-------|----------|---------------|"
            "-------------|-------------|----------|\n"
        )
        for model in sorted(set(yaml_data) | set(csv_data)):
            ym = yaml_data.get(model, {})
            cv = csv_data.get(model, {})
            yl = ym.get("lemmas", {})
            total = ym.get("total", {}).get("total_lemmas", 0) or len(yl)
            ml_any = sum(
                1 for d in yl.values() if d.get("min_time") not in (None, "null")
            )
            ml_best = 0
            if "total" in ym and "best_run" in ym["total"]:
                ml_best = ym["total"]["best_run"].get("lemmas_completed", 0)
            c_solved = sum(
                1 for d in cv.get("base_c", {}).values() if is_solved(d["result"])
            )
            s_solved = sum(
                1 for d in cv.get("base_s", {}).values() if is_solved(d["result"])
            )
            orig_solved = sum(
                1 for d in cv.get("original", {}).values() if is_solved(d["result"])
            )
            f.write(
                f"| {model} | {total} | {ml_any} | {ml_best}"
                f" | {c_solved} | {s_solved} | {orig_solved} |\n"
            )
        f.write(
            "\n**Legend:** Total: total lemmas; ML (any): union across all runs;"
            " ML (best run): single best run; C/S: Tamarin heuristic.\n"
        )


def generate_lemma_completion_by_type(
    yaml_data: dict, csv_data: dict, raw_data: dict, output_path: str | Path
) -> None:
    rows = build_lemma_completion_by_type_rows(yaml_data, csv_data, raw_data)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Lemma Completion by Type\n\n")
        f.write(
            "| Protocol | Type | Total | ML (any) | ML (best run)"
            " | Tamarin (c) | Tamarin (s) | Original |\n"
        )
        f.write(
            "|----------|------|-------|----------|---------------|"
            "-------------|-------------|----------|\n"
        )
        for r in rows:
            f.write(
                f"| {r['Protocol']} | {r['Type']} | {r['Total']}"
                f" | {r['ML (any)']} | {r['ML (best run)']}"
                f" | {r['Tamarin (c)']} | {r['Tamarin (s)']}"
                f" | {r['Original']} |\n"
            )
        f.write(
            "\n**Legend:** Total: total lemmas of that type; ML (any): union across"
            " all runs; ML (best run): single best run; C/S: Tamarin heuristic.\n"
        )


def generate_proofsize_overview(
    yaml_data: dict, csv_data: dict, output_path: str | Path
) -> None:
    rows = build_proofsize_overview_rows(yaml_data, csv_data)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Proof Size Overview\n\nAll sizes are tree sizes.\n\n")
        f.write(
            "| Protocol | ML (any) | Tamarin (c) | Tamarin (s)"
            " | Both ML | Both Tamarin | Original | Both ML | Both Original |\n"
        )
        f.write(
            "|----------|----------|-------------|-------------|"
            "---------|--------------|----------|---------|---------------|\n"
        )
        for r in rows:
            f.write(
                f"| {r['Protocol']} | {r['ML (any)']}"
                f" | {r['Tamarin (c)']} | {r['Tamarin (s)']}"
                f" | {r['Both ML (tam)']} | {r['Both Tamarin']}"
                f" | {r['Original']}"
                f" | {r['Both ML (orig)']} | {r['Both Original']} |\n"
            )
        f.write("\n**Note:** 'Both' columns restrict to lemmas solved by both.\n")
