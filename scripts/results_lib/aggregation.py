from __future__ import annotations

from .utils import fmt_time


def aggregate_runs(
    raw_data: dict,
    summaries: list[dict],
    heatmaps: list[dict],
    run_map: dict,
) -> None:
    for run_id, theories_data in raw_data.items():
        if run_id not in run_map:
            print(f"Run ID {run_id} not found in config anymore. Skipping.")
            continue
        run_name = run_map[run_id].name
        meta = theories_data.get("_meta")
        if meta and meta.get("state") in ["running", "starting"]:
            run_name = f"{run_name} (running)"

        for theory, lemmas_data in theories_data.items():
            if theory == "_meta":
                continue
            if "baseline" not in run_map[run_id].name and "single" not in run_map[run_id].name:
                for summary in summaries:
                    if theory not in summary:
                        summary[theory] = {"lemmas": {}, "total": {}}
                    summary[theory]["total"]["runs"] = (
                        summary[theory]["total"].get("runs", 0) + 1
                    )
                    summary[theory]["total"]["total_lemmas"] = len(lemmas_data)

                    time_max = 0.0
                    lemmas_completed = 0
                    lemmas_not_completed = 0
                    lemmas_completed_sum = 0

                    for lemma, metrics in lemmas_data.items():
                        count_completes = metrics.get("completes", 0)
                        lemmas_completed_sum += count_completes
                        lemmas_completed += 1 if count_completes > 0 else 0
                        lemmas_not_completed += 1 if count_completes == 0 else 0

                        if lemma not in summary[theory]["lemmas"]:
                            summary[theory]["lemmas"][lemma] = {}
                        prev = summary[theory]["lemmas"][lemma]

                        first_completion_time = metrics.get(
                            "first_completion_time", None
                        )
                        if first_completion_time is not None:
                            try:
                                first_completion_time = float(
                                    first_completion_time
                                )
                            except (TypeError, ValueError):
                                first_completion_time = None
                        if first_completion_time is not None:
                            curr = prev.get("min_time", None)
                            if curr is not None:
                                try:
                                    curr = float(curr)
                                except (TypeError, ValueError):
                                    curr = None
                            prev["min_time"] = (
                                min(curr, first_completion_time)
                                if curr is not None
                                else first_completion_time
                            )
                            time_max = max(time_max, first_completion_time)

                        first_tree_size = metrics.get("tree_size", None)
                        if first_tree_size is not None:
                            curr = prev.get("min_tree_size", None)
                            prev["min_tree_size"] = (
                                min(curr, first_tree_size)
                                if curr is not None
                                else first_tree_size
                            )

                        prev["count"] = prev.get("count", 0) + (
                            1 if count_completes > 0 else 0
                        )
                        prev["type"] = metrics.get("type", "unknown")

                    percentage = (
                        (lemmas_completed / len(lemmas_data)) * 100
                        if len(lemmas_data) > 0
                        else 0.0
                    )
                    best = summary[theory]["total"].get("best_run")
                    if best is not None:
                        best_lc = int(best["lemmas_completed"])
                        best_pct = float(
                            str(best["percentage"]).rstrip("%")
                        )
                    if (
                        best is None
                        or lemmas_completed > best_lc
                        or (
                            lemmas_completed == best_lc
                            and percentage > best_pct
                        )
                    ):
                        summary[theory]["total"]["best_run"] = {
                            "name": run_name,
                            "lemmas_completed": lemmas_completed,
                            "lemmas_completed_sum": lemmas_completed_sum,
                            "time": time_max,
                            "missing_lemmas": lemmas_not_completed,
                            "percentage": percentage,
                            "total_lemmas": len(lemmas_data),
                        }
                    if lemmas_not_completed == 0:
                        summary[theory]["total"]["all_completed_runs"] = (
                            summary[theory]["total"].get("all_completed_runs", 0) + 1
                        )
                        summary[theory]["total"].setdefault(
                            "all_complete", []
                        ).append(run_name)

            if "baseline" not in run_map[run_id].name and "single" not in run_map[run_id].name and not all(m.get("completes", 0) == 0 for m in lemmas_data.values()):
                for heatmap in heatmaps:
                    if theory not in heatmap:
                        heatmap[theory] = {}
                    for lemma, metrics in lemmas_data.items():
                        heatmap[theory][(lemma, run_name)] = metrics


def format_summary(summary: dict) -> dict:
    formatted: dict = {}
    for theory in sorted(summary.keys()):
        formatted[theory] = summary[theory]
        lemmas = formatted[theory]["lemmas"]
        for entry in lemmas.values():
            if "min_time" in entry:
                entry["min_time"] = fmt_time(entry["min_time"])
        formatted[theory]["lemmas"] = dict(
            sorted(lemmas.items(), key=lambda item: item[1]["count"], reverse=True)
        )
        best = formatted[theory]["total"].get("best_run", {})
        if "time" in best:
            best["time"] = fmt_time(best["time"])
        if "percentage" in best:
            pct = best["percentage"]
            if not isinstance(pct, str):
                best["percentage"] = f"{pct:.1f}%"

        total_lemmas = formatted[theory]["total"].get("total_lemmas", 0)
        solved_count = sum(1 for v in lemmas.values() if v["count"] > 0)
        if total_lemmas > 0:
            formatted[theory]["total"][
                "percentage"
            ] = f"{(solved_count / total_lemmas) * 100:.1f}%"
    return formatted
