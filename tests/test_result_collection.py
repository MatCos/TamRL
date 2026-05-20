"""Tests for the result collection pipeline (scripts/result_collection.py + results_lib/)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from results_lib.utils import parse_time_to_seconds
from results_lib.aggregation import aggregate_runs, format_summary
from results_lib.tables import _extract_solved_sizes, assign_groups
from result_collection import (
    _find_best_per_protocol,
    _is_subsumed,
    _merge_proto_yaml,
    _parse_step_range,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_map(names: dict[str, str]):
    class _R:
        def __init__(self, n):
            self.name = n
    return {rid: _R(name) for rid, name in names.items()}


def _lemma(completes=0, time=0.0, size=0):
    return {"completes": completes, "first_completion_time": time, "tree_size": size}


# ---------------------------------------------------------------------------
# Time parsing roundtrips correctly through format → parse
# ---------------------------------------------------------------------------


class TestTimeParsingRoundtrip:
    def test_roundtrip_all_scales(self):
        for secs in [45.0, 125.3, 3661.5]:
            from results_lib.utils import fmt_seconds
            assert parse_time_to_seconds(fmt_seconds(secs)) == pytest.approx(secs, abs=0.05)

    def test_partial_formats(self):
        assert parse_time_to_seconds("2h") == 7200.0
        assert parse_time_to_seconds("45.5s") == 45.5

    @pytest.mark.parametrize("val", [None, "null", "N/A", "---", "", "garbage"])
    def test_missing_or_invalid_returns_none(self, val):
        assert parse_time_to_seconds(val) is None


# ---------------------------------------------------------------------------
# Aggregation excludes baseline and single-lemma runs
# ---------------------------------------------------------------------------


class TestAggregation:
    def _two_run_raw(self):
        return {
            "run1": {
                "_meta": {"name": "run1 - base_s_model", "state": "finished"},
                "proto_A": {
                    "l1": {"completes": 2, "first_completion_time": 100.0, "tree_size": 50},
                    "l2": {"completes": 1, "first_completion_time": 200.0, "tree_size": 30},
                    "l3": {"completes": 0},
                },
            },
            "run2": {
                "_meta": {"name": "run2 - base_s_model", "state": "finished"},
                "proto_A": {
                    "l1": {"completes": 1, "first_completion_time": 80.0, "tree_size": 60},
                    "l2": {"completes": 0},
                    "l3": {"completes": 1, "first_completion_time": 300.0, "tree_size": 40},
                },
            },
        }

    def test_summary_reports_fastest_solve_and_smallest_tree_across_runs(self):
        raw = self._two_run_raw()
        rm = _run_map({"run1": "run1 - base_s_model", "run2": "run2 - base_s_model"})
        summary = [{}]
        aggregate_runs(raw, summary, [], rm)
        lemmas = summary[0]["proto_A"]["lemmas"]
        assert lemmas["l1"]["min_time"] == 80.0
        assert lemmas["l1"]["min_tree_size"] == 50

    def test_best_run_picks_most_completed(self):
        raw = self._two_run_raw()
        rm = _run_map({"run1": "run1 - base_s_model", "run2": "run2 - base_s_model"})
        summary = [{}]
        aggregate_runs(raw, summary, [], rm)
        assert summary[0]["proto_A"]["total"]["best_run"]["lemmas_completed"] == 2

    def test_baseline_runs_excluded_from_summary_and_heatmap(self):
        raw = {"b1": {"_meta": {"name": "baseline_c"}, "p": {"l1": _lemma(1, 10, 5)}}}
        summary, heatmaps = [{}], [{}]
        aggregate_runs(raw, summary, heatmaps, _run_map({"b1": "baseline_c"}))
        assert "p" not in summary[0]
        assert "p" not in heatmaps[0]

    def test_single_lemma_runs_excluded_from_summary_and_heatmap(self):
        raw = {"s1": {"_meta": {"name": "s1 - single - l1"}, "p": {"l1": _lemma(1, 5, 3)}}}
        summary, heatmaps = [{}], [{}]
        aggregate_runs(raw, summary, heatmaps, _run_map({"s1": "s1 - single - l1"}))
        assert "p" not in summary[0]
        assert "p" not in heatmaps[0]

    def test_mixed_string_and_float_times_compare_correctly(self):
        raw = {
            "r1": {"_meta": {"name": "r1"}, "p": {"l1": {"completes": 1, "first_completion_time": "50.0", "tree_size": 10}}},
            "r2": {"_meta": {"name": "r2"}, "p": {"l1": {"completes": 1, "first_completion_time": "30.0", "tree_size": 20}}},
        }
        summary = [{}]
        aggregate_runs(raw, summary, [], _run_map({"r1": "r1", "r2": "r2"}))
        assert summary[0]["p"]["lemmas"]["l1"]["min_time"] == 30.0


class TestFormatSummary:
    def test_lemmas_sorted_most_solved_first(self):
        summary = {
            "proto": {
                "lemmas": {"rare": {"count": 1}, "common": {"count": 5}},
                "total": {"total_lemmas": 2},
            },
        }
        result = format_summary(summary)
        assert list(result["proto"]["lemmas"].keys()) == ["common", "rare"]


# ---------------------------------------------------------------------------
# Proof size extraction filters by solved status and lemma set
# ---------------------------------------------------------------------------


class TestExtractSolvedSizes:
    def test_only_solved_lemmas_included(self):
        d = {
            "l1": {"result": "verified", "tree_size": 10},
            "l2": {"result": "timeout", "tree_size": 20},
        }
        assert _extract_solved_sizes(d) == [10]

    def test_per_type_table_only_counts_lemmas_of_that_type(self):
        d = {
            "l1": {"result": "verified", "tree_size": 10},
            "l2": {"result": "verified", "tree_size": 20},
        }
        assert _extract_solved_sizes(d, {"l2"}) == [20]

    def test_corrupt_proof_sizes_excluded_from_averages(self):
        d = {
            "l1": {"result": "verified", "tree_size": None},
            "l2": {"result": "verified", "tree_size": -1},
            "l3": {"result": "verified", "tree_size": 5},
        }
        assert _extract_solved_sizes(d) == [5]


# ---------------------------------------------------------------------------
# Protocol grouping assigns to the right category
# ---------------------------------------------------------------------------


class TestAssignGroups:
    def test_protocols_land_in_correct_categories(self):
        rows = [{"Protocol": "Tutorial"}, {"Protocol": "wireguard"}]
        grouped = assign_groups(rows)
        assert any(r["Protocol"] == "Tutorial" for r in grouped["Classical Models"])
        assert any(r["Protocol"] == "wireguard" for r in grouped["Complex Models"])


# ---------------------------------------------------------------------------
# Step range parsing
# ---------------------------------------------------------------------------


class TestParseStepRange:
    def test_users_can_select_pipeline_steps_flexibly(self):
        assert _parse_step_range("1,3-5,7", {1, 2, 3, 4, 5, 6, 7}) == {1, 3, 4, 5, 7}

    def test_invalid_step_exits(self):
        with pytest.raises(SystemExit):
            _parse_step_range("99", {1, 2, 3})


# ---------------------------------------------------------------------------
# Best/others split: subsumed detection and best-run selection
# ---------------------------------------------------------------------------


class TestIsSubsumed:
    def test_run_solving_fewer_lemmas_is_redundant(self):
        run = {"l1": _lemma(1, 10, 5)}
        other = {"l1": _lemma(1, 10, 5), "l2": _lemma(1, 20, 10)}
        assert _is_subsumed(run, other)

    def test_run_with_unique_lemma_is_not_subsumed(self):
        run = {"l1": _lemma(1, 10, 5), "l2": _lemma(1, 20, 10)}
        other = {"l1": _lemma(1, 10, 5), "l3": _lemma(1, 30, 15)}
        assert not _is_subsumed(run, other)

    def test_same_lemmas_but_faster_is_not_subsumed(self):
        assert not _is_subsumed({"l1": _lemma(1, 5, 5)}, {"l1": _lemma(1, 10, 5)})

    def test_same_lemmas_but_slower_is_subsumed(self):
        assert _is_subsumed({"l1": _lemma(1, 20, 10)}, {"l1": _lemma(1, 10, 5)})

    def test_identical_runs_are_not_subsumed(self):
        m = _lemma(1, 10, 5)
        assert not _is_subsumed({"l1": m}, {"l1": m})


class TestFindBestPerProtocol:
    def test_prefers_more_completed_over_faster(self):
        pooled = {
            "r1": {"proto": {"l1": _lemma(1, 100), "l2": _lemma(1, 200)}},
            "r2": {"proto": {"l1": _lemma(1, 50)}},
        }
        best = _find_best_per_protocol(pooled)
        assert best["proto"][0] == "r1"

    def test_breaks_completion_tie_by_fastest_time(self):
        pooled = {
            "r1": {"proto": {"l1": _lemma(1, 100)}},
            "r2": {"proto": {"l1": _lemma(1, 50)}},
        }
        assert _find_best_per_protocol(pooled)["proto"][0] == "r2"


# ---------------------------------------------------------------------------
# Merging summaries across configs keeps best per lemma
# ---------------------------------------------------------------------------


class TestMergeProtoYaml:
    def test_keeps_faster_time_and_smaller_tree_independently(self):
        dst = {"lemmas": {"l1": {"min_time": "5m 0.0s", "count": 1, "min_tree_size": 20}}}
        src = {"lemmas": {"l1": {"min_time": "2m 0.0s", "count": 1, "min_tree_size": 30}}}
        _merge_proto_yaml(dst, src)
        assert dst["lemmas"]["l1"]["min_time"] == "2m 0.0s"
        assert dst["lemmas"]["l1"]["min_tree_size"] == 20
        assert dst["lemmas"]["l1"]["count"] == 2

    def test_new_lemma_from_src_added_to_dst(self):
        dst = {"lemmas": {"l1": {"count": 1}}}
        src = {"lemmas": {"l2": {"count": 1, "min_time": "1m 0.0s"}}}
        _merge_proto_yaml(dst, src)
        assert "l2" in dst["lemmas"]
