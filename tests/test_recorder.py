import numpy as np
import pytest

from src.parser.recorder import RecorderConfig
from src.utils.recorder import Recorder


def _cfg(**overrides):
    defaults = dict(log_freq=10, log_avg_window_model_step=5, log_avg_window_env_step=5, log_avg_window=100)
    defaults.update(overrides)
    return RecorderConfig(**defaults)


def _make_recorder(tmp_path, **cfg_overrides):
    return Recorder(_cfg(**cfg_overrides), path=str(tmp_path))


# ---------------------------------------------------------------------------
# get_window: prefix routing and special-key overrides
# ---------------------------------------------------------------------------


class TestGetWindow:
    @pytest.mark.parametrize("key, expected", [
        ("Search/tree_size", 1),
        ("SearchOverview/foo", 1),
        ("SearchState/step", 1),
        ("System/mem", 1),
    ])
    def test_unit_window_prefixes(self, tmp_path, key, expected):
        r = _make_recorder(tmp_path)
        assert r.get_window(key) == expected

    def test_env_step_uses_config(self, tmp_path):
        r = _make_recorder(tmp_path, log_avg_window_env_step=20)
        assert r.get_window("EnvStep/reward") == 20

    def test_model_step_uses_config(self, tmp_path):
        r = _make_recorder(tmp_path, log_avg_window_model_step=7)
        assert r.get_window("ModelStep/loss") == 7

    def test_fallback_uses_log_avg_window(self, tmp_path):
        r = _make_recorder(tmp_path, log_avg_window=42)
        assert r.get_window("SomeOther/metric") == 42

    def test_special_window_overrides_prefix(self, tmp_path):
        r = _make_recorder(tmp_path, log_avg_window_model_step=5)
        assert r.get_window("ModelStep/n_updates") == 1


# ---------------------------------------------------------------------------
# _get_metric_value: flat and nested key lookup
# ---------------------------------------------------------------------------


class TestGetMetricValue:
    def test_flat_key(self, tmp_path):
        r = _make_recorder(tmp_path)
        assert r._get_metric_value("foo", {"foo": 10}) == 10

    def test_nested_key(self, tmp_path):
        r = _make_recorder(tmp_path)
        assert r._get_metric_value("a/b", {"a": {"b": 99}}) == 99

    def test_missing_key_returns_none(self, tmp_path):
        r = _make_recorder(tmp_path)
        assert r._get_metric_value("missing", {"other": 1}) is None

    def test_nested_missing_subkey(self, tmp_path):
        r = _make_recorder(tmp_path)
        assert r._get_metric_value("a/c", {"a": {"b": 1}}) is None

    def test_flat_takes_priority_over_nested(self, tmp_path):
        r = _make_recorder(tmp_path)
        assert r._get_metric_value("a/b", {"a/b": 5, "a": {"b": 99}}) == 5


# ---------------------------------------------------------------------------
# log_step: buffer population and windowing
# ---------------------------------------------------------------------------


class TestLogStep:
    def test_flat_values_buffered(self, tmp_path):
        r = _make_recorder(tmp_path)
        r.log_step({"Search/x": 1.0}, step=1)
        assert list(r.stats_buffer["Search/x"]) == [1.0]

    def test_nested_dict_flattened(self, tmp_path):
        r = _make_recorder(tmp_path)
        r.log_step({"Worker": {"loss": 0.5}}, step=1)
        assert "Worker/loss" in r.stats_buffer
        assert list(r.stats_buffer["Worker/loss"]) == [0.5]

    def test_window_size_respected(self, tmp_path):
        r = _make_recorder(tmp_path)
        # Search/ prefix → window=1, so buffer only keeps last value
        for i in range(5):
            r.log_step({"Search/val": float(i)}, step=i + 1)
        assert list(r.stats_buffer["Search/val"]) == [4.0]

    def test_avg_window_accumulates(self, tmp_path):
        r = _make_recorder(tmp_path, log_avg_window=10)
        for i in range(5):
            r.log_step({"Custom/val": float(i)}, step=i + 1)
        assert list(r.stats_buffer["Custom/val"]) == [0.0, 1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# write_log: frequency gating and force
# ---------------------------------------------------------------------------


class TestWriteLog:
    def test_skips_non_freq_steps(self, tmp_path):
        r = _make_recorder(tmp_path, log_freq=10)
        r.stats_buffer = {"x": np.array([1.0])}
        # First log at a freq step to set the flag
        r.write_log(step=10)
        assert r.logged_step_already is True
        # Non-freq step should reset the flag
        r.write_log(step=3)
        assert r.logged_step_already is False

    def test_logs_at_freq_step(self, tmp_path):
        r = _make_recorder(tmp_path, log_freq=10)
        r.stats_buffer = {"x": np.array([1.0])}
        r.write_log(step=10)
        assert r.logged_step_already is True

    def test_no_double_log_same_step(self, tmp_path):
        r = _make_recorder(tmp_path, log_freq=10)
        r.stats_buffer = {"x": np.array([1.0])}
        r.write_log(step=10)
        assert r.logged_step_already is True
        ts_after_first = r.last_ts
        # Second call at same step should be skipped — last_ts must not update
        r.write_log(step=10)
        assert r.logged_step_already is True
        assert r.last_ts == ts_after_first

    def test_force_bypasses_freq(self, tmp_path):
        r = _make_recorder(tmp_path, log_freq=10)
        r.stats_buffer = {"x": np.array([1.0])}
        r.write_log(step=3, force=True)
        assert r.logged_step_already is True

    def test_none_in_buffer_raises(self, tmp_path):
        r = _make_recorder(tmp_path, log_freq=1)
        from collections import deque
        r.stats_buffer = {"x": deque([None])}
        with pytest.raises(AssertionError, match="None value"):
            r.write_log(step=1)


# ---------------------------------------------------------------------------
# log_system_state
# ---------------------------------------------------------------------------


class TestLogSystemState:
    def test_stores_data(self, tmp_path):
        r = _make_recorder(tmp_path)
        data = [["worker", 123, 100, None, None, "lemma1", "theory1", 1000, 0, 5, 1.2]]
        r.log_system_state(data, step=1)
        assert r.system_state == data
