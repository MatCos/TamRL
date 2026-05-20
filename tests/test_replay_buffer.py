import random

import pytest

from src.environment.environment import State
from src.rl.replay_buffer import ACTrainingExample, Batch, ReplayBuffer


def _state(n_methods: int = 3) -> State:
    return State(
        proof_methods=[{"name": f"m{i}"} for i in range(n_methods)],
        encoded_methods=tuple(f"enc{i}" for i in range(n_methods)),
        encoded_sys="sys",
    )


def _example(n_methods: int = 3, action: int = 0, value: float = 1.0) -> ACTrainingExample:
    return ACTrainingExample(state=_state(n_methods), action=action, value=value)


# ---------------------------------------------------------------------------
# ACTrainingExample
# ---------------------------------------------------------------------------


class TestACTrainingExample:
    def test_num_proof_methods(self):
        assert _example(n_methods=5).num_proof_methods == 5

    def test_equality(self):
        a = _example(n_methods=2, action=1, value=0.5)
        b = _example(n_methods=2, action=1, value=0.5)
        assert a == b

    def test_inequality_different_action(self):
        assert _example(action=0) != _example(action=1)

    def test_inequality_different_value(self):
        assert _example(value=1.0) != _example(value=2.0)

    def test_inequality_different_type(self):
        assert _example() != "not an example"

    def test_hashable_and_consistent(self):
        a = _example(n_methods=2, action=1, value=0.5)
        b = _example(n_methods=2, action=1, value=0.5)
        assert hash(a) == hash(b)
        assert len({a, b}) == 1


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


class TestBatch:
    def test_from_items_and_len(self):
        batch = Batch.from_items([_example(), _example()])
        assert len(batch) == 2

    def test_iter(self):
        items = [_example(action=i) for i in range(3)]
        batch = Batch.from_items(items)
        assert list(batch) == items

    def test_to_list(self):
        items = [_example(action=i) for i in range(3)]
        batch = Batch.from_items(items)
        assert batch.to_list() == items

    def test_slice_batch_fits_in_one(self):
        # 3 items * 2 methods each = 6 total, budget = 10
        items = [_example(n_methods=2) for _ in range(3)]
        batches = Batch.from_items(items).slice_batch(10)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_slice_batch_splits(self):
        # 4 items * 3 methods each = 12 total, budget = 6 -> should split
        items = [_example(n_methods=3, action=i) for i in range(4)]
        batches = Batch.from_items(items).slice_batch(6)
        assert len(batches) == 2
        total_items = sum(len(b) for b in batches)
        assert total_items == 4

    def test_slice_batch_oversized_item_gets_own_batch(self):
        # One item with 10 methods, budget = 5
        items = [_example(n_methods=10), _example(n_methods=2), _example(n_methods=2)]
        batches = Batch.from_items(items).slice_batch(5)
        # Oversized item should be in its own batch
        oversized = [b for b in batches if any(e.num_proof_methods == 10 for e in b)]
        assert len(oversized) == 1
        assert len(oversized[0]) == 1

    def test_slice_batch_preserves_order_no_oversized(self):
        # All items fit within budget individually, so order is preserved
        items = [_example(n_methods=2, action=i) for i in range(6)]
        batches = Batch.from_items(items).slice_batch(5)
        recovered = [e for b in batches for e in b]
        assert recovered == items

    def test_slice_batch_preserves_all_items_with_oversized(self):
        # Oversized items get pulled out of sequence, so check as set
        items = [_example(n_methods=i + 1, action=i) for i in range(6)]
        batches = Batch.from_items(items).slice_batch(5)
        recovered = [e for b in batches for e in b]
        assert set(recovered) == set(items)
        assert len(recovered) == len(items)

    def test_slice_batch_respects_budget_per_slice(self):
        items = [_example(n_methods=3) for _ in range(10)]
        budget = 7
        batches = Batch.from_items(items).slice_batch(budget)
        for batch in batches:
            pfm_sum = sum(e.num_proof_methods for e in batch)
            # Each batch should not exceed budget (unless single oversized item)
            if len(batch) > 1:
                assert pfm_sum <= budget


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------


class TestReplayBuffer:
    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=10)
        buf.push(_example())
        buf.push(_example())
        assert len(buf) == 2

    def test_capacity_evicts_old(self):
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            buf.push(_example(action=i))
        assert len(buf) == 3
        # FIFO eviction: oldest items (actions 0, 1) should be gone
        remaining_actions = [e.action for e in buf.memory]
        assert remaining_actions == [2, 3, 4]

    def test_extend(self):
        buf = ReplayBuffer(capacity=10)
        buf.extend([_example(action=i) for i in range(4)])
        assert len(buf) == 4

    def test_filled_ratio(self):
        buf = ReplayBuffer(capacity=10)
        buf.extend([_example() for _ in range(5)])
        assert buf.filled_ratio == 0.5

    def test_filled_ratio_empty(self):
        buf = ReplayBuffer(capacity=10)
        assert buf.filled_ratio == 0.0

    def test_sample_returns_correct_size(self):
        buf = ReplayBuffer(capacity=100)
        buf.extend([_example(action=i) for i in range(20)])
        batch = buf.sample(5)
        assert len(batch) == 5

    def test_sample_resets_fresh_samples(self):
        buf = ReplayBuffer(capacity=100)
        buf.extend([_example() for _ in range(10)])
        assert buf.fresh_samples == 10
        buf.sample(3)
        assert buf.fresh_samples == 0

    def test_fresh_samples_tracks_push(self):
        buf = ReplayBuffer(capacity=100)
        buf.push(_example())
        assert buf.fresh_samples == 1
        buf.push(_example())
        assert buf.fresh_samples == 2

    def test_fresh_samples_tracks_extend(self):
        buf = ReplayBuffer(capacity=100)
        buf.extend([_example() for _ in range(5)])
        assert buf.fresh_samples == 5

    def test_sample_deterministic_with_seed(self):
        items = [_example(action=i) for i in range(20)]
        buf1 = ReplayBuffer(capacity=100)
        buf1.extend(items)
        buf2 = ReplayBuffer(capacity=100)
        buf2.extend(items)
        random.seed(42)
        batch1 = buf1.sample(5)
        random.seed(42)
        batch2 = buf2.sample(5)
        assert batch1.to_list() == batch2.to_list()


# ---------------------------------------------------------------------------
# Usage count DP consistency
# ---------------------------------------------------------------------------


def _assert_usage_stats_consistent(buf: ReplayBuffer) -> None:
    """Assert that the DP-tracked mean and min match a naive
    computation over the usage_counts deque."""
    counts = list(buf.usage_counts)
    if len(counts) == 0:
        assert buf.mean_usage_count == 0.0
        assert buf.min_usage_count == 0
        return
    assert buf.mean_usage_count == pytest.approx(
        sum(counts) / len(counts)
    )
    assert buf.min_usage_count == min(counts)


class TestUsageCountConsistency:

    def test_after_push(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(5):
            buf.push(_example(action=i))
            _assert_usage_stats_consistent(buf)

    def test_after_extend(self):
        buf = ReplayBuffer(capacity=20)
        buf.extend([_example(action=i) for i in range(12)])
        _assert_usage_stats_consistent(buf)

    def test_after_sample(self):
        buf = ReplayBuffer(capacity=50)
        buf.extend([_example(action=i) for i in range(30)])
        for _ in range(10):
            buf.sample(5)
            _assert_usage_stats_consistent(buf)

    def test_after_push_eviction(self):
        buf = ReplayBuffer(capacity=5)
        buf.extend([_example(action=i) for i in range(5)])
        buf.sample(3)
        _assert_usage_stats_consistent(buf)
        # Push beyond capacity, evicting items with nonzero counts
        for i in range(4):
            buf.push(_example(action=10 + i))
            _assert_usage_stats_consistent(buf)

    def test_after_extend_eviction(self):
        buf = ReplayBuffer(capacity=10)
        buf.extend([_example(action=i) for i in range(10)])
        buf.sample(5)
        _assert_usage_stats_consistent(buf)
        # Extend beyond capacity
        buf.extend([_example(action=i) for i in range(7)])
        _assert_usage_stats_consistent(buf)

    def test_interleaved_push_sample_evict(self):
        """Randomised stress test: interleave pushes and samples
        with evictions, checking consistency at every step."""
        random.seed(99)
        buf = ReplayBuffer(capacity=15)
        buf.extend([_example(action=i) for i in range(15)])
        for step in range(40):
            if len(buf) >= 4:
                buf.sample(4)
                _assert_usage_stats_consistent(buf)
            buf.push(_example(action=100 + step))
            _assert_usage_stats_consistent(buf)

    def test_extend_larger_than_capacity(self):
        buf = ReplayBuffer(capacity=5)
        buf.extend([_example(action=i) for i in range(3)])
        buf.sample(2)
        _assert_usage_stats_consistent(buf)
        # Extend with more items than capacity
        buf.extend([_example(action=i) for i in range(8)])
        _assert_usage_stats_consistent(buf)

    def test_empty_buffer(self):
        buf = ReplayBuffer(capacity=10)
        _assert_usage_stats_consistent(buf)
