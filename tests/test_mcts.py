from unittest.mock import MagicMock, patch
from multiprocessing import Queue

import pytest
import torch

from src.environment.environment import State, TamarinEnvironment
from src.parser.reward import RewardConfig
from src.parser.ucb import UCBConfig
from src.parser.lemma import LemmaConfig
from src.rl.mcts import MCTS, StepLog, SelectionResult
from src.rl.node import Node, NodeType
from src.rl.node_storage import NodeStorage
from src.utils.load import LemmaType

from tests.helpers import make_and_node as _and_node
from tests.helpers import make_or_node as _or_node
from tests.helpers import make_state as _state
from tests.helpers import make_terminal_or_node as _terminal_or_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reward_config(**overrides) -> RewardConfig:
    defaults = dict(branch_penalty=0.0, time_penalty=0.0,
                    time_penalty_clip=180.0, timeout_penalty=3.0)
    defaults.update(overrides)
    return RewardConfig(**defaults)


def _ucb_config(**overrides) -> UCBConfig:
    defaults = dict(pb_c_base=3200, pb_c_init=0.001, value_discount=0.99,
                    temperature=200, c_and=64, value_penalty=8,
                    invert_and=False, heuristic_weight=0.0,
                    deactivate_model=False)
    defaults.update(overrides)
    return UCBConfig(**defaults)


def _lemma_config() -> LemmaConfig:
    return LemmaConfig(
        theory_path="/tmp/test.spthy",
        lemma_name="test_lemma",
        lemma_type=LemmaType.FORALL,
        theory_name="TestTheory",
        diff_arg=False,
        suppress_output=True,
        heuristic="S",
        side=None,
    )


def _response_state(
    n_methods: int = 3, status: str = "InProgress"
) -> TamarinEnvironment.ResponseState:
    return TamarinEnvironment.ResponseState(
        proof_methods=[{"name": "Simplify"} for _ in range(n_methods)],
        encoded_methods=[f"enc{i}" for i in range(n_methods)],
        encoded_sys="sys",
        status=status,
    )


def _make_mcts(
    search_budget: int = 10,
    ucb_config: UCBConfig | None = None,
    reward_config: RewardConfig | None = None,
) -> MCTS:
    """Create an MCTS instance with mocked external dependencies."""
    env = MagicMock(spec=TamarinEnvironment)
    env.cache_hit_rate = 0.0
    env.cache_usage = 0.0
    env.client = MagicMock()
    env.client.process.pid = 12345
    env.init_queue = MagicMock()

    request_conn = MagicMock()
    response_conn = MagicMock()
    training_example_queue = MagicMock()
    search_result_queue = MagicMock()
    step_log_queue = MagicMock()

    return MCTS(
        tamarin_env=env,
        request_conn=request_conn,
        response_conn=response_conn,
        training_example_queue=training_example_queue,
        search_result_queue=search_result_queue,
        step_log_queue=step_log_queue,
        search_budget=search_budget,
        search_budget_increase_factor=1.5,
        ucb_config=ucb_config or _ucb_config(),
        reward_config=reward_config or _reward_config(),
        env_id=0,
        lemma_config=_lemma_config(),
        tamarin_timeout=60,
        backup_tamarin_timeout=120,
    )


def _expanded_or_node(n_children: int = 2, n_methods: int = 3) -> Node:
    """Creates an OR node with terminal contradictory children and priors set."""
    parent = _or_node(n_methods=n_methods)
    for i in range(n_children):
        action = f"enc{i}"
        child = _terminal_or_node()
        parent.children[action] = child
        parent.children_prior[action] = 1.0
        parent.children_prior_only_heuristic[action] = 1.0
        parent.children_prior_only_model[action] = 1.0
        child.parents.append(parent)
    parent._nn_value = 0.5
    parent._value = 0.5
    return parent


def _traversable_or_node(n_methods: int = 2) -> Node:
    """Creates an expanded OR node with unexpanded (non-terminal, non-contradictory) children.
    No children_visit_count is set, so ucb_score uses the parent-value fallback."""
    parent = _or_node(n_methods=n_methods)
    for i in range(n_methods):
        action = f"enc{i}"
        child = _or_node(n_methods=1)  # unexpanded leaf
        parent.children[action] = child
        parent.children_prior[action] = 1.0
        parent.children_prior_only_heuristic[action] = 1.0
        parent.children_prior_only_model[action] = 1.0
        child.parents.append(parent)
    parent._nn_value = 0.5
    parent._value = 0.5
    return parent


# ---------------------------------------------------------------------------
# StepLog
# ---------------------------------------------------------------------------


class TestStepLog:
    def test_empty_to_dict(self):
        log = StepLog()
        d = log.to_dict()
        assert d["EnvStep/avg_prior_score"] == 0.0
        assert d["EnvStep/avg_value_score"] == 0.0
        assert d["EnvStep/selected_action_idx"] == 0

    def test_record_and_retrieve(self):
        log = StepLog()
        log.record({"key1": 1.0, "key2": 2})
        d = log.to_dict()
        assert d["key1"] == 1.0
        assert d["key2"] == 2

    def test_record_overwrites(self):
        log = StepLog()
        log.record({"key": 1.0})
        log.record({"key": 2.0})
        assert log.to_dict()["key"] == 2.0

    def test_accumulate_selection_averages_correctly(self):
        log = StepLog()
        sel1 = SelectionResult(
            action="enc0",
            child=_or_node(),
            selected_action_idx=2,
            prior_score=0.8,
            value_score=0.6,
            pb_c=0.1,
            ucb_margin=0.2,
            ucb_entropy=0.4,
            ucb_correlations={"EnvStep/prior_correlation": 1.0},
            alignments={"EnvStep/alignment_prior_score": True},
        )
        sel2 = SelectionResult(
            action="enc1",
            child=_or_node(),
            selected_action_idx=4,
            prior_score=0.2,
            value_score=0.0,
            pb_c=0.3,
            ucb_margin=0.1,
            ucb_entropy=0.6,
            ucb_correlations={"EnvStep/prior_correlation": 0.4},
            alignments={"EnvStep/alignment_prior_score": False},
        )
        log.accumulate_selection(sel1)
        log.accumulate_selection(sel2)
        d = log.to_dict()
        # Averages: (0.8+0.2)/2=0.5, (0.6+0.0)/2=0.3, (2+4)/2=3.0
        assert d["EnvStep/avg_prior_score"] == pytest.approx(0.5)
        assert d["EnvStep/avg_value_score"] == pytest.approx(0.3)
        assert d["EnvStep/selected_action_idx"] == pytest.approx(3.0)
        assert d["EnvStep/prior_correlation"] == pytest.approx(0.7)

    def test_accumulate_selection_none_action_idx(self):
        log = StepLog()
        sel = SelectionResult(
            action="case0",
            child=_or_node(),
            selected_action_idx=None,
            prior_score=0.5,
            value_score=0.3,
            pb_c=0.1,
            ucb_margin=0.2,
            ucb_entropy=0.4,
            ucb_correlations={},
            alignments={},
        )
        log.accumulate_selection(sel)
        d = log.to_dict()
        # No OR nodes, so selected_action_idx should be 0 (no division)
        assert d["EnvStep/selected_action_idx"] == 0


# ---------------------------------------------------------------------------
# MCTS.traverse
# ---------------------------------------------------------------------------


class TestTraverse:
    def test_unexpanded_root_returns_immediately(self):
        mcts = _make_mcts()
        root = _or_node()
        step_log = StepLog()
        leaf, path = mcts.traverse(root, step_log)
        assert leaf is root
        assert len(path) == 1
        assert path[0] == (None, root)
        d = step_log.to_dict()
        assert d["EnvStep/expand_depth"] == 0

    def test_traverses_expanded_node(self):
        mcts = _make_mcts()
        root = _traversable_or_node(n_methods=2)
        step_log = StepLog()
        leaf, path = mcts.traverse(root, step_log)
        # Should traverse one level deep to an unexpanded child
        assert len(path) == 2
        assert path[0] == (None, root)
        assert not leaf.is_expanded

    def test_stops_at_contradictory_node(self):
        mcts = _make_mcts()
        root = _expanded_or_node(n_children=1, n_methods=1)
        root._is_contradictory = True
        step_log = StepLog()
        leaf, path = mcts.traverse(root, step_log)
        assert leaf is root
        assert len(path) == 1

    def test_stops_at_failed_node(self):
        mcts = _make_mcts()
        root = _or_node()
        root.failed = True
        step_log = StepLog()
        leaf, path = mcts.traverse(root, step_log)
        assert leaf is root

    def test_stops_at_solved_node(self):
        mcts = _make_mcts()
        root = _or_node()
        root.is_solved = True
        step_log = StepLog()
        leaf, path = mcts.traverse(root, step_log)
        assert leaf is root

    def test_updates_max_expanded_depth(self):
        mcts = _make_mcts()
        root = _traversable_or_node(n_methods=2)
        step_log = StepLog()
        mcts.traverse(root, step_log)
        assert mcts.max_expanded_depth == 1

    def test_stale_contradictory_child_breaks(self):
        """If a child became contradictory via another path (stale), traversal breaks."""
        mcts = _make_mcts()
        root = _expanded_or_node(n_children=2, n_methods=2)
        root.children_visit_count["enc0"] = 1
        # Make one child contradictory (simulating stale DAG)
        root.children["enc0"]._is_contradictory = True
        root.children["enc1"]._is_contradictory = True
        step_log = StepLog()
        leaf, path = mcts.traverse(root, step_log)
        # Should break at root due to stale child detection
        assert leaf is root
        assert len(path) == 1


# ---------------------------------------------------------------------------
# MCTS.inference
# ---------------------------------------------------------------------------


class TestInference:
    def test_returns_logprobs_and_value(self):
        mcts = _make_mcts()
        node = _or_node(n_methods=3)
        step_log = StepLog()
        logprobs = torch.randn(3)
        nn_value = 0.42

        mcts.response_conn.poll.return_value = True
        mcts.response_conn.recv.return_value = ("req-id", logprobs, nn_value)

        with patch("src.rl.mcts.uuid.uuid4", return_value=MagicMock(__str__=lambda _: "req-id")):
            result = mcts.inference(node, step_log)

        assert result is not None
        returned_logprobs, returned_value = result
        assert torch.equal(returned_logprobs, logprobs)
        assert returned_value == nn_value
        d = step_log.to_dict()
        assert "EnvStep/model_inference_time" in d
        assert d["EnvStep/expand_value"] == nn_value

    def test_returns_none_on_terminate(self):
        mcts = _make_mcts()
        node = _or_node(n_methods=3)
        step_log = StepLog()

        mcts.response_conn.poll.return_value = True
        mcts.response_conn.recv.return_value = None

        result = mcts.inference(node, step_log)
        assert result is None
        # Should have put a search result with stopped_early=True
        mcts.search_result_queue.put.assert_called_once()
        call_args = mcts.search_result_queue.put.call_args[0][0]
        assert call_args[1]["stopped_early"] is True

    def test_timeout_raises(self):
        mcts = _make_mcts()
        node = _or_node(n_methods=3)
        step_log = StepLog()

        mcts.response_conn.poll.return_value = False

        from src.utils.utils import QueueWaitTimeoutError
        with pytest.raises(QueueWaitTimeoutError):
            mcts.inference(node, step_log)


# ---------------------------------------------------------------------------
# MCTS.add_heuristic_bias
# ---------------------------------------------------------------------------


class TestAddHeuristicBias:
    def test_zero_weight_preserves_model(self):
        mcts = _make_mcts(ucb_config=_ucb_config(heuristic_weight=0.0))
        logprobs = torch.tensor([0.5, 0.3, 0.2])
        step_log = StepLog()
        combined, heuristic_only, model_only = mcts.add_heuristic_bias(logprobs, step_log)
        # With zero weight, combined should equal model_only
        assert torch.allclose(combined, model_only, atol=1e-6)

    def test_nonzero_weight_shifts_toward_first(self):
        mcts = _make_mcts(ucb_config=_ucb_config(heuristic_weight=1.0))
        logprobs = torch.zeros(3)  # uniform raw logprobs
        step_log = StepLog()
        combined, heuristic_only, model_only = mcts.add_heuristic_bias(logprobs, step_log)
        # Heuristic favors rank 0, so combined[0] > combined[1] > combined[2]
        assert combined[0] > combined[1]
        assert combined[1] > combined[2]

    def test_logs_prior_metrics(self):
        mcts = _make_mcts(ucb_config=_ucb_config(heuristic_weight=0.5))
        logprobs = torch.tensor([0.5, 0.3, 0.2])
        step_log = StepLog()
        mcts.add_heuristic_bias(logprobs, step_log)
        d = step_log.to_dict()
        assert "EnvStep/prior_entropy" in d
        assert "EnvStep/prior_confidence" in d
        assert "EnvStep/prior_entropy_heuristic" in d
        assert "EnvStep/prior_confidence_heuristic" in d

    def test_returns_three_tensors(self):
        mcts = _make_mcts()
        logprobs = torch.randn(5)
        step_log = StepLog()
        combined, heuristic_only, model_only = mcts.add_heuristic_bias(logprobs, step_log)
        assert combined.shape == (5,)
        assert heuristic_only.shape == (5,)
        assert model_only.shape == (5,)
        # All should be valid log-softmax outputs (sum of exp ≈ 1)
        assert torch.exp(combined).sum().item() == pytest.approx(1.0, abs=1e-5)
        assert torch.exp(heuristic_only).sum().item() == pytest.approx(1.0, abs=1e-5)
        assert torch.exp(model_only).sum().item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# MCTS.log_prior_metrics
# ---------------------------------------------------------------------------


class TestLogPriorMetrics:
    def test_records_four_metrics(self):
        mcts = _make_mcts()
        pre = torch.log(torch.tensor([0.5, 0.3, 0.2]))
        post = torch.log(torch.tensor([0.6, 0.3, 0.1]))
        step_log = StepLog()
        mcts.log_prior_metrics(pre, post, step_log)
        d = step_log.to_dict()
        assert "EnvStep/prior_entropy" in d
        assert "EnvStep/prior_confidence" in d
        assert "EnvStep/prior_entropy_heuristic" in d
        assert "EnvStep/prior_confidence_heuristic" in d

    def test_confidence_is_max_prob(self):
        mcts = _make_mcts()
        pre = torch.log(torch.tensor([0.1, 0.7, 0.2]))
        post = torch.log(torch.tensor([0.5, 0.3, 0.2]))
        step_log = StepLog()
        mcts.log_prior_metrics(pre, post, step_log)
        d = step_log.to_dict()
        assert d["EnvStep/prior_confidence"] == pytest.approx(0.7, abs=1e-5)
        assert d["EnvStep/prior_confidence_heuristic"] == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# MCTS.search_state_snapshot
# ---------------------------------------------------------------------------


class TestSearchStateSnapshot:
    def test_contains_all_keys(self):
        mcts = _make_mcts()
        snap = mcts.search_state_snapshot(step=5)
        expected_keys = {
            "SearchState/num_expanded_nodes",
            "SearchState/num_original_nodes",
            "SearchState/num_nodes",
            "SearchState/num_deduplicated_nodes",
            "SearchState/deduplicated_ratio",
            "SearchState/num_contradictory_nodes",
            "SearchState/avg_expanded_branching",
            "SearchState/max_expanded_depth",
            "SearchState/virtual_materializations",
            "SearchState/failures",
            "SearchState/step",
        }
        assert set(snap.keys()) == expected_keys
        assert snap["SearchState/step"] == 5

    def test_reflects_mcts_state(self):
        mcts = _make_mcts()
        mcts.num_expanded_nodes = 7
        mcts.failures = 2
        mcts.max_expanded_depth = 3
        snap = mcts.search_state_snapshot(step=0)
        assert snap["SearchState/num_expanded_nodes"] == 7
        assert snap["SearchState/failures"] == 2
        assert snap["SearchState/max_expanded_depth"] == 3


# ---------------------------------------------------------------------------
# MCTS.report_budget_exhausted
# ---------------------------------------------------------------------------


class TestReportBudgetExhausted:
    def test_puts_result_on_queue(self):
        mcts = _make_mcts(search_budget=100)
        mcts.env_steps = 50
        mcts.max_last_done_node = 10
        mcts.report_budget_exhausted()
        mcts.search_result_queue.put.assert_called_once()
        config, result = mcts.search_result_queue.put.call_args[0][0]
        assert config is mcts.lemma_config
        assert result["solved"] is False
        assert result["contradictory"] is False
        assert result["steps"] == 50
        assert result["dry_spell"] == 10
        assert result["new_budget"] == 150  # 100 * 1.5


# ---------------------------------------------------------------------------
# MCTS.expand_node
# ---------------------------------------------------------------------------


class TestExpandNode:
    def _make_expand_args(self, n_methods: int = 2):
        logprobs = torch.zeros(n_methods)
        return logprobs, logprobs.clone(), logprobs.clone()

    def test_single_case_creates_or_child(self):
        mcts = _make_mcts()
        node = _or_node(n_methods=2)
        step_log = StepLog()
        child_response = _response_state(n_methods=2)

        mcts.env.perform_action_bulk.return_value = (
            [{"case0": child_response}, {"case1": child_response}],
            [],  # no failures
            0.5,  # tamarin_call_time
        )

        prior, heur, model = self._make_expand_args(n_methods=2)
        mcts.expand_node(node, prior, heur, model, step_log)

        assert node.is_expanded
        assert len(node.children) == 2
        assert mcts.num_expanded_nodes == 1
        d = step_log.to_dict()
        assert d["EnvStep/num_proof_methods"] == 2
        assert d["EnvStep/tamarin_call_time"] == 0.5
        assert "EnvStep/expand_time" in d
        assert node.expand_time > 0

    def test_zero_cases_makes_contradictory(self):
        mcts = _make_mcts()
        node = _or_node(n_methods=1)
        node.nn_value = 0.5  # required before expand_node (set by __call__)
        step_log = StepLog()

        mcts.env.perform_action_bulk.return_value = (
            [{}],  # zero cases = contradictory
            [],
            0.1,
        )

        prior, heur, model = self._make_expand_args(n_methods=1)
        mcts.expand_node(node, prior, heur, model, step_log)

        child = node.children["enc0"]
        assert child.is_contradictory
        assert child.is_terminal
        assert len(mcts.contradictory_nodes_per_step) >= 1

    def test_failure_creates_failed_dummy(self):
        mcts = _make_mcts()
        node = _or_node(n_methods=2)
        step_log = StepLog()
        child_response = _response_state(n_methods=2)

        mcts.env.perform_action_bulk.return_value = (
            [{"case0": child_response}],  # only one outcome (for method 1)
            [0],  # method 0 failed
            0.1,
        )

        prior, heur, model = self._make_expand_args(n_methods=2)
        mcts.expand_node(node, prior, heur, model, step_log)

        assert node.children["enc0"].failed
        assert not node.children["enc1"].failed
        assert mcts.failures == 1

    def test_multiple_cases_creates_and_node(self):
        mcts = _make_mcts()
        node = _or_node(n_methods=1)
        step_log = StepLog()
        child_a = _response_state(n_methods=2)
        child_b = _response_state(n_methods=2)

        mcts.env.perform_action_bulk.return_value = (
            [{"case0": child_a, "case1": child_b}],
            [],
            0.1,
        )

        prior, heur, model = self._make_expand_args(n_methods=1)
        mcts.expand_node(node, prior, heur, model, step_log)

        and_child = node.children["enc0"]
        assert and_child.node_type == NodeType.AND
        assert len(and_child.children) == 2  # two cases expanded

    def test_all_failed_sets_node_failed(self):
        mcts = _make_mcts()
        node = _or_node(n_methods=2)
        step_log = StepLog()

        mcts.env.perform_action_bulk.return_value = (
            [],  # no outcomes
            [0, 1],  # both methods failed
            0.1,
        )

        prior, heur, model = self._make_expand_args(n_methods=2)
        mcts.expand_node(node, prior, heur, model, step_log)

        assert node.failed



# ---------------------------------------------------------------------------
# MCTS._propagate_child_status
# ---------------------------------------------------------------------------


class TestPropagateChildStatus:
    def test_contradictory_child_updates_parent(self):
        mcts = _make_mcts()
        parent = _expanded_or_node(n_children=1, n_methods=1)
        parent.children_visit_count["enc0"] = 1
        mcts.contradictory_nodes_per_step = []
        mcts._propagate_child_status(parent)
        assert parent.is_contradictory
        assert mcts.last_done_node == 0

    def test_no_done_children_no_update(self):
        mcts = _make_mcts()
        parent = _or_node(n_methods=2)
        # Add non-contradictory, non-solved children
        for i in range(2):
            child = _or_node(n_methods=1)
            child.children["enc0"] = _terminal_or_node()
            child._nn_value = 0.5
            child._value = 0.5
            parent.children[f"enc{i}"] = child
            parent.children_prior[f"enc{i}"] = 0.5
            parent.children_prior_only_heuristic[f"enc{i}"] = 0.5
            parent.children_prior_only_model[f"enc{i}"] = 0.5
        parent._nn_value = 0.5
        parent._value = 0.5
        mcts.contradictory_nodes_per_step = []
        mcts.last_done_node = 5
        mcts._propagate_child_status(parent)
        assert not parent.is_contradictory
        assert mcts.last_done_node == 5  # unchanged

    def test_all_failed_propagates(self):
        mcts = _make_mcts()
        parent = _or_node(n_methods=2)
        for i in range(2):
            child = _or_node()
            child.failed = True
            parent.children[f"enc{i}"] = child
        mcts.contradictory_nodes_per_step = []
        mcts._propagate_child_status(parent)
        assert parent.failed


# ---------------------------------------------------------------------------
# MCTS.backpropagate_and_collect
# ---------------------------------------------------------------------------


class TestBackpropagateAndCollect:
    def test_contradictory_leaf_propagates_up(self):
        """When the leaf was made contradictory by expand_node, backpropagation
        propagates contradictory status up through the search path and extracts
        training examples."""
        mcts = _make_mcts()
        # Build: root -> leaf (expanded, contradictory after expand_node)
        root = _or_node(n_methods=1)
        # Simulate expand_node having created root's child
        leaf = _expanded_or_node(n_children=1, n_methods=1)
        # _expanded_or_node creates a node with a contradictory terminal child
        # and _propagate_child_status would have made it contradictory.
        # Simulate that here:
        leaf._is_contradictory = True
        root.children["enc0"] = leaf
        root.children_prior["enc0"] = 1.0
        root.children_prior_only_heuristic["enc0"] = 1.0
        root.children_prior_only_model["enc0"] = 1.0
        root.children_visit_count["enc0"] = 1
        leaf.parents.append(root)
        root._nn_value = 0.5
        root._value = 0.5

        search_path = [(None, root), (0, leaf)]
        step_log = StepLog()

        done_root = mcts.backpropagate_and_collect(leaf, search_path, step_log)

        # leaf is contradictory → root becomes contradictory via update()
        assert done_root is root
        assert root.is_contradictory
        d = step_log.to_dict()
        # done_length = 2 (root + leaf)
        assert d["EnvStep/done_length"] == 2
        assert d["EnvStep/contradictory_share"] == pytest.approx(1.0)
        assert d["EnvStep/num_collected_samples"] >= 1
        mcts.training_example_queue.put.assert_called_once()

    def test_unexpanded_leaf_trimmed_from_path(self):
        """When the leaf is unexpanded (e.g. stale DAG node), it is trimmed from
        the search path before backpropagation."""
        mcts = _make_mcts()
        root = _or_node(n_methods=1)
        child = _or_node(n_methods=1)
        root.children["enc0"] = child
        root.children_prior["enc0"] = 1.0
        root.children_prior_only_heuristic["enc0"] = 1.0
        root.children_prior_only_model["enc0"] = 1.0
        child.parents.append(root)
        root._nn_value = 0.5
        root._value = 0.5
        # child is unexpanded → gets trimmed; backprop only runs on root
        search_path = [(None, root), (0, child)]
        step_log = StepLog()

        done_root = mcts.backpropagate_and_collect(child, search_path, step_log)

        # root's child isn't contradictory/solved → no done_root
        assert done_root is None
        d = step_log.to_dict()
        assert "EnvStep/done_length" not in d
        assert d["EnvStep/num_collected_samples"] == 0

    def test_no_done_root_skips_done_logging(self):
        """When no node becomes contradictory/solved, done_length and
        contradictory_share are NOT logged."""
        mcts = _make_mcts()
        # Build: root -> leaf (expanded, with a non-contradictory child)
        root = _or_node(n_methods=1)
        leaf = _or_node(n_methods=1)
        grandchild = _or_node(n_methods=1)
        # Make leaf expanded with a non-contradictory child
        leaf.children["enc0"] = grandchild
        leaf.children_prior["enc0"] = 1.0
        leaf.children_prior_only_heuristic["enc0"] = 1.0
        leaf.children_prior_only_model["enc0"] = 1.0
        grandchild.parents.append(leaf)
        leaf._nn_value = 0.5
        leaf._value = 0.5
        # Wire root -> leaf
        root.children["enc0"] = leaf
        root.children_prior["enc0"] = 1.0
        root.children_prior_only_heuristic["enc0"] = 1.0
        root.children_prior_only_model["enc0"] = 1.0
        leaf.parents.append(root)
        root._nn_value = 0.5
        root._value = 0.5
        root.children_visit_count["enc0"] = 1

        search_path = [(None, root), (0, leaf)]
        step_log = StepLog()

        done_root = mcts.backpropagate_and_collect(leaf, search_path, step_log)

        assert done_root is None
        d = step_log.to_dict()
        assert "EnvStep/done_length" not in d
        assert "EnvStep/contradictory_share" not in d
        assert d["EnvStep/num_collected_samples"] == 0


# ---------------------------------------------------------------------------
# MCTS.__call__ (integration)
# ---------------------------------------------------------------------------


class TestMCTSCall:
    def _setup_single_step_proof(self, mcts: MCTS) -> None:
        """Configure mocks so that the first step finds a contradictory proof."""
        init_state = _state(n_methods=1)
        mcts.env.get_initial_state.return_value = init_state

        # Inference returns uniform logprobs and value
        mcts.response_conn.poll.return_value = True

        def mock_recv():
            return ("req-id", torch.zeros(1), 0.5)

        mcts.response_conn.recv.side_effect = mock_recv

        # Expansion: single method yields zero cases = contradictory
        mcts.env.perform_action_bulk.return_value = (
            [{}],  # zero cases = contradictory
            [],
            0.1,
        )
        mcts.env.check_proof.return_value = "proof_file_content"

    @patch("src.rl.mcts.uuid.uuid4")
    def test_immediate_proof(self, mock_uuid):
        mock_uuid.return_value = MagicMock(__str__=lambda _: "req-id")
        mcts = _make_mcts(search_budget=10)
        self._setup_single_step_proof(mcts)

        mcts()

        # Should have reported success
        mcts.search_result_queue.put.assert_called()
        call_args = mcts.search_result_queue.put.call_args[0][0]
        _, result = call_args
        assert result["contradictory"] is True
        assert result["proof_file"] == "proof_file_content"
        assert mcts.env_steps == 1

    @patch("src.rl.mcts.uuid.uuid4")
    def test_budget_exhausted(self, mock_uuid):
        mock_uuid.return_value = MagicMock(__str__=lambda _: "req-id")
        mcts = _make_mcts(search_budget=2)
        init_state = _state(n_methods=1)
        mcts.env.get_initial_state.return_value = init_state

        # Inference returns logprobs (fresh tensor each call)
        mcts.response_conn.poll.return_value = True
        mcts.response_conn.recv.side_effect = lambda: ("req-id", torch.zeros(1), 0.5)

        # Each expansion: single method yields one non-terminal case
        def make_bulk_result(*args, **kwargs):
            child_response = _response_state(n_methods=1, status="InProgress")
            return ([{"case0": child_response}], [], 0.1)

        mcts.env.perform_action_bulk.side_effect = make_bulk_result

        mcts()

        # Last call should be budget exhausted
        last_call = mcts.search_result_queue.put.call_args[0][0]
        _, result = last_call
        assert result["solved"] is False
        assert result["contradictory"] is False
        assert result["new_budget"] == 3  # 2 * 1.5

    @patch("src.rl.mcts.uuid.uuid4")
    def test_terminate_signal(self, mock_uuid):
        mock_uuid.return_value = MagicMock(__str__=lambda _: "req-id")
        mcts = _make_mcts(search_budget=10)
        init_state = _state(n_methods=1)
        mcts.env.get_initial_state.return_value = init_state

        # Inference returns None (terminate)
        mcts.response_conn.poll.return_value = True
        mcts.response_conn.recv.return_value = None

        mcts()

        mcts.search_result_queue.put.assert_called_once()
        _, result = mcts.search_result_queue.put.call_args[0][0]
        assert result["stopped_early"] is True
        assert mcts.env_steps == 1

    @patch("src.rl.mcts.uuid.uuid4")
    def test_step_log_emitted_each_step(self, mock_uuid):
        mock_uuid.return_value = MagicMock(__str__=lambda _: "req-id")
        mcts = _make_mcts(search_budget=2)
        init_state = _state(n_methods=1)
        mcts.env.get_initial_state.return_value = init_state

        mcts.response_conn.poll.return_value = True
        mcts.response_conn.recv.side_effect = lambda: ("req-id", torch.zeros(1), 0.5)

        def make_bulk_result(*args, **kwargs):
            child_response = _response_state(n_methods=1, status="InProgress")
            return ([{"case0": child_response}], [], 0.1)

        mcts.env.perform_action_bulk.side_effect = make_bulk_result

        mcts()

        # step_log_queue should have been called once per step
        assert mcts.step_log_queue.put.call_count == 2


# ---------------------------------------------------------------------------
# _resolve_first_step
# ---------------------------------------------------------------------------


class TestResolveFirstStep:
    def test_heuristic_returns_zero(self):
        mcts = _make_mcts()
        mcts.first_step = "heuristic"
        state = State(
            proof_methods=[{"name": "Simplify"}, {"name": "Induction"}],
            encoded_methods=("enc_s", "enc_i"),
            encoded_sys="sys",
        )
        assert mcts._resolve_first_step(state) == 0

    def test_simplify_finds_correct_index(self):
        mcts = _make_mcts()
        mcts.first_step = "simplify"
        state = State(
            proof_methods=[{"name": "Induction"}, {"name": "Simplify"}],
            encoded_methods=("enc_i", "enc_s"),
            encoded_sys="sys",
        )
        assert mcts._resolve_first_step(state) == 1

    def test_induction_finds_correct_index(self):
        mcts = _make_mcts()
        mcts.first_step = "induction"
        state = State(
            proof_methods=[{"name": "Simplify"}, {"name": "Induction"}],
            encoded_methods=("enc_s", "enc_i"),
            encoded_sys="sys",
        )
        assert mcts._resolve_first_step(state) == 1

    def test_single_method_returns_zero(self):
        """When only one method exists, pick it regardless of first_step."""
        mcts = _make_mcts()
        mcts.first_step = "induction"
        state = State(
            proof_methods=[{"name": "Simplify"}],
            encoded_methods=("enc_s",),
            encoded_sys="sys",
        )
        assert mcts._resolve_first_step(state) == 0

    def test_missing_method_raises(self):
        mcts = _make_mcts()
        mcts.first_step = "induction"
        state = State(
            proof_methods=[
                {"name": "Simplify"},
                {"goal": {"type": "Premise", "args": []}},
            ],
            encoded_methods=("enc_s", "enc_p"),
            encoded_sys="sys",
        )
        with pytest.raises(ValueError, match="Induction.*not found"):
            mcts._resolve_first_step(state)


# ---------------------------------------------------------------------------
# MCTS.__call__ with first_step
# ---------------------------------------------------------------------------


def _state_with_simplify_and_induction() -> State:
    """Initial state with Simplify (idx 0) and Induction (idx 1)."""
    return State(
        proof_methods=[{"name": "Simplify"}, {"name": "Induction"}],
        encoded_methods=("enc_s", "enc_i"),
        encoded_sys="sys",
    )


def _post_first_step_response(
    status: str = "InProgress",
) -> TamarinEnvironment.ResponseState:
    return TamarinEnvironment.ResponseState(
        proof_methods=[
            {"goal": {"type": "Premise", "args": []}},
        ],
        encoded_methods=["enc_premise"],
        encoded_sys="sys_post",
        status=status,
    )


class TestMCTSCallFirstStep:
    @patch("src.rl.mcts.uuid.uuid4")
    def test_first_step_simplify_applies_before_search(self, mock_uuid):
        """With first_step='simplify', perform_action is called with the
        Simplify method to pre-expand the root."""
        mock_uuid.return_value = MagicMock(__str__=lambda _: "req-id")
        mcts = _make_mcts(search_budget=1)
        mcts.first_step = "simplify"

        init_state = _state_with_simplify_and_induction()
        mcts.env.get_initial_state.return_value = init_state

        post_response = _post_first_step_response()
        mcts.env.perform_action.return_value = {"simplify": post_response}

        # Search loop: inference + expand on the post-first-step child
        mcts.response_conn.poll.return_value = True
        mcts.response_conn.recv.side_effect = lambda: (
            "req-id",
            torch.zeros(1),
            0.5,
        )

        def make_bulk_result(*args, **kwargs):
            child = _response_state(n_methods=1, status="InProgress")
            return ([{"case0": child}], [], 0.1)

        mcts.env.perform_action_bulk.side_effect = make_bulk_result

        mcts()

        # perform_action should have been called with (init_state, 0)
        # because Simplify is at index 0
        mcts.env.perform_action.assert_called_once_with(init_state, 0)

    @patch("src.rl.mcts.uuid.uuid4")
    def test_first_step_induction_picks_correct_method(self, mock_uuid):
        mock_uuid.return_value = MagicMock(__str__=lambda _: "req-id")
        mcts = _make_mcts(search_budget=1)
        mcts.first_step = "induction"

        init_state = _state_with_simplify_and_induction()
        mcts.env.get_initial_state.return_value = init_state

        # Induction produces 2 cases (AND node)
        post_response = _post_first_step_response()
        mcts.env.perform_action.return_value = {
            "base_case": post_response,
            "step_case": post_response,
        }

        mcts.response_conn.poll.return_value = True
        mcts.response_conn.recv.side_effect = lambda: (
            "req-id",
            torch.zeros(1),
            0.5,
        )

        def make_bulk_result(*args, **kwargs):
            child = _response_state(n_methods=1, status="InProgress")
            return ([{"case0": child}], [], 0.1)

        mcts.env.perform_action_bulk.side_effect = make_bulk_result

        mcts()

        # Induction is at index 1
        mcts.env.perform_action.assert_called_once_with(init_state, 1)

    @patch("src.rl.mcts.uuid.uuid4")
    def test_first_step_heuristic_picks_index_zero(self, mock_uuid):
        mock_uuid.return_value = MagicMock(__str__=lambda _: "req-id")
        mcts = _make_mcts(search_budget=1)
        mcts.first_step = "heuristic"

        init_state = _state_with_simplify_and_induction()
        mcts.env.get_initial_state.return_value = init_state

        post_response = _post_first_step_response()
        mcts.env.perform_action.return_value = {"simplify": post_response}

        mcts.response_conn.poll.return_value = True
        mcts.response_conn.recv.side_effect = lambda: (
            "req-id",
            torch.zeros(1),
            0.5,
        )

        def make_bulk_result(*args, **kwargs):
            child = _response_state(n_methods=1, status="InProgress")
            return ([{"case0": child}], [], 0.1)

        mcts.env.perform_action_bulk.side_effect = make_bulk_result

        mcts()

        mcts.env.perform_action.assert_called_once_with(init_state, 0)

    def test_first_step_terminal_reports_immediately(self):
        """If the forced first step makes the proof terminal (contradictory),
        report success and return without entering the search loop."""
        mcts = _make_mcts(search_budget=10)
        mcts.first_step = "simplify"

        init_state = _state_with_simplify_and_induction()
        mcts.env.get_initial_state.return_value = init_state

        terminal_response = _post_first_step_response(
            status="Contradictory"
        )
        mcts.env.perform_action.return_value = {
            "simplify": terminal_response
        }
        mcts.env.check_proof.return_value = "proof_file_content"

        mcts()

        # No search steps should have run
        assert mcts.env_steps == 0
        mcts.env.perform_action_bulk.assert_not_called()
        mcts.response_conn.poll.assert_not_called()

        # Should have reported success
        call_args = mcts.search_result_queue.put.call_args[0][0]
        _, result = call_args
        assert result["contradictory"] is True

    def test_first_step_failure_asserts(self):
        """If perform_action returns None, assert fires."""
        mcts = _make_mcts(search_budget=1)
        mcts.first_step = "simplify"

        init_state = _state_with_simplify_and_induction()
        mcts.env.get_initial_state.return_value = init_state
        mcts.env.perform_action.return_value = None

        with pytest.raises(AssertionError, match="failed to apply"):
            mcts()

    @patch("src.rl.mcts.uuid.uuid4")
    def test_first_step_multiple_cases_creates_and_node(self, mock_uuid):
        """If first step produces multiple cases, an AND node is created."""
        mock_uuid.return_value = MagicMock(__str__=lambda _: "req-id")
        mcts = _make_mcts(search_budget=1)
        mcts.first_step = "induction"

        init_state = _state_with_simplify_and_induction()
        mcts.env.get_initial_state.return_value = init_state

        post_resp = _post_first_step_response()
        mcts.env.perform_action.return_value = {
            "base": post_resp,
            "step": post_resp,
        }

        mcts.response_conn.poll.return_value = True
        mcts.response_conn.recv.side_effect = lambda: (
            "req-id",
            torch.zeros(1),
            0.5,
        )

        def make_bulk_result(*args, **kwargs):
            child = _response_state(n_methods=1, status="InProgress")
            return ([{"case0": child}], [], 0.1)

        mcts.env.perform_action_bulk.side_effect = make_bulk_result

        mcts()

        # Root should have been pre-expanded with one AND child
        mcts.env.perform_action.assert_called_once()

    @patch("src.rl.mcts.uuid.uuid4")
    def test_first_step_search_unchanged(self, mock_uuid):
        """With first_step='search' (default), perform_action should not
        be called — the old behavior is preserved."""
        mock_uuid.return_value = MagicMock(__str__=lambda _: "req-id")
        mcts = _make_mcts(search_budget=1)
        # first_step defaults to "search"

        init_state = _state_with_simplify_and_induction()
        mcts.env.get_initial_state.return_value = init_state

        mcts.response_conn.poll.return_value = True
        mcts.response_conn.recv.side_effect = lambda: (
            "req-id",
            torch.zeros(2),
            0.5,
        )

        def make_bulk_result(*args, **kwargs):
            child = _response_state(n_methods=2, status="InProgress")
            return (
                [{"case0": child}, {"case1": child}],
                [],
                0.1,
            )

        mcts.env.perform_action_bulk.side_effect = make_bulk_result

        mcts()

        # perform_action (singular) should NOT have been called
        mcts.env.perform_action.assert_not_called()
        # perform_action_bulk should have been called (normal expand)
        mcts.env.perform_action_bulk.assert_called()


# ---------------------------------------------------------------------------
# _apply_forced_first_step
# ---------------------------------------------------------------------------


class TestApplyForcedFirstStep:
    def test_single_case_creates_or_child(self):
        mcts = _make_mcts()
        mcts.first_step = "simplify"
        root = Node(_state_with_simplify_and_induction(), NodeType.OR, [], 0)

        resp = _post_first_step_response()
        mcts.env.perform_action.return_value = {"simplify": resp}

        mcts._apply_forced_first_step(root)

        assert root.is_expanded
        assert len(root.children) == 1
        assert "enc_s" in root.children
        child = root.children["enc_s"]
        assert child.node_type == NodeType.OR
        assert not child.is_contradictory

    def test_multiple_cases_creates_and_node(self):
        mcts = _make_mcts()
        mcts.first_step = "induction"
        root = Node(_state_with_simplify_and_induction(), NodeType.OR, [], 0)

        resp = _post_first_step_response()
        mcts.env.perform_action.return_value = {
            "base": resp,
            "step": resp,
        }

        mcts._apply_forced_first_step(root)

        assert root.is_expanded
        assert len(root.children) == 1
        assert "enc_i" in root.children
        and_child = root.children["enc_i"]
        assert and_child.node_type == NodeType.AND
        assert len(and_child.children) == 2

    def test_zero_cases_makes_contradictory(self):
        mcts = _make_mcts()
        mcts.first_step = "simplify"
        state = State(
            proof_methods=[{"name": "Simplify"}],
            encoded_methods=("enc_s",),
            encoded_sys="sys",
        )
        root = Node(state, NodeType.OR, [], 0)

        mcts.env.perform_action.return_value = {}

        mcts._apply_forced_first_step(root)

        assert root.is_expanded
        child = root.children["enc_s"]
        assert child.is_contradictory
        assert child.is_terminal
        assert root.is_contradictory

    def test_failure_asserts(self):
        mcts = _make_mcts()
        mcts.first_step = "simplify"
        state = State(
            proof_methods=[{"name": "Simplify"}],
            encoded_methods=("enc_s",),
            encoded_sys="sys",
        )
        root = Node(state, NodeType.OR, [], 0)

        mcts.env.perform_action.return_value = None

        with pytest.raises(AssertionError, match="failed to apply"):
            mcts._apply_forced_first_step(root)
