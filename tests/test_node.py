import math

import pytest

from src.parser.reward import RewardConfig
from src.parser.ucb import UCBConfig
from src.rl.node import (
    DEVALUE_SCORE,
    MAX_VALUE,
    Node,
    NodeType,
    UCBScore,
    reward_function,
    ucb_score,
)

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


# ---------------------------------------------------------------------------
# NodeType
# ---------------------------------------------------------------------------


class TestNodeType:
    def test_values(self):
        assert NodeType.OR.value == 1
        assert NodeType.AND.value == 2


# ---------------------------------------------------------------------------
# Node construction and properties
# ---------------------------------------------------------------------------


class TestNodeInit:
    def test_defaults(self):
        node = _or_node()
        assert node.node_type == NodeType.OR
        assert node.children == {}
        assert not node.is_terminal
        assert not node.is_solved
        assert not node.failed
        assert not node.has_value
        assert node.visit_count == 0
        assert not node.is_expanded

    def test_terminal_contradictory(self):
        node = _terminal_or_node()
        assert node.is_terminal
        assert node.is_contradictory
        assert node.value == MAX_VALUE

    def test_terminal_solved(self):
        node = _terminal_or_node(is_contradictory=False, is_solved=True)
        assert node.is_terminal
        assert node.is_solved
        assert node.value == MAX_VALUE


class TestValueProperty:
    def test_terminal_returns_max_value(self):
        node = _terminal_or_node()
        assert node.value == MAX_VALUE

    def test_unexpanded_raises(self):
        node = _or_node()
        with pytest.raises(AssertionError):
            _ = node.value

    def test_unvisited_expanded_raises(self):
        # Expanded (has children) but visit_count=0 for an AND node
        node = _and_node()
        child = _terminal_or_node()
        node.children["c"] = child
        # AND node with no visits and no _value set
        with pytest.raises(AssertionError):
            _ = node.value

    def test_set_and_get(self):
        node = _or_node()
        # Make it look expanded and visited
        child = _terminal_or_node()
        node.children["enc0"] = child
        node.children_visit_count["enc0"] = 1
        node._nn_value = 0.5
        node.value = 0.8
        assert node.value == 0.8


class TestNNValue:
    def test_only_or_nodes(self):
        node = _and_node()
        with pytest.raises(AssertionError):
            node.nn_value = 0.5

    def test_sets_value_on_first_assignment(self):
        node = _or_node()
        child = _terminal_or_node()
        node.children["enc0"] = child
        node.nn_value = 0.7
        assert node.nn_value == 0.7
        assert node._value == 0.7

    def test_cannot_set_when_value_exists(self):
        node = _or_node()
        child = _terminal_or_node()
        node.children["enc0"] = child
        node._value = 0.3
        with pytest.raises(AssertionError):
            node.nn_value = 0.5


class TestVisitCount:
    def test_unexpanded_is_zero(self):
        node = _or_node()
        assert node.visit_count == 0

    def test_expanded_or_starts_at_one(self):
        node = _or_node()
        node.children["enc0"] = _terminal_or_node()
        # Expanded OR node counts the expansion itself as 1 visit
        assert node.visit_count == 1

    def test_expanded_and_starts_at_zero(self):
        node = _and_node()
        node.children["c"] = _terminal_or_node()
        # AND nodes don't count expansion as a visit
        assert node.visit_count == 0

    def test_increments_with_child_visits(self):
        node = _or_node()
        node.children["enc0"] = _terminal_or_node()
        node.children_visit_count["enc0"] = 3
        # 3 child visits + 1 for expansion = 4
        assert node.visit_count == 4


class TestIsExpanded:
    def test_no_children(self):
        assert not _or_node().is_expanded

    def test_with_children(self):
        node = _or_node()
        node.children["enc0"] = _terminal_or_node()
        assert node.is_expanded


class TestStateProperty:
    def test_or_node_returns_state(self):
        node = _or_node()
        assert node.state == _state()

    def test_and_node_raises(self):
        node = _and_node()
        with pytest.raises(AssertionError):
            _ = node.state


class TestPriorSums:
    def test_empty(self):
        node = _or_node()
        assert node.prior_sum == 0.0
        assert node.prior_sum_only_heuristic == 0.0
        assert node.prior_sum_only_model == 0.0

    def test_sums(self):
        node = _or_node()
        node.children_prior = {"a": 0.3, "b": 0.7}
        node.children_prior_only_heuristic = {"a": 0.4, "b": 0.6}
        node.children_prior_only_model = {"a": 0.5, "b": 0.5}
        assert node.prior_sum == pytest.approx(1.0)
        assert node.prior_sum_only_heuristic == pytest.approx(1.0)
        assert node.prior_sum_only_model == pytest.approx(1.0)


class TestRepr:
    def test_contains_type(self):
        r = repr(_or_node())
        assert "NodeType.OR" in r


# ---------------------------------------------------------------------------
# Node.update (OR)
# ---------------------------------------------------------------------------


class TestUpdateOR:
    def test_propagates_solved_from_child(self):
        parent = _expanded_or_node(n_children=2)
        parent.children["enc0"].is_solved = True
        parent.children_visit_count["enc0"] = 1
        parent.update(_reward_config(), tamarin_timeout=60)
        assert parent.is_solved

    def test_propagates_contradictory(self):
        parent = _expanded_or_node(n_children=1)
        parent.children_visit_count["enc0"] = 1
        became = parent.update(_reward_config(), tamarin_timeout=60)
        assert parent.is_contradictory
        assert became is True

    def test_returns_false_if_already_contradictory(self):
        parent = _expanded_or_node(n_children=1)
        parent._is_contradictory = True
        parent.children_visit_count["enc0"] = 1
        became = parent.update(_reward_config(), tamarin_timeout=60)
        assert became is False

    def test_all_failed_propagates(self):
        parent = _expanded_or_node(n_children=2)
        for child in parent.children.values():
            child.failed = True
        parent.update(_reward_config(), tamarin_timeout=60)
        assert parent.failed

    def test_not_all_failed(self):
        parent = _expanded_or_node(n_children=2)
        parent.children["enc0"].failed = True
        parent.update(_reward_config(), tamarin_timeout=60)
        assert not parent.failed

    def test_value_computation_with_visits(self):
        parent = _expanded_or_node(n_children=1, n_methods=1)
        parent.children_visit_count["enc0"] = 2
        parent.update(_reward_config(), tamarin_timeout=60)
        # value = (nn_value + visits * (reward + child_value)) / (1 + visit_count)
        # reward for terminal child = -(0 + 0 + 0 + 1) = -1
        # child value = MAX_VALUE = 1.0
        # path_value = -1 + 1.0 = 0.0
        # weighted_sum = 0.5 + 2 * 0.0 = 0.5
        # visit_count = 2 (child visits) + 1 (expansion) = 3
        # value = 0.5 / (1 + 3) = 0.125
        assert parent.value == pytest.approx(0.125)


# ---------------------------------------------------------------------------
# Node.update (AND)
# ---------------------------------------------------------------------------


class TestUpdateAND:
    def _make_and_with_children(self, n=2) -> Node:
        parent = _and_node()
        for i in range(n):
            child = _or_node(n_methods=2)
            child.children[f"enc0"] = _terminal_or_node()
            child._nn_value = 0.5
            child._value = 0.5
            child.children_visit_count["enc0"] = 1
            key = f"case{i}"
            parent.children[key] = child
            parent.children_visit_count[key] = 1
            child.parents.append(parent)
        return parent

    def test_all_contradictory_propagates(self):
        parent = self._make_and_with_children(2)
        for child in parent.children.values():
            child._is_contradictory = True
        became = parent.update(_reward_config(), tamarin_timeout=60)
        assert parent.is_contradictory
        assert became is True

    def test_not_all_contradictory(self):
        parent = self._make_and_with_children(2)
        list(parent.children.values())[0]._is_contradictory = True
        parent.update(_reward_config(), tamarin_timeout=60)
        assert not parent.is_contradictory

    def test_solved_child_propagates(self):
        parent = self._make_and_with_children(2)
        solved_child = list(parent.children.values())[0]
        solved_child.is_solved = True
        solved_child.value = 0.9
        parent.update(_reward_config(), tamarin_timeout=60)
        assert parent.is_solved
        assert parent.value == 0.9

    def test_value_is_min_of_visited_non_contradictory(self):
        parent = self._make_and_with_children(3)
        children = list(parent.children.values())
        children[0].value = 0.8
        children[1].value = 0.3
        children[2].value = 0.6
        parent.update(_reward_config(), tamarin_timeout=60)
        assert parent.value == pytest.approx(0.3)

    def test_contradictory_children_excluded_from_min(self):
        parent = self._make_and_with_children(2)
        children = list(parent.children.values())
        children[0].value = 0.2
        children[0]._is_contradictory = True
        children[1].value = 0.8
        parent.update(_reward_config(), tamarin_timeout=60)
        assert parent.value == pytest.approx(0.8)

    def test_no_visited_children_gives_max_value(self):
        parent = _and_node()
        # Add unexpanded children (visit_count=0)
        for i in range(2):
            child = _or_node(n_methods=1)
            key = f"case{i}"
            parent.children[key] = child
            parent.children_visit_count[key] = 1
        parent.update(_reward_config(), tamarin_timeout=60)
        # All children have visit_count=0 (unexpanded OR), so optimistic MAX_VALUE
        assert parent._value == MAX_VALUE

    def test_all_failed_propagates(self):
        parent = self._make_and_with_children(2)
        for child in parent.children.values():
            child.failed = True
        parent.update(_reward_config(), tamarin_timeout=60)
        assert parent.failed


# ---------------------------------------------------------------------------
# reward_function
# ---------------------------------------------------------------------------


class TestRewardFunction:
    def test_and_child_no_penalties(self):
        parent = _expanded_or_node(n_children=1)
        and_child = _and_node()
        r = reward_function(parent, and_child, _reward_config(), tamarin_timeout=60)
        assert r == -1.0  # -(0 + 0 + 0 + 1)

    def test_terminal_child_no_penalties(self):
        parent = _expanded_or_node(n_children=1)
        child = _terminal_or_node()
        r = reward_function(parent, child, _reward_config(), tamarin_timeout=60)
        assert r == -1.0

    def test_branch_penalty(self):
        parent = _expanded_or_node(n_children=1, n_methods=4)
        parent.max_branching_prefix = 4
        child = _or_node(n_methods=8)
        # Not terminal, not AND
        cfg = _reward_config(branch_penalty=2.0)
        r = reward_function(parent, child, cfg, tamarin_timeout=60)
        # branch_penalty = (8 / max(4, 4)) * 2.0 = 4.0
        # total = -(0 * 0 + 2.0 * 4.0 + 0 + 1) = -(8 + 1) = -9.0
        assert r == pytest.approx(-9.0)

    def test_time_penalty(self):
        parent = _expanded_or_node(n_children=1, n_methods=2)
        child = _or_node(n_methods=1)
        child.expand_time = 30.0
        cfg = _reward_config(time_penalty=60.0)
        r = reward_function(parent, child, cfg, tamarin_timeout=60)
        # time_penalty = min(30, 180) / 180 = 1/6
        # total = -(60 * 1/6 + 0 + 0 + 1) = -(10 + 1) = -11
        assert r == pytest.approx(-11.0)

    def test_timeout_penalty(self):
        parent = _expanded_or_node(n_children=1, n_methods=2)
        child = _or_node(n_methods=1)
        child.expand_time = 75.0  # >= tamarin_timeout + 10 = 70
        cfg = _reward_config(timeout_penalty=5.0)
        r = reward_function(parent, child, cfg, tamarin_timeout=60)
        # timeout_penalty fires: 5.0
        # total = -(0 + 0 + 5 + 1) = -6.0
        assert r == pytest.approx(-6.0)

    def test_no_timeout_penalty_within_grace(self):
        parent = _expanded_or_node(n_children=1, n_methods=2)
        child = _or_node(n_methods=1)
        child.expand_time = 65.0  # < 60 + 10 = 70
        cfg = _reward_config(timeout_penalty=5.0)
        r = reward_function(parent, child, cfg, tamarin_timeout=60)
        assert r == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# ucb_score
# ---------------------------------------------------------------------------


class TestUCBScore:
    def _setup(self) -> tuple[Node, str, Node]:
        parent = _expanded_or_node(n_children=2, n_methods=2)
        parent.children_visit_count["enc0"] = 1
        action = "enc0"
        child = parent.children[action]
        return parent, action, child

    def test_returns_ucb_score_dataclass(self):
        parent, action, child = self._setup()
        result = ucb_score(_ucb_config(), _reward_config(), 60, parent, action, child)
        assert isinstance(result, UCBScore)

    def test_devalue_failed_child(self):
        parent, action, child = self._setup()
        child.failed = True
        result = ucb_score(_ucb_config(), _reward_config(), 60, parent, action, child)
        assert result.total_score == DEVALUE_SCORE

    def test_devalue_contradictory_and_child(self):
        parent = _and_node()
        parent.max_branching_prefix = 5
        child_a = _or_node(n_methods=1)
        child_a.children["enc0"] = _terminal_or_node()
        child_a._nn_value = 0.5
        child_a._value = 0.5
        child_a._is_contradictory = True
        child_b = _or_node(n_methods=1)
        child_b.children["enc0"] = _terminal_or_node()
        child_b._nn_value = 0.5
        child_b._value = 0.5
        parent.children = {"case0": child_a, "case1": child_b}
        parent.children_prior = {"case0": 0.5, "case1": 0.5}
        parent.children_prior_only_heuristic = {"case0": 0.5, "case1": 0.5}
        parent.children_prior_only_model = {"case0": 0.5, "case1": 0.5}
        parent.children_visit_count = {"case0": 1}
        result = ucb_score(_ucb_config(), _reward_config(), 60, parent, "case0", child_a)
        assert result.total_score == DEVALUE_SCORE

    def test_deactivate_model_uses_heuristic_only(self):
        parent, action, child = self._setup()
        cfg = _ucb_config(deactivate_model=True)
        result = ucb_score(cfg, _reward_config(), 60, parent, action, child)
        assert result.total_score == result.prior_score_heuristic

    def test_unvisited_child_uses_parent_value(self):
        parent = _expanded_or_node(n_children=2, n_methods=2)
        # enc0 not visited
        action = "enc0"
        child = parent.children[action]
        cfg = _ucb_config(value_penalty=2)
        result = ucb_score(cfg, _reward_config(), 60, parent, action, child)
        expected_value = parent.value - 2
        expected_value_score = cfg.value_discount ** (-1 - expected_value)
        assert result.value_score == pytest.approx(expected_value_score)

    def test_and_node_prior_scaled_by_c_and(self):
        parent = _and_node()
        child = _or_node(n_methods=1)
        parent.children = {"case0": child}
        parent.children_prior = {"case0": 1.0}
        parent.children_prior_only_heuristic = {"case0": 1.0}
        parent.children_prior_only_model = {"case0": 1.0}
        cfg = _ucb_config(c_and=10)
        result = ucb_score(cfg, _reward_config(), 60, parent, "case0", child)
        # For AND nodes, prior_score is multiplied by c_and
        # Compute what prior_score would be without c_and
        pb_c = (
            math.log((parent.visit_count + cfg.pb_c_base + 1) / cfg.pb_c_base)
            + cfg.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / 1  # no visits
        base_prior = pb_c * 1.0 / 1.0
        assert result.prior_score == pytest.approx(base_prior * 10)


# ---------------------------------------------------------------------------
# is_contradictory assertions
# ---------------------------------------------------------------------------


class TestIsContradictoryProperty:
    def test_or_node_requires_at_least_one_contradictory_child(self):
        node = _or_node()
        child = _or_node()
        node.children["enc0"] = child
        # Setting contradictory on OR with no contradictory children should fail
        with pytest.raises(AssertionError):
            node.is_contradictory = True

    def test_and_node_requires_all_contradictory_children(self):
        node = _and_node()
        child_a = _or_node()
        child_a._is_contradictory = True
        child_b = _or_node()  # not contradictory
        node.children = {"a": child_a, "b": child_b}
        with pytest.raises(AssertionError):
            node.is_contradictory = True

    def test_valid_or_contradictory(self):
        node = _or_node()
        child = _or_node()
        child._is_contradictory = True
        node.children["enc0"] = child
        node.is_contradictory = True
        assert node.is_contradictory

    def test_valid_and_contradictory(self):
        node = _and_node()
        child_a = _or_node()
        child_a._is_contradictory = True
        child_b = _or_node()
        child_b._is_contradictory = True
        node.children = {"a": child_a, "b": child_b}
        node.is_contradictory = True
        assert node.is_contradictory
