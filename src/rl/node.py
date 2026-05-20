import enum
import math
from dataclasses import dataclass

from src.environment.environment import State
from src.parser import RewardConfig, UCBConfig

# set the maximum value a node can get
MAX_VALUE = 1.0
DEVALUE_SCORE = -1e9


class NodeType(enum.Enum):
    OR = 1
    AND = 2


class Node:
    """A node in the MCTS search tree."""

    def __init__(
        self,
        state: State | None,
        node_type: NodeType,
        parents: list["Node"],
        max_branching_prefix: int,
        is_contradictory: bool = False,
        is_terminal: bool = False,
        is_solved: bool = False,
        failed: bool = False,
    ):
        self.node_type: NodeType = node_type
        self.children: dict[str, Node] = {}
        self._state: State | None = state

        self.case_names: dict[str, str | None] = {}

        self._is_contradictory: bool = is_contradictory
        self.is_terminal: bool = is_terminal
        self.is_solved: bool = is_solved
        self.failed: bool = failed
        self._value: float | None = None
        self._nn_value: float | None = None

        self.value_sum: float = 0.0
        self.max_branching_prefix: int = max_branching_prefix
        self.children_visit_count: dict[str, int] = {}
        self.children_prior: dict[str, float] = {}
        self.children_prior_only_heuristic: dict[str, float] = {}
        self.children_prior_only_model: dict[str, float] = {}

        self.parents: list["Node"] = parents

        self.virtual_actions: dict[str, int] = {}
        self.expand_time = 0.0

    @property
    def value(self) -> float:
        """Returns the value of the node."""
        if self.is_terminal:
            assert (
                self.is_solved or self.is_contradictory
            ), "Terminal node must be solved or contradictory."
            return MAX_VALUE
        if not self.is_expanded:
            print(f"Trying to access value of unexpanded node: {self}")
            assert False, f"Cannot access value of unexpanded node. Node: {self}"
        assert (
            self.visit_count != 0
        ), f"Cannot access value of unvisited node. Node: {self}"
        assert self._value is not None, f"Node value is not set. Node: {self}"
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        """Sets the value of the node."""
        self._value = value

    @property
    def nn_value(self) -> float:
        """Returns the neural network value of the node."""
        assert self.node_type == NodeType.OR, "Only OR nodes have NN values."
        assert self.is_expanded, "Cannot access NN value of unexpanded node."
        assert self._nn_value is not None, "NN value is not set."
        return self._nn_value

    @nn_value.setter
    def nn_value(self, value: float) -> None:
        """Sets the neural network value of the node."""
        assert self.node_type == NodeType.OR, "Only OR nodes have NN values."
        assert (
            self._value is None
        ), "NN values should only be set when no value exists yet"
        # setting an nn value means the nide was just expanded
        # Set the cashed value to the nn value
        self._value = value
        self._nn_value = value

    @property
    def has_value(self) -> bool:
        """Checks whether the node has a value."""
        return self._value is not None

    @property
    def visit_count(self) -> int:
        """Returns the visit count of the node."""
        # visit count is how often the children were visited + a first time the node was expanded
        # (except for AND nodes because the get expanded without actually being visited)
        visit_count = sum(self.children_visit_count.values()) + int(
            self.is_expanded and self.node_type != NodeType.AND
        )
        return visit_count

    def update(self, reward_config: RewardConfig, tamarin_timeout: int) -> bool:
        """Refreshes the value of the node based on its type."""
        if self.node_type == NodeType.OR:
            return self._update_or(reward_config, tamarin_timeout)
        else:
            return self._update_and()

    def _update_or(self, reward_config: RewardConfig, tamarin_timeout: int) -> bool:
        """Computes the value for an OR node"""

        assert (
            self.node_type == NodeType.OR
        ), "Node must be an OR node to aggregate value."

        assert self.nn_value is not None, "NN value must be set when refreshing value."
        weighted_sum = self.nn_value

        contradictory = False
        all_failed = True

        for action, child in self.children.items():
            if child.is_contradictory:
                contradictory = True
            if not child.failed:
                all_failed = False
            if child.is_solved:
                self.is_solved = True
            edge_visits = self.children_visit_count.get(action, 0)
            if edge_visits > 0:
                # The value of this specific edge is Reward + Child's Value
                path_value = (
                    reward_function(self, child, reward_config, tamarin_timeout)
                    + child.value
                )
                weighted_sum += edge_visits * path_value

        self.failed = all_failed and len(self.virtual_actions) == 0
        self.value = weighted_sum / (1 + self.visit_count)
        became_contradictory = contradictory and not self.is_contradictory
        self.is_contradictory = contradictory
        return became_contradictory

    def _update_and(self) -> bool:
        """Backpropagates an AND node.
        returns whether the node became contradictory
        """

        assert (
            self.node_type == NodeType.AND
        ), "Node must be an AND node to aggregate value."

        self.failed = all(child.failed for child in self.children.values())

        solved_children = [child for child in self.children.values() if child.is_solved]
        if len(solved_children) > 0:
            assert len(solved_children) == 1, "Only one solved child should exist."
            self.value = solved_children[0].value  # propagate solved value directly
            self.is_solved = True
            return False

        contradictory = all(child.is_contradictory for child in self.children.values())

        became_contradictory = contradictory and not self.is_contradictory
        self.is_contradictory = contradictory

        visited_children = [
            child
            for child in self.children.values()
            if child.visit_count > 0 and not child.is_contradictory
            # ignore contradictory nodes so the values reflects
            # the difficulty of the remaining proof
            # ignore unvisited nodes as the don't have a value
        ]
        if len(visited_children) == 0:
            self.value = MAX_VALUE  # optimistic value for uncertain nodes
            return became_contradictory

        self.value = min(child.value for child in visited_children)
        return became_contradictory

    # TODO remove assertions eventually
    @property
    def is_contradictory(self) -> bool:
        if self._is_contradictory:
            if self.node_type == NodeType.OR and len(self.children) > 0:
                assert any(child.is_contradictory for child in self.children.values())
            elif self.node_type == NodeType.AND:
                assert all(child.is_contradictory for child in self.children.values())
        return self._is_contradictory

    # TODO remove assertions eventually
    @is_contradictory.setter
    def is_contradictory(self, value: bool) -> None:

        if self.node_type == NodeType.OR and value and len(self.children) > 0:
            assert any(child.is_contradictory for child in self.children.values())
        elif self.node_type == NodeType.AND and value:
            assert all(child.is_contradictory for child in self.children.values())
        self._is_contradictory = value

    @property
    def state(self) -> State:
        assert self.node_type == NodeType.OR, "Only OR nodes have states."
        assert self._state is not None, f"Node state is None. Node: {self}"
        return self._state

    @property
    def prior_sum(self) -> float:
        """Returns the sum of the priors of the children."""
        return sum(self.children_prior.values())

    @property
    def prior_sum_only_heuristic(self) -> float:
        """Returns the sum of the priors of the children."""
        return sum(self.children_prior_only_heuristic.values())

    @property
    def prior_sum_only_model(self) -> float:
        """Returns the sum of the priors of the children."""
        return sum(self.children_prior_only_model.values())

    @property
    def is_expanded(self) -> bool:
        """Checks whether the node has been expanded."""
        return len(self.children) > 0

    def __repr__(self) -> str:
        return f"<Node(type={self.node_type}, value={self._value}, visit_count={self.visit_count}, is_terminal={self.is_terminal}, is_solved={self.is_solved}, is_contradictory={self.is_contradictory}, is_expanded={self.is_expanded}, failed={self.failed})>"


def reward_function(
    parent: Node, child: Node, reward_config: RewardConfig, tamarin_timeout: int
) -> float:
    if child.node_type == NodeType.AND:
        branch_penalty = 0.0
        time_penalty = 0.0
        timeout_penalty = 0.0
    elif child.is_terminal:
        branch_penalty = 0.0
        time_penalty = 0.0
        timeout_penalty = 0.0
    else:
        branch_penalty = (
            len(child.state.encoded_methods)
            / max(
                parent.max_branching_prefix,
                (
                    len(parent.state.encoded_methods)
                    if parent.node_type == NodeType.OR
                    else 0
                ),
            )
            * reward_config.branch_penalty
        )
        time_penalty = (
            min(child.expand_time, reward_config.time_penalty_clip)
            / reward_config.time_penalty_clip
            if reward_config.time_penalty_clip > 0
            else 0.0
        )
        timeout_penalty = (
            reward_config.timeout_penalty
            # add 10 seconds grace time to allow for some computation
            # #outside the tamarin call if we did not time out
            if child.expand_time >= (tamarin_timeout + 10)
            else 0.0
        )

    return -(
        reward_config.time_penalty * time_penalty
        + reward_config.branch_penalty * branch_penalty
        + timeout_penalty
        + 1
    )


@dataclass
class UCBScore:
    total_score: float
    prior_score: float
    value_score: float
    pb_c: float
    prior_score_heuristic: float
    prior_score_model: float
    prior_score_no_visit: float
    prior_score_no_visit_heuristic: float
    prior_score_no_visit_model: float


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(
    config: UCBConfig,
    reward_config: RewardConfig,
    tamarin_timeout: int,
    parent: Node,
    action: str,
    child: Node | None = None,
) -> UCBScore:

    devalue = False
    pb_c = (
        math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
        + config.pb_c_init
    )
    pb_c *= math.sqrt(parent.visit_count) / (
        parent.children_visit_count.get(action, 0) + 1
    )

    assert (
        action in parent.children_prior
    ), f"The action doesn't have a prior. Action: {action}, Parent: {parent}"
    prior_score = pb_c * parent.children_prior[action] / parent.prior_sum
    prior_score_heuristic = (
        pb_c
        * parent.children_prior_only_heuristic[action]
        / parent.prior_sum_only_heuristic
    )
    prior_score_model = (
        pb_c * parent.children_prior_only_model[action] / parent.prior_sum_only_model
    )
    prior_score_no_visit = parent.children_prior[action] / parent.prior_sum
    prior_score_no_visit_heuristic = (
        parent.children_prior_only_heuristic[action] / parent.prior_sum_only_heuristic
    )
    prior_score_no_visit_model = (
        parent.children_prior_only_model[action] / parent.prior_sum_only_model
    )
    if child is not None and parent.children_visit_count.get(action, 0) > 0:
        if child.visit_count == 0 and child.node_type == NodeType.AND:
            # this is the case were the AND nodes children don't haven been visited yet
            # since AND nodes don't have an nn_value we need to wait for the children to get a value.
            # we return 0 here to be optimistic about unvisited AND nodes.
            child_value = 0.0
        else:
            child_value = child.value
        value = (
            reward_function(parent, child, reward_config, tamarin_timeout) + child_value
        )
    elif parent.node_type == NodeType.AND and parent.visit_count == 0:
        # For AND nodes where no child hasn't been visited yet,
        # we use the value penalty without the parent value.
        value = -config.value_penalty
    else:
        value = parent.value - config.value_penalty

    value_score = config.value_discount ** (-1 - value)

    if parent.node_type == NodeType.AND:
        if config.invert_and:
            # Invert value score for AND nodes.
            value_score = 1 - value_score
        if child is not None and child.is_contradictory:
            # Avoid re-selecting proven subgoals.
            devalue = True

        prior_score *= config.c_and * len(parent.children)

    if child is not None and child.failed:
        # Avoid selecting failed nodes.
        devalue = True

    total_score = prior_score + value_score
    if config.deactivate_model:
        total_score = prior_score_heuristic

    if devalue:
        total_score = DEVALUE_SCORE
    return UCBScore(
        total_score=total_score,
        prior_score=prior_score,
        value_score=value_score,
        pb_c=pb_c,
        prior_score_heuristic=prior_score_heuristic,
        prior_score_model=prior_score_model,
        prior_score_no_visit=prior_score_no_visit,
        prior_score_no_visit_heuristic=prior_score_no_visit_heuristic,
        prior_score_no_visit_model=prior_score_no_visit_model,
    )
