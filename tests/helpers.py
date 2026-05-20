"""Shared test helpers for node / state construction."""

from src.environment.environment import State
from src.rl.node import Node, NodeType


def make_state(n_methods: int = 3) -> State:
    return State(
        proof_methods=[{"name": "Simplify"} for _ in range(n_methods)],
        encoded_methods=tuple(f"enc{i}" for i in range(n_methods)),
        encoded_sys="sys",
    )


def make_or_node(n_methods: int = 3, **kwargs) -> Node:
    return Node(state=make_state(n_methods), node_type=NodeType.OR,
                parents=[], max_branching_prefix=0, **kwargs)


def make_and_node(**kwargs) -> Node:
    return Node(state=None, node_type=NodeType.AND,
                parents=[], max_branching_prefix=0, **kwargs)


def make_terminal_or_node(is_contradictory: bool = True, is_solved: bool = False) -> Node:
    return Node(state=make_state(1), node_type=NodeType.OR, parents=[],
                max_branching_prefix=0, is_contradictory=is_contradictory,
                is_terminal=True, is_solved=is_solved)
