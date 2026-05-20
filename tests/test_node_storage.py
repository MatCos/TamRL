from src.rl.node import NodeType
from src.rl.node_storage import NodeStorage

from tests.helpers import make_and_node as _and_node
from tests.helpers import make_or_node as _or_node
from tests.helpers import make_state as _state
from tests.helpers import make_terminal_or_node as _terminal_or_node


# ---------------------------------------------------------------------------
# NodeStorage init
# ---------------------------------------------------------------------------


class TestNodeStorageInit:
    def test_empty(self):
        ns = NodeStorage()
        assert ns.num_nodes == 0
        assert ns.num_original_nodes == 0
        assert ns.num_contradictory_nodes == 0
        assert ns.deduplicated_nodes == 0
        assert ns.deduplicated_ratio == 0.0


# ---------------------------------------------------------------------------
# NodeStorage.add_node
# ---------------------------------------------------------------------------


class TestNodeStorageAddNode:
    def test_add_or_node(self):
        ns = NodeStorage()
        parent = _or_node(n_methods=2)
        child_state = _state(1)
        node = ns.add_node(parent, "enc0", child_state, NodeType.OR)
        assert node in ns.or_node_table.values()
        assert ns.num_nodes == 1
        assert ns.num_original_nodes == 1
        assert parent.children["enc0"] is node

    def test_add_and_node(self):
        ns = NodeStorage()
        parent = _or_node()
        node = ns.add_node(parent, "enc0", None, NodeType.AND)
        assert node in ns.and_node_list
        assert ns.num_nodes == 1

    def test_add_terminal_node_to_dummy_list(self):
        ns = NodeStorage()
        parent = _or_node()
        node = ns.add_node(parent, "enc0", _state(1), NodeType.OR, is_terminal=True)
        assert node in ns.dummy_node_list
        assert ns.num_nodes == 1

    def test_add_failed_node_to_dummy_list(self):
        ns = NodeStorage()
        parent = _or_node()
        node = ns.add_node(parent, "enc0", None, NodeType.OR, failed=True)
        assert node in ns.dummy_node_list

    def test_deduplication_existing_or_node(self):
        ns = NodeStorage()
        parent = _or_node(n_methods=2)
        state = _state(1)
        first = ns.add_node(parent, "enc0", state, NodeType.OR)
        # Add same state again from a different parent
        parent2 = _or_node(n_methods=2)
        second = ns.add_node(parent2, "enc1", state, NodeType.OR)
        assert first is second
        assert ns.num_nodes == 1
        assert ns.num_original_nodes == 2
        assert ns.deduplicated_nodes == 1
        assert ns.deduplicated_ratio == 0.5
        # Both parents should be recorded
        assert parent in second.parents
        assert parent2 in second.parents

    def test_contradictory_count(self):
        ns = NodeStorage()
        parent = _or_node()
        ns.add_node(parent, "enc0", None, NodeType.OR,
                    is_contradictory=True, is_terminal=True)
        assert ns.num_contradictory_nodes == 1

    def test_case_name_stored(self):
        ns = NodeStorage()
        parent = _or_node()
        ns.add_node(parent, "enc0", _state(1), NodeType.OR, case="case_1")
        assert parent.case_names["enc0"] == "case_1"

    def test_max_branching_prefix_from_or_parent(self):
        ns = NodeStorage()
        parent = _or_node(n_methods=5)
        parent.max_branching_prefix = 3
        child_state = _state(2)
        node = ns.add_node(parent, "enc0", child_state, NodeType.OR)
        # max(parent.max_branching_prefix=3, len(parent.encoded_methods)=5) = 5
        assert node.max_branching_prefix == 5

    def test_max_branching_prefix_from_and_parent(self):
        ns = NodeStorage()
        and_parent = _and_node()
        and_parent.max_branching_prefix = 7
        child_state = _state(2)
        node = ns.add_node(and_parent, "case0", child_state, NodeType.OR)
        # AND parent preserves its own max_branching_prefix
        assert node.max_branching_prefix == 7

    def test_dedup_takes_min_branching_prefix(self):
        ns = NodeStorage()
        state = _state(1)
        parent1 = _or_node(n_methods=10)
        parent1.max_branching_prefix = 2
        ns.add_node(parent1, "enc0", state, NodeType.OR)
        node = ns.or_node_table[state]
        assert node.max_branching_prefix == 10  # max(2, 10)

        # Second path with smaller branching
        parent2 = _or_node(n_methods=3)
        parent2.max_branching_prefix = 1
        ns.add_node(parent2, "enc0", state, NodeType.OR)
        # min(10, max(1, 3)) = min(10, 3) = 3
        assert node.max_branching_prefix == 3


# ---------------------------------------------------------------------------
# NodeStorage.num_nodes
# ---------------------------------------------------------------------------


class TestNodeStorageNumNodes:
    def test_counts_all_types(self):
        ns = NodeStorage()
        parent = _or_node(n_methods=4)
        ns.add_node(parent, "enc0", _state(1), NodeType.OR)
        ns.add_node(parent, "enc1", None, NodeType.AND)
        ns.add_node(parent, "enc2", None, NodeType.OR, is_terminal=True)
        assert ns.num_nodes == 3
        assert len(ns.or_node_table) == 1
        assert len(ns.and_node_list) == 1
        assert len(ns.dummy_node_list) == 1
