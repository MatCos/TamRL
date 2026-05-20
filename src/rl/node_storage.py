from src.environment.environment import State
from src.rl.node import Node, NodeType


class NodeStorage:
    def __init__(self):
        self.or_node_table: dict[State, Node] = {}
        self.and_node_list: list[Node] = []
        self.dummy_node_list: list[Node] = []

        self.num_original_nodes = 0
        self.num_contradictory_nodes = 0

    @property
    def num_nodes(self) -> int:
        return (
            len(self.or_node_table)
            + len(self.and_node_list)
            + len(self.dummy_node_list)
        )

    @property
    def deduplicated_nodes(self) -> int:
        return self.num_original_nodes - self.num_nodes

    @property
    def deduplicated_ratio(self) -> float:
        if self.num_original_nodes == 0:
            return 0.0
        return self.deduplicated_nodes / self.num_original_nodes

    def add_node(
        self,
        parent: Node,
        action: str,
        state: State | None,
        node_type: NodeType,
        is_contradictory: bool = False,
        is_terminal: bool = False,
        is_solved: bool = False,
        failed: bool = False,
        case: str | None = None,
    ):
        self.num_original_nodes += 1
        self.num_contradictory_nodes += int(is_contradictory)
        if state is None or state not in self.or_node_table:
            node = Node(
                state,
                parents=[parent],
                max_branching_prefix=(
                    max(parent.max_branching_prefix, len(parent.state.encoded_methods))
                    if parent.node_type == NodeType.OR
                    else parent.max_branching_prefix
                ),
                node_type=node_type,
                is_contradictory=is_contradictory,
                is_terminal=is_terminal,
                is_solved=is_solved,
                failed=failed,
            )
            parent.children[action] = node
            parent.case_names[action] = case
            if node_type == NodeType.AND:
                self.and_node_list.append(node)
            elif is_terminal or failed:

                self.dummy_node_list.append(node)
            else:
                assert state is not None
                self.or_node_table[state] = node
            return node
        else:
            node = self.or_node_table[state]
            branching = (
                max(parent.max_branching_prefix, len(parent.state.encoded_methods))
                if parent.node_type == NodeType.OR
                else parent.max_branching_prefix
            )
            node.max_branching_prefix = min(node.max_branching_prefix, branching)
            parent.children[action] = node
            parent.case_names[action] = case
            node.parents.append(parent)
            return node
