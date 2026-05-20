import dataclasses
import random
import uuid
from multiprocessing import Queue
from multiprocessing.connection import Connection
from time import time

import numpy as np
import torch
import torch.nn.functional as F

from src.environment.environment import State, TamarinEnvironment
from src.parser import RewardConfig, UCBConfig
from src.parser.lemma import LemmaConfig
from src.rl.node import DEVALUE_SCORE, Node, NodeType, UCBScore, reward_function, ucb_score
from src.rl.node_storage import NodeStorage
from src.rl.replay_buffer import ACTrainingExample
from src.utils.utils import (
    QUEUE_TIMEOUT,
    QueueWaitTimeoutError,
    TamarinCallTimeoutError,
)


@dataclasses.dataclass
class StepResult:
    depth: int
    done_root: Node | None
    done_root_parent: Node | None
    done_length: int
    training_examples: set[ACTrainingExample]
    env_step_log: dict[str, float | int]
    terminated: bool = False


@dataclasses.dataclass(frozen=True)
class SelectionResult:
    action: str
    child: Node
    selected_action_idx: int | None
    prior_score: float
    value_score: float
    pb_c: float
    ucb_margin: float
    ucb_entropy: float
    ucb_correlations: dict[str, float]
    alignments: dict[str, bool]


class StepLog:
    """Accumulates per-step metrics for logging."""

    def __init__(self) -> None:
        self._sum_prior_score = 0.0
        self._sum_value_score = 0.0
        self._sum_pb_c = 0.0
        self._sum_ucb_margin = 0.0
        self._sum_ucb_entropy = 0.0
        self._sum_ucb_correlations: dict[str, float] = {}
        self._sum_alignments: dict[str, int] = {}
        self._num_alignment_selections = 0
        self._selected_action_idx_sum = 0
        self._num_selections = 0
        self._num_or_nodes = 0
        self._entries: dict[str, float | int] = {}

    def accumulate_selection(self, sel: SelectionResult) -> None:
        self._sum_prior_score += sel.prior_score
        self._sum_value_score += sel.value_score
        self._sum_pb_c += sel.pb_c
        self._sum_ucb_margin += sel.ucb_margin
        self._sum_ucb_entropy += sel.ucb_entropy
        for k, v in sel.ucb_correlations.items():
            self._sum_ucb_correlations[k] = self._sum_ucb_correlations.get(k, 0.0) + v
        for k, v in sel.alignments.items():
            self._sum_alignments[k] = int(self._sum_alignments.get(k, 0)) + v
        if sel.alignments:
            self._num_alignment_selections += 1
        self._num_selections += 1
        if sel.selected_action_idx is not None:
            self._selected_action_idx_sum += sel.selected_action_idx
            self._num_or_nodes += 1

    def record(self, entries: dict[str, float | int]) -> None:
        self._entries.update(entries)

    def to_dict(self) -> dict[str, float | int]:
        n = self._num_selections
        result: dict[str, float | int] = {}
        result.update(
            {k: v / n if n > 0 else 0.0 for k, v in self._sum_ucb_correlations.items()}
        )
        n_align = self._num_alignment_selections
        result.update(
            {k: v / n_align if n_align > 0 else 0 for k, v in self._sum_alignments.items()}
        )
        result.update(
            {
                "EnvStep/avg_prior_score": (
                    self._sum_prior_score / n if n > 0 else 0.0
                ),
                "EnvStep/avg_value_score": (
                    self._sum_value_score / n if n > 0 else 0.0
                ),
                "EnvStep/avg_pb_c": self._sum_pb_c / n if n > 0 else 0.0,
                "EnvStep/avg_ucb_margin": (self._sum_ucb_margin / n if n > 0 else 0.0),
                "EnvStep/avg_ucb_entropy": (
                    self._sum_ucb_entropy / n if n > 0 else 0.0
                ),
                "EnvStep/selected_action_idx": (
                    self._selected_action_idx_sum / self._num_or_nodes
                    if self._num_or_nodes > 0
                    else 0
                ),
            }
        )
        result.update(self._entries)
        return result


class MCTS:

    def __init__(
        self,
        tamarin_env: TamarinEnvironment,
        request_conn: "Connection[tuple[str, State]]",
        response_conn: "Connection[tuple[str, torch.Tensor, float] | None]",
        training_example_queue: "Queue[list[ACTrainingExample]]",
        search_result_queue: "Queue[tuple[LemmaConfig, dict[str, object]]]",
        step_log_queue: "Queue[tuple[int, dict[str, float | int]]]",
        search_budget: int,
        search_budget_increase_factor: float,
        ucb_config: UCBConfig,
        reward_config: RewardConfig,
        env_id: int,
        lemma_config: LemmaConfig,
        tamarin_timeout: int,
        backup_tamarin_timeout: int,
        first_step: str = "search",
        expand_top_n: int = 0,
    ):
        self.env = tamarin_env
        self.first_step = first_step
        self.search_budget = search_budget
        self.expand_top_n = expand_top_n
        self.search_budget_increase_factor = search_budget_increase_factor
        self.env_id = env_id
        self.lemma_config = lemma_config

        # Request/receive logprobs from the agent
        self.request_conn = request_conn
        self.response_conn = response_conn
        # Send experience to the trainer
        self.training_example_queue = training_example_queue
        # Queue to send search results to the trainer
        self.search_result_queue = search_result_queue
        # Queue to send step logs to the trainer
        self.step_log_queue = step_log_queue
        # UCB config
        self.ucb_config = ucb_config
        # Reward config
        self.reward_config = reward_config
        self.tamarin_timeout = tamarin_timeout
        self.backup_tamarin_timeout = backup_tamarin_timeout

        self.max_expanded_depth = 0
        self.env_steps = 0
        self.num_expanded_nodes = 0
        self.contradictory_nodes_per_step: list[Node] = (
            []
        )  # TODO only for asserts, use counter eventually
        self.last_done_node = 0
        self.max_last_done_node = 0
        self.failures = 0
        self.virtual_materializations = 0

        self.node_storage = NodeStorage()

    @property
    def avg_expanded_branching(self) -> float:
        # every step expands expands a children hence adds one to the branching factor sum
        # every step adds multiple nodes
        if self.node_storage.num_original_nodes == 0:
            return 0.0
        return self.env_steps / self.node_storage.num_original_nodes

    def _materialize_virtual_action(
        self, node: Node, action: str
    ) -> Node | None:
        """Call Tamarin for a single virtual action and create the child."""
        method_index = node.virtual_actions.pop(action)

        cases_outcomes, failures, _ = self.env.perform_action_bulk(
            node.state, [method_index]
        )

        if method_index in failures:
            self.failures += 1
            self._set_child_priors(node, action, -1e9, -1e9, -1e9)
            self.node_storage.add_node(
                node, action, None, NodeType.OR,
                is_contradictory=False, is_terminal=False,
                failed=True,
            )
            return None

        case_outcomes = cases_outcomes.pop(0)

        if len(case_outcomes) == 0:
            p, ph, pm = 1.0, 1.0, 1.0
            self._set_child_priors(node, action, p, ph, pm)
        else:
            p = node.children_prior[action]
            ph = node.children_prior_only_heuristic[action]
            pm = node.children_prior_only_model[action]

        self._add_action_child(node, action, case_outcomes, p, ph, pm)

        return node.children[action]

    def select_child(self, node: Node) -> SelectionResult:
        assert node.is_expanded, "Cannot select child from unexpanded node."

        scored_children: list[tuple[UCBScore, str, Node | None]] = []
        for action, child in node.children.items():
            if child.failed:
                continue
            score = ucb_score(
                self.ucb_config,
                self.reward_config,
                self.tamarin_timeout,
                node,
                action,
                child,
            )
            scored_children.append((score, action, child))

        # Score virtual actions (unmaterialized children)
        for action in node.virtual_actions:
            score = ucb_score(
                self.ucb_config,
                self.reward_config,
                self.tamarin_timeout,
                node,
                action,
            )
            scored_children.append((score, action, None))

        assert (
            len(scored_children) > 0
        ), f"No valid children to select from. Node: {node}, children: {node.children}"

        # Compute alignment on the unsorted list so argmax reflects true rankings.
        # Skip AND nodes (uniform priors by construction) and devalued nodes
        # (their total_score is overridden, so alignment is meaningless).
        non_devalued = [
            (score, action, child)
            for score, action, child in scored_children
            if score.total_score != DEVALUE_SCORE
        ]
        if node.node_type == NodeType.OR and len(non_devalued) > 1:
            best_indices = {
                k: np.argmax([getattr(score, k) for score, _, _ in non_devalued])
                for k in UCBScore.__annotations__.keys()
            }
            agrees = {
                f"EnvStep/alignment_{k}": best_indices[k] == best_indices["total_score"]
                for k in best_indices.keys()
                if k != "total_score"
            }
        else:
            agrees = {}

        scored_children.sort(key=lambda x: x[0].total_score, reverse=True)

        best_score, best_action, best_child = scored_children[0]

        # Materialize virtual action if selected
        if best_child is None:
            self.virtual_materializations += 1
            best_child = self._materialize_virtual_action(
                node, best_action
            )
            if best_child is None or best_child.failed:
                # Materialization failed — retry with remaining candidates
                for score, action, child in scored_children[1:]:
                    if child is None:
                        self.virtual_materializations += 1
                        child = self._materialize_virtual_action(
                            node, action
                        )
                        if child is None or child.failed:
                            continue
                    if child.failed:
                        continue
                    best_score, best_action, best_child = score, action, child
                    break

        if len(scored_children) > 1:
            ucb_margin = best_score.total_score - scored_children[1][0].total_score
        else:
            ucb_margin = 0.0

        # Calculate entropy of the UCB scores
        scores_tensor = torch.tensor([x[0].total_score for x in scored_children])
        probs = F.softmax(scores_tensor, dim=0)
        ucb_entropy = -(probs * torch.log(probs + 1e-9)).sum().item()

        ucb_correlations = self.ucb_correlation([x[0] for x in scored_children])

        if node.node_type == NodeType.OR:
            selected_action_idx = node.state.encoded_methods.index(best_action)
        else:
            selected_action_idx = None
        node.children_visit_count[best_action] = (
            node.children_visit_count.get(best_action, 0) + 1
        )
        assert best_child is not None, (
            f"All candidates failed. Node: {node}"
        )
        assert not best_child.failed or (
            all(child.failed for child in node.children.values())
            and len(node.virtual_actions) == 0
            and len(best_child.parents) >= 1
        ), f"Failed nodes should not be selected. Children list: {scored_children}"
        return SelectionResult(
            action=best_action,
            child=best_child,
            selected_action_idx=selected_action_idx,
            prior_score=best_score.prior_score,
            value_score=best_score.value_score,
            pb_c=best_score.pb_c,
            ucb_margin=ucb_margin,
            ucb_entropy=ucb_entropy,
            ucb_correlations=ucb_correlations,
            alignments=agrees,
        )

    @staticmethod
    def ucb_correlation(scores: list[UCBScore]) -> dict[str, float]:
        if len(scores) < 2:
            return {}

        keys = {
            "prior_score": "EnvStep/prior_correlation",
            "value_score": "EnvStep/value_correlation",
            "pb_c": "EnvStep/pb_c_correlation",
            "prior_score_heuristic": "EnvStep/prior_heuristic_correlation",
            "prior_score_model": "EnvStep/prior_model_correlation",
            "prior_score_no_visit": "EnvStep/prior_no_visit_correlation",
            "prior_score_no_visit_heuristic": "EnvStep/prior_no_visit_heuristic_correlation",
            "prior_score_no_visit_model": "EnvStep/prior_no_visit_model_correlation",
        }

        total_scores = torch.tensor([s.total_score for s in scores])
        mean_total = torch.mean(total_scores)
        std_total = torch.std(total_scores)

        correlations = {}

        if std_total == 0:
            return {corr_key: 0.0 for corr_key in keys.values()}

        for key, corr_key in keys.items():
            values = torch.tensor([getattr(s, key) for s in scores])
            std_val = torch.std(values)
            if std_val == 0:
                correlations[corr_key] = 0.0
            else:
                mean_val = torch.mean(values)
                c_total = total_scores - mean_total
                c_val = values - mean_val
                cov = torch.sum(c_total * c_val)
                correlations[corr_key] = (
                    cov / ((len(scores) - 1) * std_total * std_val)
                ).item()

        return correlations

    def traverse(
        self, root_node: Node, step_log: StepLog
    ) -> tuple[Node, list[tuple[int | str | None, Node]]]:
        """select expanded nodes until a leaf (not expanded) is reached"""
        node = root_node
        search_path: list[tuple[int | str | None, Node]] = [(None, node)]
        depth = 0
        traversal_start_time = time()
        while (
            node.is_expanded
            and not node.is_contradictory
            and not node.failed
            and not node.is_solved
        ):
            sel = self.select_child(node)

            assert not node.is_contradictory
            if (
                any(node.children[a].is_contradictory for a in node.children)
                and node.node_type == NodeType.OR
            ):
                # if a child is contradictory, the child has been set  as contradictory by another path through the graph. In a tree this could not happen, but in a graph we have stale nodes.
                node.children_visit_count[sel.action] -= 1
                break
            assert (
                all(not node.children[a].is_contradictory for a in node.children)
                or node.node_type == NodeType.AND
            )

            step_log.accumulate_selection(sel)
            search_path.append(
                (
                    (
                        sel.selected_action_idx
                        if node.node_type == NodeType.OR
                        else sel.action
                    ),
                    sel.child,
                )
            )
            depth += 1
            node = sel.child
        traversal_time = time() - traversal_start_time
        self.max_expanded_depth = max(self.max_expanded_depth, depth)
        step_log.record(
            {
                "EnvStep/traversal_time": traversal_time,
                "EnvStep/expand_depth": depth,
            }
        )

        return node, search_path

    def inference(
        self, node: Node, step_log: StepLog
    ) -> None | tuple[torch.Tensor, float]:
        assert node.node_type == NodeType.OR, "Only OR nodes can be inferenced."
        assert (
            node._state is not None
        ), f"Node state should not be None before inference. Node: {node}"
        model_inference_start_time = time()
        request_id = str(uuid.uuid4())
        self.request_conn.send((request_id, node.state))
        if not self.response_conn.poll(QUEUE_TIMEOUT):
            raise QueueWaitTimeoutError("Timeout waiting for inference response")
        response: tuple[str, torch.Tensor, float] | None = self.response_conn.recv()
        model_inference_time = time() - model_inference_start_time

        if response is None:
            # terminate signal received
            self.search_result_queue.put(
                (
                    self.lemma_config,
                    {
                        "solved": False,
                        "contradictory": False,
                        "steps": self.env_steps,
                        "dry_spell": self.max_last_done_node,
                        "stopped_early": True,
                        "new_budget": self.search_budget,
                    },
                )
            )
            return None

        resp_id, prior_logprobs, nn_value = response
        assert (
            resp_id == request_id
        ), f"Response ID mismatch: response {resp_id}, expected {request_id}"
        assert prior_logprobs.shape[0] == len(
            node.state.proof_methods
        ), "Number of proof methods doesn't match priors"

        step_log.record(
            {
                "EnvStep/model_inference_time": model_inference_time,
                "EnvStep/expand_value": nn_value,
            }
        )

        return prior_logprobs, nn_value

    def log_prior_metrics(
        self,
        pre_heuristic_logprobs: torch.Tensor,
        post_heuristic_logprobs: torch.Tensor,
        step_log: StepLog,
    ) -> None:
        """Log pre- and post-heuristic uncertainty metrics."""
        probs = pre_heuristic_logprobs.exp()
        entropy = -(probs * pre_heuristic_logprobs).sum().item()
        confidence = probs.max().item()

        probs_post = post_heuristic_logprobs.exp()
        entropy_post = -(probs_post * post_heuristic_logprobs).sum().item()
        confidence_post = probs_post.max().item()

        step_log.record(
            {
                "EnvStep/prior_entropy": entropy,
                "EnvStep/prior_confidence": confidence,
                "EnvStep/prior_entropy_heuristic": entropy_post,
                "EnvStep/prior_confidence_heuristic": confidence_post,
            }
        )

    def add_heuristic_bias(
        self, prior_logprobs: torch.Tensor, step_log: StepLog
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """bayesian prior bias based on rank of the pfms"""
        raw_logprobs = prior_logprobs
        ranks = torch.arange(prior_logprobs.shape[0], device=prior_logprobs.device)
        rank_bias = -self.ucb_config.heuristic_weight * ranks.float()
        prior_logprobs = F.log_softmax(raw_logprobs + rank_bias, dim=0)
        prior_logprobs_model_only = F.log_softmax(raw_logprobs, dim=0)
        prior_logprobs_heuristic_only = F.log_softmax(rank_bias, dim=0)

        self.log_prior_metrics(raw_logprobs, prior_logprobs, step_log)
        return prior_logprobs, prior_logprobs_heuristic_only, prior_logprobs_model_only

    def search_state_snapshot(self, step: int) -> dict[str, float | int]:
        return {
            "SearchState/num_expanded_nodes": self.num_expanded_nodes,
            "SearchState/num_original_nodes": self.node_storage.num_original_nodes,
            "SearchState/num_nodes": self.node_storage.num_nodes,
            "SearchState/num_deduplicated_nodes": self.node_storage.deduplicated_nodes,
            "SearchState/deduplicated_ratio": self.node_storage.deduplicated_ratio,
            "SearchState/num_contradictory_nodes": self.node_storage.num_contradictory_nodes,
            "SearchState/avg_expanded_branching": self.avg_expanded_branching,
            "SearchState/max_expanded_depth": self.max_expanded_depth,
            "SearchState/failures": self.failures,
            "SearchState/virtual_materializations": self.virtual_materializations,
            "SearchState/step": step,
        }

    def report_budget_exhausted(self):
        self.search_result_queue.put(
            (
                self.lemma_config,
                {
                    "solved": False,
                    "contradictory": False,
                    "steps": self.env_steps,
                    "dry_spell": self.max_last_done_node,
                    "new_budget": int(
                        self.search_budget * self.search_budget_increase_factor
                    ),
                },
            )
        )

    def proof_success(self, game_root: Node):
        # proof done, end this search
        tree, number_nodes = self.extract_proof_tree(game_root)
        file_str = self.env.check_proof(
            proof=tree,
            size=number_nodes,
            status=("CompleteProof" if game_root.is_contradictory else "TraceFound"),
            timeout=self.backup_tamarin_timeout,
        )
        self.search_result_queue.put(
            (
                self.lemma_config,
                {
                    "solved": game_root.is_solved,
                    "contradictory": game_root.is_contradictory,
                    "steps": self.env_steps,
                    "tree": tree,
                    "tree_size": number_nodes,
                    "dry_spell": self.max_last_done_node,
                    "proof_file": file_str,
                    "new_budget": self.search_budget,
                },
            )
        )

    def backpropagate_and_collect(
        self, node: Node, search_path: list, step_log: StepLog
    ) -> Node | None:
        if not node.is_expanded:
            # could happen if we come to this node through a graph link
            # or if the node is failed.
            search_path = search_path[:-1]

        backprop_start_time = time()
        # backpropagate value and flags (solved, contradictory)
        done_root, done_root_parent, done_length = self.backpropagate(
            [node for _, node in search_path]
        )

        # collect experiences

        extract_start = done_root_parent if done_root_parent is not None else done_root
        if extract_start is not None:
            training_examples, _ = self.extract_training_examples(extract_start)
        else:
            training_examples = set()

        if len(training_examples) > 0:
            self.training_example_queue.put(
                sorted(
                    list(training_examples),
                    key=lambda x: (
                        x.state.encoded_sys,
                        x.state.encoded_methods,
                        x.action,
                    ),
                )
            )

        backprop_time = time() - backprop_start_time

        if done_root is not None:
            step_log.record(
                {
                    "EnvStep/done_length": done_length,
                    "EnvStep/contradictory_share": done_length / len(search_path),
                }
            )

        step_log.record(
            {
                "EnvStep/backprop_time": backprop_time,
                "EnvStep/num_collected_samples": len(training_examples),
            }
        )
        return done_root

    def _apply_forced_first_step(self, game_root: Node) -> None:
        """Pre-expand game_root with only the forced first-step action.

        After this, game_root is expanded with a single child (the forced
        method). The tree structure is identical to what expand_node would
        produce, but without exploring any other methods.
        """
        state = game_root.state
        method_index = self._resolve_first_step(state)
        action = state.encoded_methods[method_index]

        case_outcomes = self.env.perform_action(state, method_index)

        assert case_outcomes is not None, (
            f"Forced first step '{self.first_step}' "
            f"(method {method_index}) failed to apply."
        )

        # Set nn_value so _propagate_child_status can call update().
        # Value is irrelevant since the forced step isn't a learned decision.
        game_root.nn_value = 0.0

        self._add_action_child(game_root, action, case_outcomes, 1.0, 1.0, 1.0)
        self._propagate_child_status(game_root)

    def _resolve_first_step(self, state: State) -> int:
        """Returns the method index for the forced first step."""
        if self.first_step == "heuristic":
            return 0

        if len(state.proof_methods) == 1:
            return 0

        if self.first_step == "simplify":
            target_name = "Simplify"
        elif self.first_step == "induction":
            target_name = "Induction"
        else:
            raise ValueError(f"Unknown first_step: '{self.first_step}'")
        for i, pm in enumerate(state.proof_methods):
            if pm.get("name") == target_name:
                return i

        raise ValueError(
            f"First step '{self.first_step}' requested but "
            f"'{target_name}' not found in proof methods: "
            f"{[pm.get('name', pm.get('goal', {}).get('type')) for pm in state.proof_methods]}"
        )

    def __call__(self) -> None:
        init_state = self.env.get_initial_state(timeout=self.backup_tamarin_timeout)

        game_root = Node(init_state, NodeType.OR, [], 0)

        # Send root state to trainer for inference diagnostics.
        assert self.env.client is not None
        self.env.init_queue.put(
            (
                self.lemma_config,
                init_state,
                self.env.client.process.pid,
                self.env_id,
            )
        )

        if self.first_step != "search":
            self._apply_forced_first_step(game_root)
            if game_root.is_solved or game_root.is_contradictory:
                self.proof_success(game_root)
                return
        # game_root is not part of NodeTable. Cant be duplicate anyway.
        for i in range(self.search_budget):

            step_log = StepLog()
            start_time = time()
            self.contradictory_nodes_per_step = []
            self.last_done_node += 1
            leaf_node, search_path = self.traverse(game_root, step_log)

            self.env_steps += 1

            # If the node is not expanded yet, we do it now.
            # Otherwise we skip to backpropagation immediately, to just update stale nodes.
            # this case distinction is necessary because of stale nodes in the graph.
            if (
                not leaf_node.is_expanded
                and not leaf_node.is_terminal
                and not leaf_node.failed
            ):

                inference = self.inference(leaf_node, step_log)
                if inference is None:
                    # terminate signal received during inference
                    return
                prior_logprobs, leaf_node.nn_value = inference

                (
                    prior_logprobs,
                    prior_logprobs_heuristic_only,
                    prior_logprobs_model_only,
                ) = self.add_heuristic_bias(prior_logprobs, step_log)

                # expand node by adding children and priors
                try:
                    self.expand_node(
                        leaf_node,
                        prior_logprobs,
                        prior_logprobs_heuristic_only,
                        prior_logprobs_model_only,
                        step_log,
                    )
                except TamarinCallTimeoutError as e:
                    print(search_path)
                    raise e

                assert not leaf_node.is_terminal

            done_root = self.backpropagate_and_collect(leaf_node, search_path, step_log)
            total_step_time = time() - start_time
            step_log.record(
                {
                    "EnvStep/step_time": total_step_time,
                    "EnvStep/num_new_contradictory_nodes": len(
                        self.contradictory_nodes_per_step
                    ),
                    "EnvStep/last_done_node": self.last_done_node,
                    "EnvStep/tamarin_cache_hit_rate": self.env.cache_hit_rate,
                    "EnvStep/tamarin_cache_usage": self.env.cache_usage,
                }
            )
            step_log.record(self.search_state_snapshot(i))

            self.step_log_queue.put((self.env_id, step_log.to_dict()))
            self.max_last_done_node = max(self.max_last_done_node, self.last_done_node)

            if game_root.is_solved or game_root.is_contradictory:
                assert (
                    done_root is game_root
                ), "Root should be done if proof is complete."
                self.proof_success(game_root)
                return

        self.report_budget_exhausted()

    # TODO remove eventually
    def assert_valid_extraction(self, training_examples) -> None:
        _contradictory_nodes_per_step = [
            node
            for node in self.contradictory_nodes_per_step
            if node.node_type == NodeType.OR and len(node.children) > 0
        ]
        assert len(_contradictory_nodes_per_step) <= len(
            training_examples
        ), f"Number of contradictory nodes {len(_contradictory_nodes_per_step)} exceeds number of training examples {len(training_examples)}."

    @staticmethod
    def _set_child_priors(
        node: Node, action: str, combined: float, heuristic: float, model: float
    ) -> None:
        node.children_prior[action] = combined
        node.children_prior_only_heuristic[action] = heuristic
        node.children_prior_only_model[action] = model

    def _add_action_child(
        self,
        node: Node,
        action: str,
        case_outcomes: dict[str, "TamarinEnvironment.ResponseState"],
        combined_prior: float,
        heuristic_prior: float,
        model_prior: float,
    ) -> None:
        """Add a child to node for a single action's case outcomes.

        Handles 0 cases (contradictory), 1 case (OR child),
        and N cases (AND node + immediate expansion).
        """
        self._set_child_priors(
            node, action, combined_prior, heuristic_prior, model_prior
        )

        if len(case_outcomes) == 0:
            self.node_storage.add_node(
                node,
                action,
                None,
                NodeType.OR,
                is_contradictory=True,
                is_terminal=True,
            )
        elif len(case_outcomes) == 1:
            case, child = next(iter(case_outcomes.items()))
            self.node_storage.add_node(
                parent=node,
                action=action,
                case=case,
                state=State.from_response(child),
                node_type=NodeType.OR,
                is_contradictory=child.is_contradictory,
                is_terminal=child.is_terminal,
                is_solved=child.is_solved,
            )
        else:
            self.node_storage.add_node(
                parent=node, action=action, state=None, node_type=NodeType.AND
            )
            self.expand_and_node(node.children[action], case_outcomes)

    def expand_node(
        self,
        node: Node,
        prior_logprobs: torch.Tensor,
        prior_logprobs_heuristic_only: torch.Tensor,
        prior_logprobs_model_only: torch.Tensor,
        step_log: StepLog,
    ):
        """Expands a node by exploring its children."""

        expand_start_time = time()
        assert node.node_type == NodeType.OR, "Only OR nodes can be expanded."

        prior_probs = torch.exp(prior_logprobs / self.ucb_config.temperature).tolist()
        prior_probs_model_only = torch.exp(
            prior_logprobs_model_only / self.ucb_config.temperature
        ).tolist()
        prior_probs_heuristic_only = torch.exp(
            prior_logprobs_heuristic_only / self.ucb_config.temperature
        ).tolist()
        num_proof_methods = len(node.state.proof_methods)
        num_cases = 0
        self.num_expanded_nodes += 1
        done_actions = 0
        tamarin_call_time = 0.0

        if self.expand_top_n > 0 and self.expand_top_n < num_proof_methods:
            # Partial expansion: set priors for ALL methods, expand only top-N
            for i, action in enumerate(node.state.encoded_methods):
                self._set_child_priors(
                    node, action,
                    prior_probs[i],
                    prior_probs_heuristic_only[i],
                    prior_probs_model_only[i],
                )
            sorted_indices = sorted(
                range(num_proof_methods),
                key=lambda i: prior_probs[i],
                reverse=True,
            )
            eager_indices = sorted_indices[: self.expand_top_n]
            for i in sorted_indices[self.expand_top_n :]:
                node.virtual_actions[node.state.encoded_methods[i]] = i
        else:
            eager_indices = list(range(num_proof_methods))

        cases_outcomes, failures, tamarin_call_time = (
            self.env.perform_action_bulk(node.state, eager_indices)
        )

        for i in eager_indices:
            action = node.state.encoded_methods[i]
            if i in failures:
                self.failures += 1
                self._set_child_priors(node, action, -1e9, -1e9, -1e9)
                self.node_storage.add_node(
                    node, action, None, NodeType.OR,
                    is_contradictory=False, is_terminal=False,
                    failed=True,
                )
            else:
                case_outcomes = cases_outcomes.pop(0)
                num_cases += len(case_outcomes)
                done_actions += 1

                if len(case_outcomes) == 0:
                    p, ph, pm = 1.0, 1.0, 1.0
                else:
                    p = prior_probs[i]
                    ph = prior_probs_heuristic_only[i]
                    pm = prior_probs_model_only[i]

                self._add_action_child(
                    node, action, case_outcomes, p, ph, pm
                )

                child_node = node.children[action]
                if (
                    child_node.node_type == NodeType.OR
                    and child_node.is_contradictory
                ):
                    self.contradictory_nodes_per_step.append(child_node)
                    self.last_done_node = 0
                    break
                if node.is_contradictory:
                    break

        self._propagate_child_status(node)

        expand_time = time() - expand_start_time
        node.expand_time = expand_time
        step_log.record(
            {
                "EnvStep/num_proof_methods": num_proof_methods,
                "EnvStep/num_cases": (
                    num_cases / done_actions if done_actions > 0 else 0.0
                ),
                "EnvStep/tamarin_call_time": tamarin_call_time,
                "EnvStep/expand_time": expand_time,
            }
        )

    def _propagate_child_status(self, node: Node) -> None:
        """Propagate contradictory/solved/failed status from children to parent."""
        contradictory_children: list[Node] = []
        has_done_child = False
        all_failed = True
        for child_node in node.children.values():
            if child_node.is_contradictory:
                contradictory_children.append(child_node)
                has_done_child = True
            if child_node.is_solved:
                has_done_child = True
            if not child_node.failed:
                all_failed = False

        if has_done_child:
            self.contradictory_nodes_per_step.extend(contradictory_children)
            self.last_done_node = 0
            node.update(
                reward_config=self.reward_config, tamarin_timeout=self.tamarin_timeout
            )
        if all_failed and len(node.virtual_actions) == 0:
            node.failed = True

    def expand_and_node(
        self, node: Node, case_outcomes: dict[str, TamarinEnvironment.ResponseState]
    ) -> None:
        """Expands an AND node by exploring its children."""
        assert node.node_type == NodeType.AND, "Node must be an AND node to expand."

        self.num_expanded_nodes += 1

        for case, child in case_outcomes.items():
            node.children_prior[case] = 1.0 / len(case_outcomes)
            node.children_prior_only_heuristic[case] = 1.0 / len(case_outcomes)
            node.children_prior_only_model[case] = 1.0 / len(case_outcomes)
            self.node_storage.add_node(
                parent=node,
                action=case,
                state=State.from_response(child),
                node_type=NodeType.OR,
                is_contradictory=child.is_contradictory,
                is_terminal=child.is_terminal,
                is_solved=child.is_solved,
                case=case,
            )
            if child.is_contradictory:
                self.last_done_node = 0
                self.contradictory_nodes_per_step.append(node.children[case])
            if child.is_solved:
                # propagate solved status and values immediately in case of solved child
                node.update(
                    reward_config=self.reward_config,
                    tamarin_timeout=self.tamarin_timeout,
                )
                break

        if all(child.is_contradictory for child in node.children.values()):
            node.is_contradictory = True
            self.node_storage.num_contradictory_nodes += 1
            self.last_done_node = 0
            self.contradictory_nodes_per_step.append(node)
            node.update(
                reward_config=self.reward_config, tamarin_timeout=self.tamarin_timeout
            )
        if all(child.failed for child in node.children.values()):
            node.failed = True

    def backpropagate(
        self,
        search_path: list[Node],
    ) -> tuple[Node | None, Node | None, int]:
        """Backpropagate values and flags (contradictor / solved) back through the tree.

        Args:
            search_path (list[Node]): The path the was taken in this step, with the last node expanded right before.
            value (float): The state value to backpropagate.

        Returns:
            tuple[ Node | None, int, list[Node] ]:
                done root node (the root of the contradictory / solved subtree, None if nothing solved or contradictory)
                number of nodes that are done on the path from done_root to the leaf
        """

        assert search_path[
            -1
        ].is_expanded, (
            f"Cannot backpropagate from unexpanded node. Node {search_path[-1]}"
        )

        done_root = None
        done_root_parent = None
        done_length = 0

        if search_path[-1].is_contradictory or search_path[-1].is_solved:
            done_root = search_path[-1]
            done_length += 1

        if (
            search_path[-1].node_type == NodeType.OR
            and all(c.has_value for c in search_path[-1].children.values())
            or search_path[-1].node_type == NodeType.AND
            and any(c.has_value for c in search_path[-1].children.values())
        ):
            search_path[-1].update(
                reward_config=self.reward_config, tamarin_timeout=self.tamarin_timeout
            )

        for idx, node in reversed(list(enumerate(search_path[:-1]))):
            # aggregate contradictory / solved status from children

            became_contradictory = node.update(
                reward_config=self.reward_config, tamarin_timeout=self.tamarin_timeout
            )

            if became_contradictory:
                self.node_storage.num_contradictory_nodes += 1
                self.last_done_node = 0
                self.contradictory_nodes_per_step.append(node)
            if node.is_contradictory or node.is_solved:
                done_root = node
                done_root_parent = search_path[idx - 1] if idx > 0 else None
                done_length += 1

        return done_root, done_root_parent, done_length

    def select_optimal_action(self, node: Node) -> tuple[str, Node]:
        """Selects the optimal action from the node."""
        assert node.node_type == NodeType.OR
        # prefer solved over contradictory
        if any(child.is_solved for child in node.children.values()):
            selection = [
                (action, child)
                for action, child in node.children.items()
                if child.is_solved
            ]
        else:
            selection = [
                (action, child)
                for action, child in node.children.items()
                if child.is_contradictory
            ]
        assert (
            len(selection) > 0
        ), f"No optimal action found: {list(node.children.items())}"
        [(action, node)] = random.sample(selection, 1)
        return action, node

    def extract_roots(self, done_root: Node | None, game_root: Node) -> list[Node]:
        # extract training examples from contradictory nodes
        if done_root == game_root:
            roots = [game_root]
        elif done_root is None:
            roots = []
        else:
            parents = [
                parent
                for parent in done_root.parents
                if parent.node_type == NodeType.OR
            ]
            if len(parents) == 0:
                roots = [done_root]
            else:
                roots = parents

        return roots

    def extract_training_examples(
        self, node: Node
    ) -> tuple[set[ACTrainingExample], float]:
        """Extracts transitions and computes pure value targets from the proof tree.

        Value targets are computed bottom-up from the proof structure:
        - Terminal nodes: 1.0 (MAX_VALUE, proof done)
        - OR nodes: reward_function(parent, child) + child_value_target
        - AND nodes: min(child_value_targets) (hardest subgoal)

        This avoids nn_value contamination in training targets.

        Returns:
            tuple of (training examples, value_target).
        """

        # base case: terminal node
        if node.is_terminal:
            return set(), 1.0

        assert node.is_expanded, f"Cannot extract from unexpanded node. {node}"

        transitions: set[ACTrainingExample] = set()

        # if node is OR, select optimal action
        # (meaning those that are solved or contradictory)
        # add this and recurse on this child
        if node.node_type == NodeType.OR:
            action, child = self.select_optimal_action(node)
            action_idx = node.state.encoded_methods.index(action)
            child_transitions, child_value_target = self.extract_training_examples(child)
            value_target = (
                reward_function(node, child, self.reward_config, self.tamarin_timeout)
                + child_value_target
            )
            transitions = {ACTrainingExample(node.state, action_idx, value_target)}
            if not transitions.isdisjoint(child_transitions):
                assert False, "Cycle found in training example extraction."
            transitions.update(child_transitions)
            return transitions, value_target

        # if node is AND, recurse on selected children
        if node.node_type == NodeType.AND:
            child_value_targets: list[float] = []
            for action, child in node.children.items():
                # if the node is solved only recurse on the children that are solved
                # otherwise recurse on all
                if child.is_solved or node.is_contradictory:
                    ret, child_vt = self.extract_training_examples(child)
                    transitions.update(ret)
                    child_value_targets.append(child_vt)
            value_target = min(child_value_targets) if child_value_targets else 1.0
            return transitions, value_target

        raise ValueError("Node has incompatible type/state for extraction.")

    def extract_proof_tree(self, node: Node) -> tuple[dict, int]:
        """Extracts the proof tree from the MCTS search tree."""
        assert node.node_type == NodeType.OR
        if node.is_terminal:
            return {
                "action": "contradictory" if node.is_contradictory else "solved",
                "children": {},
            }, 1
        action, child = self.select_optimal_action(node)
        if child.node_type == NodeType.AND and child.is_contradictory:
            children = {}
            number_nodes = 0
            for case, grandchild in child.children.items():
                children[case], _number_nodes = self.extract_proof_tree(grandchild)
                number_nodes += _number_nodes
        elif child.node_type == NodeType.AND and child.is_solved:
            case, grandchild = [
                (case, grandchild)
                for case, grandchild in child.children.items()
                if grandchild.is_solved
            ][0]
            rec_return, number_nodes = self.extract_proof_tree(grandchild)
            children = {case: rec_return}
        else:
            if child.is_terminal and node.case_names[action] is None:
                children = {}
                number_nodes = 0
            else:
                case_name = node.case_names[action]
                assert (
                    case_name is not None
                ), f"Case name for action {action} should not be None. Node: {node}"
                rec_return, number_nodes = self.extract_proof_tree(child)
                children = {case_name: rec_return}

        return {
            "action": action,
            "children": children,
        }, 1 + number_nodes
