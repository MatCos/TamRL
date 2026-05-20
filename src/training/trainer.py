from __future__ import annotations

import json
import os
import random
import shutil
import time
import traceback
from multiprocessing.connection import Connection, wait
from queue import Empty
from typing import Any, cast

import numpy as np
import psutil
import torch
from torch.multiprocessing import Manager, Pipe, Process, Queue
from tqdm import tqdm

from src.environment import State, TamarinCache, TamarinEnvironment
from src.parser import LemmaConfig, RewardConfig, SearchConfig, UCBConfig
from src.parser.parser_rl import RLConfig
from src.rl.ac_agent import ACAgent
from src.rl.mcts import MCTS
from src.utils.recorder import Recorder
from src.rl.replay_buffer import ACTrainingExample, ReplayBuffer
from src.utils.utils import (
    QUEUE_TIMEOUT,
    QueueWaitTimeoutError,
    create_device,
    dataclass_to_dict,
)


def run_worker(
    lemma_config: LemmaConfig,
    request_conn: Connection,
    response_conn: Connection,
    cache_request_conn: Connection,
    cache_response_conn: Connection,
    fill_cache_conn: "Queue[tuple[LemmaConfig, State, list[int], list[dict[str, TamarinEnvironment.ResponseState]], list[int], float]]",
    training_example_queue: "Queue[list[ACTrainingExample]]",
    init_queue: "Queue[tuple[LemmaConfig, State, int, int]]",
    search_result_queue: "Queue[tuple[LemmaConfig, dict[str, Any]]]",
    step_log_queue: "Queue[tuple[int, dict[str, float | int]]]",
    search_config: SearchConfig,
    search_budget: int,
    ucb_config: UCBConfig,
    reward_config: RewardConfig,
    env_id: int,
    seed: int,
    timeout: int,
    backup_timeout: int,
    deterministic_mode: bool = False,
) -> None:
    print(
        f"Worker {env_id}: Starting worker process for lemma {lemma_config.lemma_name} with budget {search_budget}..."
    )
    try:
        # Seed randomness
        random.seed(seed + env_id)
        np.random.seed(seed + env_id)
        torch.manual_seed(seed + env_id)
        if torch.cuda.is_available() and deterministic_mode:
            torch.cuda.manual_seed_all(seed + env_id)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=False)
        tamarin_env = TamarinEnvironment(
            cache_request_conn,
            cache_response_conn,
            fill_cache_conn,
            init_queue,
            timeout,
            backup_timeout,
            id=env_id,
        )
        tamarin_env.set(
            lemma_config=lemma_config,
        )
        mcts = MCTS(
            tamarin_env=tamarin_env,
            request_conn=request_conn,
            response_conn=response_conn,
            training_example_queue=training_example_queue,
            search_result_queue=search_result_queue,
            step_log_queue=step_log_queue,
            search_budget=search_budget,
            search_budget_increase_factor=search_config.budget_increase_factor,
            ucb_config=ucb_config,
            reward_config=reward_config,
            env_id=env_id,
            lemma_config=lemma_config,
            tamarin_timeout=timeout,
            backup_tamarin_timeout=backup_timeout,
            first_step=search_config.first_step,
            expand_top_n=search_config.expand_top_n,
        )
        # Run the MCTS search
        mcts()
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        print(f"Worker {env_id}: Exception occurred: {e}")
        raise e

    finally:
        tamarin_env.close_client()  # ensure environment is properly closed

        print(f"Worker {env_id}: stopped.")


def run_cache_worker(admin_conn, fill_queue, max_memory_mb: float) -> None:
    print("Cache worker starting...")
    tamarin_cache = TamarinCache(admin_conn, fill_queue, max_memory_mb)
    tamarin_cache()
    print("Cache worker stopped.")


class Trainer:
    def __init__(self, config: RLConfig) -> None:
        """Main training function for the RL agent."""
        print(
            f"stdout resolves to: {os.path.realpath(f'/proc/{os.getpid()}/fd/1')}",
            flush=True,
        )
        self.device = create_device(config.main.force_cpu)
        self.steps = 0

        self.config = config

        self.lemmas_to_train = [
            (lemma, 0) for lemma in config.lemmas
        ]  # lemma and startup delay
        self.lemmas_complete: dict[LemmaConfig, int] = {}
        self.searches: dict[LemmaConfig, int] = {}
        self.processes: dict[int, tuple[Process, LemmaConfig, int | None]] = {}
        self.budgets: dict[LemmaConfig, int] = {}
        self.worker_last_activity: dict[int, float] = {}
        self.worker_steps: dict[int, int] = {}
        self.first_completion_time: dict[LemmaConfig, float] = {}
        self.min_tree_size: dict[LemmaConfig, int] = {}
        self.training_state = "running"

        self.search_config = config.search
        self.ucb_config = config.ucb
        self.reward_config = config.reward
        self.wandb_run = None
        if config.main.use_wandb:
            import wandb

            self.wandb_run = wandb.init(
                project="tamarin-rl",
                config=dataclass_to_dict(config),
                name=config.main.run_name + config.main.wandb_suffix,
                id=config.main.run_name,
                resume="allow",
            )

        self.recorder = Recorder(
            config.recorder, config.main.run_path, self.wandb_run, config.env.num_envs
        )
        self.agent, self.steps = ACAgent.load_or_create_agent(
            config.main,
            config.agent,
            config.optimizer,
            config.tokenizer,
            config.transformer,
            self.device,
            self.recorder,
        )

        # Determinism check: If we want strict determinism, we can't rely on
        # queue.empty() or poll(). We need a strict cadence.
        self.deterministic_mode = config.env.num_envs == 1

        self._setup_queues()
        self.setup_tamarin_cache_process()
        self.setup_worker_process()

        self.replay_buffer: ReplayBuffer[ACTrainingExample] = ReplayBuffer(
            config.agent.replay_buffer_capacity,
            test_set_size=config.agent.test_set_size,
        )

        self.root_state: dict[LemmaConfig, State] = {}

        self.search_state: dict[int, dict[str, float | int]] = {}

    def aggregate_search_state(
        self, env_idx: int, search_step: dict[str, float | int]
    ) -> dict[str, float | int]:
        sum_keys = [
            "SearchState/num_expanded_nodes",
            "SearchState/num_original_nodes",
            "SearchState/num_nodes",
            "SearchState/num_deduplicated_nodes",
            "SearchState/num_contradictory_nodes",
            "SearchState/failures",
        ]
        max_keys = ["SearchState/max_expanded_depth"]

        if env_idx not in self.search_state:
            self.search_state[env_idx] = search_step
        else:
            self.search_state[env_idx].update(search_step)

        ret_search_step = {}

        for key in sum_keys:
            ret_search_step[key] = sum(
                self.search_state[i][key] for i in self.search_state
            )
        for key in max_keys:
            ret_search_step[key] = max(
                self.search_state[i][key] for i in self.search_state
            )

        remaining_keys = (
            {k for v in self.search_state.values() for k in v.keys()}
            - set(sum_keys)
            - set(max_keys)
        )

        for key in remaining_keys:

            values = [
                self.search_state[i][key]
                for i in self.search_state
                if key in self.search_state[i]
            ]

            ret_search_step[key] = sum(values) / len(values)
        return ret_search_step

    def _process_step_log(self, env_idx: int, step_log: dict[str, float | int]) -> None:
        self.worker_steps[env_idx] = int(step_log.pop("SearchState/step"))
        agg_step_log = self.aggregate_search_state(env_idx, step_log)
        agg_step_log.update({f"Worker/{env_idx}/{k}": v for k, v in step_log.items()})
        agg_step_log["EnvStep/num_envs"] = len(self.processes)
        self.recorder.log_step(agg_step_log, self.steps)

    def refresh_pipes(
        self, idx
    ) -> tuple[Connection, Connection, Connection, Connection]:
        """Close pipes and create new ones for"""
        # Close existing pipes
        if idx in self.inference_request_conns:
            assert (
                idx in self.inference_response_conns
            ), f"Response connection for idx {idx} not found when refreshing pipes."
            self.inference_request_conns[idx].close()
            self.inference_response_conns[idx].close()

        # Create new pipes
        request_conn, worker_request_conn = Pipe(duplex=False)
        worker_response_conn, response_conn = Pipe(duplex=False)

        # Update connections
        self.inference_request_conns[idx] = request_conn
        self.inference_response_conns[idx] = response_conn

        cache_request_conn, worker_cache_request_conn = Pipe(duplex=False)
        worker_cache_response_conn, cache_response_conn = Pipe(duplex=False)

        if self.tamarin_cache_process is not None:
            self.cache_admin_conn[1].send(
                (
                    "UPDATE",
                    idx,
                    cache_request_conn,
                    cache_response_conn,
                )
            )

        # Close the ends that were sent to the cache, as the Trainer doesn't need them
        cache_request_conn.close()
        cache_response_conn.close()

        return (
            worker_request_conn,
            worker_response_conn,
            worker_cache_request_conn,
            worker_cache_response_conn,
        )

    def _setup_queues(self) -> None:
        """Sets up the pipes for communication between trainer and actors."""

        self.manager = Manager()

        # Connections for receiving action requests in the trainer (blocking)
        self.inference_request_conns: dict[int, "Connection[tuple[str, State]]"] = {}
        # Connections for sending action responses from the trainer (blocking)
        self.inference_response_conns: dict[
            int, "Connection[tuple[str, torch.Tensor, float]| None]"
        ] = {}

        self.fill_cache_queue: "Queue[tuple[LemmaConfig, State, list[dict[str, TamarinEnvironment.ResponseState]], list[int]]]" = cast(
            "Queue[tuple[LemmaConfig, State, list[dict[str, TamarinEnvironment.ResponseState]], list[int]]]",
            self.manager.Queue(),
        )

        # Get experiences from the workers (non-blocking)
        self.training_example_queue: "Queue[list[ACTrainingExample]]" = cast(
            "Queue[list[ACTrainingExample]]",
            self.manager.Queue(),
        )
        # Get root states from the workers, whenever new search starts (non-blocking)
        self.init_queue: "Queue[tuple[LemmaConfig, State, int, int]]" = cast(
            "Queue[tuple[LemmaConfig, State, int, int]]", self.manager.Queue()
        )
        # Get search summaries from the workers (non-blocking)
        self.search_result_queue: "Queue[tuple[LemmaConfig, dict[str, Any]]]" = cast(
            "Queue[tuple[LemmaConfig, dict[str, Any]]]", self.manager.Queue()
        )
        # Get step logs from the workers (non-blocking)
        self.step_log_queue: "Queue[tuple[int, dict[str, float | int]]]" = cast(
            "Queue[tuple[int, dict[str, float | int]]]",
            self.manager.Queue(),
        )

    def setup_worker_process(self) -> bool:
        # try to find a lemma that is not on startup delay
        # start such lemmas if num_envs allow it
        # A lemma is also eligible if it has delay > 0 but no worker is currently running it
        active_lemmas = {lemma for _, (_, lemma, _) in self.processes.items()}
        worker_counts: dict = {}
        for _, (_, lemma, _) in self.processes.items():
            worker_counts[lemma] = worker_counts.get(lemma, 0) + 1
        max_w = self.search_config.max_workers_per_lemma
        lemmas = [
            (lemma, delay)
            for lemma, delay in self.lemmas_to_train
            if (delay == 0 or lemma not in active_lemmas)
            and worker_counts.get(lemma, 0) < max_w
        ]
        # if all remaining lemmas are at the worker limit, allow any of them
        if not lemmas:
            lemmas = [
                (lemma, delay)
                for lemma, delay in self.lemmas_to_train
                if delay == 0 or lemma not in active_lemmas
            ]
        any_started = False
        while len(self.processes) < self.config.env.num_envs and len(lemmas) > 0:
            any_started = True
            # start env and reinsert lemma with delay
            lemma_config, delay = lemmas.pop(0)
            self.lemmas_to_train.remove((lemma_config, delay))
            self.lemmas_to_train.append((lemma_config, self.config.env.stall_frequency))

            if lemma_config not in self.searches:
                self.searches[lemma_config] = 0
            if lemma_config not in self.lemmas_complete:
                self.lemmas_complete[lemma_config] = 0
            if lemma_config not in self.budgets:
                self.budgets[lemma_config] = self.search_config.budget

            env_idx = next(
                (i for i in range(self.config.env.num_envs) if i not in self.processes),
                None,
            )
            assert env_idx is not None, "No available environment index found."

            (
                worker_request_conn,
                worker_response_conn,
                worker_cache_request_conn,
                worker_cache_response_conn,
            ) = self.refresh_pipes(env_idx)

            p = Process(
                target=run_worker,
                args=(
                    lemma_config,
                    worker_request_conn,
                    worker_response_conn,
                    worker_cache_request_conn,
                    worker_cache_response_conn,
                    self.fill_cache_queue,
                    self.training_example_queue,
                    self.init_queue,
                    self.search_result_queue,
                    self.step_log_queue,
                    self.search_config,
                    self.budgets[lemma_config],
                    self.ucb_config,
                    self.reward_config,
                    env_idx,
                    self.config.main.seed,
                    self.config.env.request_timeout,
                    self.config.env.backup_request_timeout,
                    self.deterministic_mode,
                ),
            )
            p.start()

            # Close the worker-side connections in the trainer process.
            # The worker process now has its own handles.
            worker_request_conn.close()
            worker_response_conn.close()
            worker_cache_request_conn.close()
            worker_cache_response_conn.close()

            self.processes[env_idx] = (p, lemma_config, None)  # None for not set yet
            self.worker_last_activity[env_idx] = time.time()
            print(
                f"Trainer: Starting training on {lemma_config.lemma_name} with theory: {lemma_config.theory_path}"
            )

        return any_started

    def setup_tamarin_cache_process(self) -> None:
        # "EXIT"  for closing, ("UPDATE", idx, cache_req_reader, cache_res_writer)
        self.cache_admin_conn = Pipe(duplex=False)  # receive, send

        self.tamarin_cache_process = Process(
            target=run_cache_worker,
            args=(
                self.cache_admin_conn[0],
                self.fill_cache_queue,
                self.config.env.cache_max_memory_mb,
            ),
        )
        self.tamarin_cache_process.start()
        self.cache_admin_conn[0].close()
        print("Trainer: Tamarin cache process started.", flush=True)

    @staticmethod
    def _graceful_kill(
        p: Process, label: str, join_timeout: int = 20, terminate_timeout: int = 5
    ) -> None:
        """Join → terminate → kill escalation for a multiprocessing.Process."""
        p.join(join_timeout)
        if p.is_alive():
            print(f"Trainer: {label} did not exit in time. Terminating...", flush=True)
            p.terminate()
            p.join(terminate_timeout)
            if p.is_alive():
                print(f"Trainer: {label} did not terminate. Killing...", flush=True)
                p.kill()
                p.join()

    def kill_process(self, idx: int) -> None:
        print(
            f"Trainer: Killing process {idx} for {self.processes[idx][1].lemma_name}..."
        )
        try:
            self.inference_response_conns[idx].send(None)
        except (OSError, BrokenPipeError):
            pass

        # Drain request pipe to unblock worker if it is waiting to write
        try:
            while self.inference_request_conns[idx].poll():
                self.inference_request_conns[idx].recv()
        except (OSError, EOFError, BrokenPipeError):
            pass

        p, lemma_config, tamarin_pid = self.processes[idx]
        self._graceful_kill(p, f"Worker {idx}")
        assert not p.is_alive(), f"Worker {idx} is still alive after kill attempt."

        if tamarin_pid is not None and psutil.pid_exists(tamarin_pid):
            try:
                process = psutil.Process(tamarin_pid)
                print(
                    f"Trainer: Attempting to terminate tamarin process {tamarin_pid} for environment {idx}..."
                )
                process.terminate()
                process.wait(timeout=3)
            except psutil.TimeoutExpired:
                print(
                    f"Trainer: Tamarin process {tamarin_pid} for environment {idx} did not terminate in time. Killing..."
                )
                process.kill()
                process.wait()
            except psutil.NoSuchProcess:
                print(
                    f"Trainer: Tamarin process {tamarin_pid} for environment {idx} already exited."
                )
            except psutil.AccessDenied:
                print(f"WARNING: Trainer: Permission denied to kill PID {tamarin_pid}.")

        del self.processes[idx]
        if idx in self.worker_last_activity:
            del self.worker_last_activity[idx]

        print(
            f"Trainer: Worker {idx} for {lemma_config.lemma_name} killed successfully.",
            flush=True,
        )

    def kill_tamarin_cache_process(self) -> None:
        try:
            self.cache_admin_conn[1].send("EXIT")
        except (OSError, BrokenPipeError):
            pass
        assert self.tamarin_cache_process is not None
        self._graceful_kill(self.tamarin_cache_process, "Node cache process", 5, 5)
        print("Trainer: Node cache process joined")

    def ensure_cache_alive(self) -> None:
        if self.tamarin_cache_process is None:
            return
        if self.tamarin_cache_process.is_alive():
            return
        print(
            f"WARNING: Trainer: Cache process died (exit code "
            f"{self.tamarin_cache_process.exitcode}). Restarting...",
            flush=True,
        )
        self.tamarin_cache_process.join()
        try:
            self.cache_admin_conn[1].close()
        except OSError:
            pass
        self.setup_tamarin_cache_process()

    def check_root_states(self) -> None:
        try:
            while True:
                lemma, state, pid, env_id = self.init_queue.get_nowait()
                self.root_state[lemma] = state
                if env_id not in self.processes or lemma != self.processes[env_id][1]:
                    continue  # this can happen if a worker was killed and restarted with a different lemma before the old one was cleaned up from the queue
                self.processes[env_id] = (self.processes[env_id][0], lemma, pid)
        except Empty:
            pass

    def initial_search_log(self) -> None:
        # Initial log before training starts
        initial_log = {
            "SearchOverview/searches": 0,
            "SearchOverview/lemmas_completed": 0,
            "SearchOverview/lemmas_completed_sum": 0,
            "SearchOverview/lemmas_done": 0,
            "SearchOverview/lemmas_exhausted": 0,
            "SearchOverview/lemmas_pending": len(self.config.lemmas),
            "SearchOverview/lemmas_failed": 0,
        }
        keys = ["steps", "lemma_completes", "searches", "dry_spell"]
        for lemma_config in self.config.lemmas:
            for key in keys:
                initial_log[
                    f"Search/{lemma_config.theory_name}/{lemma_config.lemma_name}/{key}"
                ] = 0

        self.recorder.log_step(
            initial_log,
            self.steps,
        )

    def log_search_result(self) -> bool:
        # log search results and update repeats
        any_search_complete = False
        try:
            while True:
                lemma_config, search_result = self.search_result_queue.get_nowait()
                any_search_complete = True

                self.budgets[lemma_config] = max(
                    search_result.get("new_budget", self.budgets[lemma_config]),
                    self.budgets[lemma_config],
                )

                if search_result["solved"] or search_result["contradictory"]:
                    self.searches[lemma_config] += 1
                    self.lemmas_complete[lemma_config] += 1
                    if lemma_config not in self.first_completion_time:
                        self.first_completion_time[lemma_config] = (
                            time.time() - self.recorder.start_time
                        )
                    if "tree_size" in search_result:
                        prev = self.min_tree_size.get(lemma_config, search_result["tree_size"])
                        self.min_tree_size[lemma_config] = min(prev, search_result["tree_size"])
                    print(
                        f"Trainer: Solved {lemma_config.lemma_name}!"
                        if search_result["solved"]
                        else f"Trainer: Contradictory {lemma_config.lemma_name}!"
                    )
                    print(
                        f"Trainer: Completed {self.lemmas_complete[lemma_config]}/{min(self.config.search.num_lemma_completes, self.config.search.num_searches)} for {lemma_config.lemma_name} successful searches."
                    )
                elif (
                    "stopped_early" in search_result and search_result["stopped_early"]
                ):
                    print(
                        f"Trainer: Search for {lemma_config.lemma_name} stopped early after {search_result['steps']} steps."
                    )
                else:
                    self.searches[lemma_config] += 1
                    print(
                        f"Trainer: Search {self.searches[lemma_config]}/{self.config.search.num_searches} for {lemma_config.lemma_name} ran out of budget."
                    )

                search_log = {
                    "SearchOverview/searches": sum(self.searches.values()),
                    "SearchOverview/lemmas_completed_sum": sum(
                        self.lemmas_complete.values()
                    ),
                    f"Search/{lemma_config.theory_name}/{lemma_config.lemma_name}/steps": search_result[
                        "steps"
                    ],
                    f"Search/{lemma_config.theory_name}/{lemma_config.lemma_name}/lemma_completes": self.lemmas_complete[
                        lemma_config
                    ],
                    f"Search/{lemma_config.theory_name}/{lemma_config.lemma_name}/searches": self.searches[
                        lemma_config
                    ],
                    f"Search/{lemma_config.theory_name}/{lemma_config.lemma_name}/dry_spell": search_result[
                        "dry_spell"
                    ],
                }

                num_lemmas_completed = 0
                num_lemmas_done = 0
                num_lemmas_exhausted = 0
                num_lemmas_pending = 0
                num_lemmas_failed = 0

                for lemma in self.config.lemmas:
                    n_complete = self.lemmas_complete.get(lemma, 0)
                    n_searches = self.searches.get(lemma, 0)

                    # lemma is complete if solved/contradictory at least once
                    complete = n_complete > 0
                    # lemma is done if it has either been solved/contradictory required number of times or exhausted search budget required number of times
                    done = (
                        n_complete >= self.config.search.num_lemma_completes
                        or n_searches >= self.config.search.num_searches
                    )
                    # lemma is exhausted if it has exhausted search budget required number of times and not been solved/contradictory required number of times
                    exhausted = (
                        n_searches >= self.config.search.num_searches
                        and n_complete < self.config.search.num_lemma_completes
                    )
                    # lemma is pending if it has not been solved/contradictory once but has not exhausted search budget required number of times
                    pending = (
                        n_complete == 0 and n_searches < self.config.search.num_searches
                    )
                    # lemma is failed if it has exhausted search budget required number of times and not been solved/contradictory once
                    failed = (
                        n_searches >= self.config.search.num_searches
                        and n_complete == 0
                    )

                    num_lemmas_completed += int(complete)
                    num_lemmas_done += int(done)
                    num_lemmas_exhausted += int(exhausted)
                    num_lemmas_pending += int(pending)
                    num_lemmas_failed += int(failed)

                search_log["SearchOverview/lemmas_completed"] = num_lemmas_completed
                search_log["SearchOverview/lemmas_exhausted"] = num_lemmas_exhausted
                search_log["SearchOverview/lemmas_pending"] = num_lemmas_pending
                search_log["SearchOverview/lemmas_done"] = num_lemmas_done
                search_log["SearchOverview/lemmas_failed"] = num_lemmas_failed

                if "tree_size" in search_result:
                    search_log[
                        f"Search/{lemma_config.theory_name}/{lemma_config.lemma_name}/tree_size"
                    ] = search_result["tree_size"]

                self.recorder.log_step(
                    search_log,
                    self.steps,
                )

                self.save_tree(search_result, lemma_config)
                self.write_raw_json()

        except Empty:
            pass
        if any_search_complete:
            self.recorder.write_log(self.steps, force=True)
        return any_search_complete

    def save_tree(
        self, search_result: dict[str, Any], lemma_config: LemmaConfig
    ) -> None:
        if "tree" in search_result and "proof_file" in search_result:
            tree_dir = os.path.join(self.config.main.run_path, "proof_trees")
            os.makedirs(tree_dir, exist_ok=True)
            tree_path = os.path.join(
                tree_dir,
                f"{lemma_config.theory_name}_{lemma_config.lemma_name}_search_{self.searches[lemma_config]}_complete_{self.lemmas_complete[lemma_config]}",
            )
            with open(tree_path + ".json", "w", encoding="utf-8") as f:
                json.dump(search_result["tree"], f, indent=2)
            with open(tree_path + ".spthy", "w", encoding="utf-8") as f:
                f.write(search_result["proof_file"])

    def write_raw_json(self) -> None:
        run_id = self.config.main.run_name
        data: dict[str, Any] = {
            run_id: {
                "_meta": {
                    "name": run_id,
                    "state": self.training_state,
                    "config": dataclass_to_dict(self.config),
                }
            }
        }
        for lemma in self.config.lemmas:
            theory = lemma.theory_name
            if theory not in data[run_id]:
                data[run_id][theory] = {}
            entry: dict[str, Any] = {
                "completes": self.lemmas_complete.get(lemma, 0),
                "type": lemma.lemma_type.value,
            }
            if lemma in self.first_completion_time:
                entry["first_completion_time"] = round(
                    self.first_completion_time[lemma], 1
                )
            if lemma in self.min_tree_size:
                entry["tree_size"] = self.min_tree_size[lemma]
            data[run_id][theory][lemma.lemma_name] = entry

        path = os.path.join(self.config.main.run_path, "raw.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def check_alive_processes(self) -> None:
        # Check for crashed workers
        for env_idx, (p, lemma, tamarin_pid) in list(self.processes.items()):
            if not p.is_alive():
                if p.exitcode != 0:
                    print(
                        f"WARNING: Worker {env_idx} (Lemma: {lemma.lemma_name}) died unexpectedly with exit code {p.exitcode}. Stopping..."
                    )
                else:
                    print(
                        f"Trainer: Worker {env_idx} (Lemma: {lemma.lemma_name}) terminated with exit code 0. Cleaning up."
                    )
                p.join()
                del self.processes[env_idx]

    def clean_processes(self) -> bool:

        self.check_alive_processes()

        # collect all lemmas that have completed required number of searches or lemma completions
        any_killed = False
        lemma_completes = set(
            lemma
            for lemma, count in self.lemmas_complete.items()
            if count >= self.config.search.num_lemma_completes
        )
        lemma_exhausted = set(
            lemma
            for lemma, count in self.searches.items()
            if count >= self.config.search.num_searches
        )
        kill_list = lemma_completes | lemma_exhausted
        # get the corresponding process indices and kill them
        kill_idx_list = [
            (idx, lemma)
            for idx, (_, lemma, _) in self.processes.items()
            if lemma in kill_list
        ]
        for kill_idx, lemma in kill_idx_list:
            any_killed = True
            self.kill_process(kill_idx)
            print(f"Trainer: Finished all training on {lemma.lemma_name}.")
            self.lemmas_to_train = [
                (_lemma, delay)
                for _lemma, delay in self.lemmas_to_train
                if _lemma != lemma
            ]

        self.ensure_cache_alive()

        # start new processes if possible
        any_started = self.setup_worker_process()

        return any_killed or any_started

    def snapshot_worker_state(self) -> None:

        def get_memory_usage(pid: int | None) -> int | None:
            if pid is None or not psutil.pid_exists(pid):
                return None
            try:
                process = psutil.Process(pid)
                mem_info = process.memory_info()
                return mem_info.rss // (1024 * 1024)  # Convert bytes to MB
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                return None

        # headers = [
        #     "Type",
        #     "ID",
        #     "PID",
        #     "mem (MB)",
        #     "t_pid",
        #     "t_mem (MB)",
        #     "Lemma",
        #     "Theory",
        #     "Budget",
        #     "Comp",
        #     "Srch",
        #     "Delay/Act",
        # ]
        data: list[list[str | int | None]] = []
        now = time.time()

        mem = get_memory_usage(os.getpid()) or 0

        data.append(
            [
                "TRAINER",
                os.getpid(),
                mem,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ]
        )

        if self.tamarin_cache_process is not None:

            mem_cache = get_memory_usage(self.tamarin_cache_process.pid)
            data.append(
                [
                    "CACHE",
                    self.tamarin_cache_process.pid,
                    mem_cache,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ]
            )
            mem += mem_cache or 0

        for env_id in sorted(self.processes.keys()):
            p, lemma, tamarin_pid = self.processes[env_id]
            n_complete = self.lemmas_complete.get(lemma, 0)
            n_searches = self.searches.get(lemma, 0)
            last_act = now - self.worker_last_activity.get(env_id, now)
            t_mem_this = get_memory_usage(tamarin_pid)
            mem_this = get_memory_usage(p.pid)
            mem += (t_mem_this or 0) + (mem_this or 0)
            if env_id in self.search_state and t_mem_this is not None:
                self.search_state[env_id]["SearchState/tamarin_mem"] = t_mem_this
            data.append(
                [
                    f"WORKER {env_id}",
                    p.pid,
                    mem_this,
                    tamarin_pid,
                    t_mem_this,
                    lemma.lemma_name[:40],
                    lemma.theory_name[:30],
                    f"{self.worker_steps.get(env_id, 0)}/{self.budgets.get(lemma, self.search_config.budget)}",
                    n_complete,
                    n_searches,
                    int(round(last_act, 0)),
                ]
            )

        for lemma, delay in self.lemmas_to_train:
            n_complete = self.lemmas_complete.get(lemma, 0)
            n_searches = self.searches.get(lemma, 0)
            data.append(
                [
                    "QUEUE",
                    None,
                    None,
                    None,
                    None,
                    lemma.lemma_name[:40],
                    lemma.theory_name[:30],
                    f"{self.budgets.get(lemma, self.search_config.budget)}",
                    n_complete,
                    n_searches,
                    delay,
                ]
            )

        finished_lemmas = []
        for lemma in self.config.lemmas:
            n_complete = self.lemmas_complete.get(lemma, 0)
            n_searches = self.searches.get(lemma, 0)
            if (
                n_complete >= self.config.search.num_lemma_completes
                or n_searches >= self.config.search.num_searches
            ):
                finished_lemmas.append((lemma, n_complete, n_searches))

        for lemma, n_complete, n_searches in finished_lemmas:
            data.append(
                [
                    "DONE",
                    None,
                    None,
                    None,
                    None,
                    lemma.lemma_name[:40],
                    lemma.theory_name[:30],
                    f"{self.budgets.get(lemma, self.search_config.budget)}",
                    n_complete,
                    n_searches,
                    None,
                ]
            )
        self.recorder.log_system_state(data, self.steps)
        self.recorder.log_step({"System/memory_usage": mem}, self.steps)
        self._log_disk_usage()

    def _log_disk_usage(self) -> None:
        log: dict[str, float] = {}

        paths_to_check = {
            "tmp": os.environ.get("TMPDIR", "/tmp"),
            "stdout": f"/proc/{os.getpid()}/fd/1",
        }
        for label, path in paths_to_check.items():
            try:
                usage = shutil.disk_usage(os.path.realpath(path))
                log[f"System/disk_{label}_used_gb"] = usage.used / 1024**3
                log[f"System/disk_{label}_free_gb"] = usage.free / 1024**3
                log[f"System/disk_{label}_used_pct"] = 100 * usage.used / usage.total
            except Exception:
                pass

        login = os.environ.get("USER", os.environ.get("USERNAME", ""))
        tmpdir = os.environ.get("TMPDIR", "/tmp")
        for cache_dir in [
            os.path.join(tmpdir, f"tamarin-prover-cache-{login}"),
            "/tmp/tamarin",
        ]:
            if os.path.exists(cache_dir):
                try:
                    size_bytes = sum(
                        f.stat().st_size for f in os.scandir(cache_dir) if f.is_file()
                    )
                    label = "tamarin_cache_" + os.path.basename(cache_dir).replace(
                        "-", "_"
                    )
                    log[f"System/disk_{label}_mb"] = size_bytes / 1024**2
                except Exception:
                    pass

        self.recorder.log_step(log, self.steps)

    def __call__(self):
        """Main training loop.

        The loop processes queues and pipes in the following order:
        1. init_queue: Non-blocking. Updates the initial state if available.
        2. action request and response connections: Non-blocking between environment
           but blocks to send a response after a received request.
        3. experience_queue: Non-blocking. Collects all available experiences.
        4. simulation_result_queue: Non-blocking. Collects simulation results.
        5. search_result_queue: Non-blocking. Collects search results.
        """

        self.initial_search_log()
        self.snapshot_worker_state()

        exit_code = 0

        try:
            # Training loop continues until all processes have completed the required number of lemma completions or searches.
            # processes are dynamically added and removed depending on the number of lemmas to train and the number of environments.
            pbar = tqdm(initial=self.steps, desc="Training Steps")
            while len(self.processes) > 0:

                self.check_root_states()

                start_wait_time = time.time()
                while True:
                    readers = [
                        self.inference_request_conns[env_idx]
                        for env_idx in self.processes.keys()
                    ]
                    if not readers or wait(readers, timeout=1):
                        break

                    self.check_alive_processes()

                    if time.time() - start_wait_time > QUEUE_TIMEOUT:
                        raise QueueWaitTimeoutError(
                            "Timeout while waiting for action requests from workers."
                        )

                for env_idx in self.processes.keys():
                    # In deterministic mode, we block until we get a request if we expect one
                    # or we simply block-wait if we know the worker is active.
                    # However, simply removing poll() for num_envs=1 helps significantly
                    # if the worker is guaranteed to be waiting.
                    try:
                        if (
                            self.deterministic_mode
                            or self.inference_request_conns[env_idx].poll()
                        ):
                            self.worker_last_activity[env_idx] = time.time()
                            request_id, state = self.inference_request_conns[
                                env_idx
                            ].recv()

                            policy_tensor, value = self.agent.inference(state)

                            assert policy_tensor.shape[0] == len(
                                state.proof_methods
                            ), "Number of proof methods doesn't match tensor shape"
                            self.inference_response_conns[env_idx].send(
                                (request_id, policy_tensor.to("cpu"), value.item())
                            )
                    except (EOFError, OSError):
                        print(f"Trainer: Lost connection to worker {env_idx}. skipping")
                        continue

                if self.deterministic_mode:
                    assert (
                        len(self.processes) <= 1
                    ), "Deterministic mode only supports 1 env"
                    # If the worker died between inference rounds, don't block
                    # on the step log queue — nothing will arrive.
                    self.check_alive_processes()
                    if self.processes:
                        # In deterministic mode, we block on the step log (always produced),
                        # then drain training examples (not always produced).
                        env_idx, step_log = self.step_log_queue.get(
                            timeout=QUEUE_TIMEOUT
                        )
                        # By the time the step log arrives, any training examples
                        # from this step are already on the queue.
                        try:
                            self.replay_buffer.extend(
                                self.training_example_queue.get_nowait()
                            )
                        except Empty:
                            pass
                        self.steps += 1
                        pbar.update(1)
                        self._process_step_log(env_idx, step_log)
                else:
                    # Non-blocking collection of training sample and logs from the experience queue
                    try:
                        while True:
                            self.replay_buffer.extend(
                                self.training_example_queue.get_nowait()
                            )
                    except Empty:
                        pass
                    try:
                        while True:
                            env_idx, step_log = self.step_log_queue.get_nowait()
                            self.steps += 1
                            pbar.update(1)
                            self._process_step_log(env_idx, step_log)
                    except Empty:
                        pass

                updated = self.agent.update_model(
                    replay_buffer=self.replay_buffer,
                    env_steps=self.steps,
                    root_state=self.root_state,
                )
                # decrease delay if updated
                # delay ensures that we don't start a new search on the same lemma with the same model weights.
                self.lemmas_to_train = [
                    (lemma, max(0, delay - int(updated)))
                    for lemma, delay in self.lemmas_to_train
                ]

                any_search_complete = self.log_search_result()

                # if search complete, save model checkpoint
                if any_search_complete:
                    self.agent.save_checkpoint(
                        self.config.main.run_path,
                        f"_step_{self.agent.model_updates}_search_{sum(self.searches.values())}_complete_{sum(self.lemmas_complete.values())}",
                        self.steps,
                    )

                # clean up processes that have completed required number of lemma completions or searches and start new ones if possible
                # start new processes if possible
                self.clean_processes()

                self.snapshot_worker_state()
            pbar.close()

            print(
                f"Training complete! Total steps: {self.steps}, Total model updates: {self.agent.model_updates}"
            )
            self.training_state = "finished"
            self.write_raw_json()

        except Exception as e:
            print(f"Error during training on: {e}")
            exit_code = 1
            raise e

        finally:
            try:
                for process_idx in list(self.processes.keys()):
                    self.kill_process(process_idx)
                self.kill_tamarin_cache_process()
            except Exception as e:
                print(f"Error during cleanup: {e}")
                exit_code = 1
                raise e
            finally:
                if self.wandb_run:
                    self.wandb_run.finish(exit_code=exit_code)
