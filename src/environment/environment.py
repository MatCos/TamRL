from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.connection import Connection
from typing import Any

import psutil
import requests

from src.client.client import TamarinClient
from src.parser.lemma import LemmaConfig
from src.utils.utils import (
    CASE_NAME,
    CONTRADICTORY,
    ENCODED_METHODS,
    ENCODED_SYS,
    PROOF_METHODS,
    QUEUE_TIMEOUT,
    SOLVED,
    STATUS,
    THEORY_INDEX,
    THEORY_KIND,
    QueueWaitTimeoutError,
    TamarinCallTimeoutError,
)

MAX_REPEATS = 3


@dataclass(frozen=True, eq=False)
class State:
    """A class representing a state in the Tamarin proof search."""

    proof_methods: list[dict[str, Any]]
    encoded_methods: tuple[str, ...]
    encoded_sys: str

    @classmethod
    def from_response(cls, response: TamarinEnvironment.ResponseState) -> "State":
        return cls(
            proof_methods=response.proof_methods,
            encoded_methods=tuple(response.encoded_methods),
            encoded_sys=response.encoded_sys,
        )

    def __hash__(self) -> int:
        return hash(self.encoded_methods) ^ hash(self.encoded_sys)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, State):
            return False
        return (
            self.encoded_sys == value.encoded_sys
            and self.encoded_methods == value.encoded_methods
        )

    def __str__(self) -> str:
        return f"State(encoded_sys={self.encoded_sys}, encoded_methods={self.encoded_methods}, proof_methods={self.proof_methods})"


class TamarinEnvironment:
    """A long-lived RL environment that manages Tamarin clients for different proofs."""

    @dataclass(frozen=True)
    class ResponseState:
        """A class representing a state in the Tamarin proof search."""

        proof_methods: list[dict[str, Any]]
        encoded_methods: list[str]
        encoded_sys: str
        status: str

        @property
        def is_terminal(self) -> bool:
            """Checks whether the proof is finished in this state."""
            return self.status in {CONTRADICTORY, SOLVED}

        @property
        def is_contradictory(self) -> bool:
            """Checks whether the proof is contradictory in this state."""
            return self.status == CONTRADICTORY

        @property
        def is_solved(self) -> bool:
            """Checks whether the proof is solved in this state."""
            return self.status == SOLVED

    def __init__(
        self,
        cache_request_conn: "Connection[tuple[str, LemmaConfig, State, list[int]]]",
        cache_response_conn: "Connection[tuple[str, list[dict[str, TamarinEnvironment.ResponseState]] | None, float, float, float, list[int], list[int]]]",
        fill_cache_conn: "Queue[tuple[LemmaConfig, State, list[int], list[dict[str, TamarinEnvironment.ResponseState]], list[int], float]]",
        init_queue: "Queue[tuple[LemmaConfig, State, int, int]]",
        timeout: int,
        backup_timeout: int,
        id: int,
    ) -> None:
        """Initializes the TamarinEnvironment."""
        self.lemma_config: LemmaConfig | None = None
        self.client: TamarinClient | None = None
        self.theory_kind: str | None = None
        self.initial_state: State | None = None
        self.theory_index: int | None = None
        self.ip_addr: str | None = None
        self.env_steps: int = 0
        self.cache_request_conn = cache_request_conn
        self.cache_response_conn = cache_response_conn
        self.fill_cache_conn = fill_cache_conn
        self.init_queue = init_queue
        self.timeout = timeout
        self.backup_timeout = backup_timeout
        self.id = id

        self.cache_hit_rate: float = 0.0
        self.cache_usage: float = 0.0
        self._cache_alive: bool = True

    def set(
        self,
        lemma_config: LemmaConfig,
        ip_addr: str = "127.0.0.1",
    ):
        """Resets the environment with a new lemma to prove."""
        if self.client:
            self.close_client()

        self.lemma_config = lemma_config

        self.env_steps = 0
        self.ip_addr = ip_addr

        self.client = TamarinClient(
            theory_path=self.lemma_config.theory_path,
            ip_addr=ip_addr,
            suppress_output=self.lemma_config.suppress_output,
            diff_arg=self.lemma_config.diff_arg,
            theory_name=self.lemma_config.theory_name,
            heuristic=self.lemma_config.heuristic,
            timeout=self.backup_timeout,
        )

        print(
            f"Env {self.id}: Trying to connect to Tamarin server. Port {self.client.port}, PID {self.client.process.pid}",
            flush=True,
        )
        overview = self.client.server_overview
        if self.client.theory_name not in overview:
            print(
                f"WARNING: Env {self.id}: Theory name {self.client.theory_name} not found in server overview. Available theories: {list(overview.keys())}. Port {self.client.port}, PID {self.client.process.pid}. Resetting environment with a new port might fix this.",
                flush=True,
            )
            self.reset()
        theory = self.client.server_overview[self.client.theory_name]
        print(
            f"Env {self.id}: Connected to Tamarin server. Got server overview. Port: {self.client.port}, PID: {self.client.process.pid}, Theory: {theory['theoryName']}, Lemma: {self.lemma_config.lemma_name}",
            flush=True,
        )
        self.theory_kind = theory[THEORY_KIND]
        self.theory_index = theory[THEORY_INDEX]

        self.get_initial_state(timeout=self.backup_timeout)

    def reset(self) -> None:
        assert self.lemma_config is not None
        assert self.ip_addr is not None
        assert self.client is not None

        old_pid = self.client.process.pid
        self.set(
            lemma_config=self.lemma_config,
            ip_addr=self.ip_addr,
        )
        # if process still alive, kill it
        if psutil.pid_exists(old_pid):
            try:
                psutil.Process(old_pid).kill()
                print(
                    f"WARNING: Env {self.id}: Killed old Tamarin process in reset with PID {old_pid}"
                )
            except Exception as e:
                print(
                    f"WARNING: Env {self.id}: Failed to kill old Tamarin process in reset with PID {old_pid}: {e}"
                )

    def perform_action(
        self,
        state: State,
        method_index: int,
    ) -> dict[str, TamarinEnvironment.ResponseState] | None:
        """Applies a proof method to the system and returns the resulting cases
        with their systems in a dict."""
        assert (
            self.client is not None
        ), "Environment must be reset with a lemma before performing an action."
        assert self.theory_kind is not None
        assert self.lemma_config is not None

        request_id = str(uuid.uuid4())
        self.cache_request_conn.send(
            (request_id, self.lemma_config, state, [method_index])
        )

        while True:
            if not self.cache_response_conn.poll(QUEUE_TIMEOUT):
                raise QueueWaitTimeoutError("Timeout waiting for cache response")
            try:
                response_id, *cache_response_data = self.cache_response_conn.recv()
            except (EOFError, OSError) as e:
                print(
                    f"WARNING: Env {self.id}: Error reading from cache response pipe: {e}",
                    flush=True,
                )
                raise

            if response_id == request_id:
                cache_response = tuple(cache_response_data)
                break
            print(
                f"WARNING: Env {self.id}: Discarding stale cache response. Expected {request_id}, got {response_id} !!!"
            )

        # cache_response is (responses, call_time, hit_rate, usage, cached_failures, uncached)

        self.cache_hit_rate = cache_response[2]
        self.cache_usage = cache_response[3]
        if cache_response[0] is not None:
            uncached = cache_response[5]
            if len(uncached) == 0:
                if cache_response[0] == [] and method_index in cache_response[4]:
                    return None
                assert (
                    len(cache_response[0]) == 1
                ), f"Expected exactly one response from cache, got {len(cache_response[0])}"
                return cache_response[0][0]

        # Apply the chosen proof method
        response = self.client.apply_proofmethod_to_system(
            self.theory_kind,
            self.theory_index,
            self.lemma_config.lemma_name,
            state.encoded_methods[method_index],
            state.encoded_sys,
            timeout=self.timeout,
        )

        if not response.ok:
            # TODO: Cannot print a nice error here since we only have the
            # encoded system. Fix me?
            raise RuntimeError(
                f"Failed to apply proof method {method_index} in env {self.id} "
                f"for theory kind {self.theory_kind}, theory index {self.theory_index}, "
                f"lemma name {self.lemma_config.lemma_name}, side {self.lemma_config.side}."
            )

        # A list of case names mapped to their new states
        cases: list[dict[str, Any]] = json.loads(response.text)

        cases_dict: dict[str, TamarinEnvironment.ResponseState] = {
            case[CASE_NAME]: TamarinEnvironment.ResponseState(
                proof_methods=case["system"][PROOF_METHODS],
                encoded_methods=case["system"][ENCODED_METHODS],
                encoded_sys=case["system"][ENCODED_SYS],
                status=case["system"][STATUS],
            )
            for case in cases
        }

        return cases_dict

    def perform_action_bulk(
        self,
        state: State,
        method_index: list[int],
    ) -> tuple[list[dict[str, TamarinEnvironment.ResponseState]], list[int], float]:
        """Applies a proof method to the system and returns the resulting cases
        with their systems in a dict."""
        assert (
            self.client is not None
        ), "Environment must be reset with a lemma before performing an action."
        assert self.theory_kind is not None
        assert self.lemma_config is not None
        request_id = str(uuid.uuid4())

        # cache_response: (responses, duration, hit_rate, usage, cached_failures, uncached)
        cache_response: tuple[
            list[dict[str, TamarinEnvironment.ResponseState]] | None,
            float,
            float,
            float,
            list[int],
            list[int],
        ] = (
            None,
            0.0,
            self.cache_hit_rate,
            self.cache_usage,
            [],
            [],
        )

        if self._cache_alive:
            try:
                self.cache_request_conn.send(
                    (request_id, self.lemma_config, state, method_index)
                )
            except (BrokenPipeError, EOFError, OSError):
                print(
                    f"WARNING: Env {self.id}: Cache pipe broken on send. "
                    f"Falling back to direct Tamarin calls.",
                    flush=True,
                )
                self._cache_alive = False

        if self._cache_alive:
            while True:
                if not self.cache_response_conn.poll(QUEUE_TIMEOUT):
                    print(
                        f"WARNING: Env {self.id}: Timeout waiting for cache response. Ignoring cache and performing bulk action.",
                        flush=True,
                    )
                    break
                try:
                    response_id, *cache_response_data = (
                        self.cache_response_conn.recv()
                    )
                except (BrokenPipeError, EOFError, OSError):
                    print(
                        f"WARNING: Env {self.id}: Cache pipe broken on recv. "
                        f"Falling back to direct Tamarin calls.",
                        flush=True,
                    )
                    self._cache_alive = False
                    break

                if response_id == request_id:
                    cache_response = tuple(cache_response_data)
                    break
                print(
                    f"WARNING: Env {self.id}: Discarding stale cache response. Expected {request_id}, got {response_id}",
                    flush=True,
                )

        cached_responses = cache_response[0]
        cached_call_time = cache_response[1]
        self.cache_hit_rate = cache_response[2]
        self.cache_usage = cache_response[3]
        cached_failures = cache_response[4]
        uncached = cache_response[5]

        if cached_responses is not None and len(uncached) == 0:
            return cached_responses, cached_failures, cached_call_time

        indices_to_call = uncached if cached_responses is not None else method_index

        start_time = time.time()
        responses, fresh_failures = self.client_bulk_connection(state, indices_to_call)
        end_time = time.time()
        fresh_duration = end_time - start_time

        if any(not response.ok for response in responses):
            assert self.lemma_config is not None
            raise RuntimeError(
                f"Failed to apply proof method "
                f"for theory kind {self.theory_kind}, theory index {self.theory_index}, "
                f"lemma name {self.lemma_config.lemma_name}"
                f"System: {state.encoded_sys}\n"
                f"Encoded methods: {[state.encoded_methods[i] for i in indices_to_call]}\n"
                f"Proof methods: {[state.proof_methods[i] for i in indices_to_call]}\n"
                f"Responses: {[response.text for response in responses]}"
            )

        fresh_cases: list[dict[str, TamarinEnvironment.ResponseState]] = []
        for response in responses:
            cases: list[dict[str, Any]] = json.loads(response.text)
            cases_dict: dict[str, TamarinEnvironment.ResponseState] = {
                case[CASE_NAME]: TamarinEnvironment.ResponseState(
                    proof_methods=case["system"][PROOF_METHODS],
                    encoded_methods=case["system"][ENCODED_METHODS],
                    encoded_sys=case["system"][ENCODED_SYS],
                    status=case["system"][STATUS],
                )
                for case in cases
            }
            fresh_cases.append(cases_dict)

        assert len(fresh_cases) + len(fresh_failures) == len(
            indices_to_call
        ), "Number of responses and failures must add up to number of method indices."
        self.fill_cache_conn.put(
            (
                self.lemma_config,
                state,
                indices_to_call,
                list(fresh_cases),
                fresh_failures,
                fresh_duration,
            )
        )

        if cached_responses is None:
            return fresh_cases, fresh_failures, fresh_duration

        # Partial hit: merge cached + fresh results in method_index order
        result_map: dict[int, dict[str, TamarinEnvironment.ResponseState]] = {}
        cached_failures_set = set(cached_failures)
        uncached_set = set(uncached)
        fresh_failures_set = set(fresh_failures)
        all_failures_set = cached_failures_set | fresh_failures_set

        for i, resp in zip(
            (i for i in method_index if i not in cached_failures_set and i not in uncached_set),
            cached_responses,
        ):
            result_map[i] = resp

        for i, resp in zip(
            (i for i in indices_to_call if i not in fresh_failures_set),
            fresh_cases,
        ):
            result_map[i] = resp

        merged = [result_map[i] for i in method_index if i not in all_failures_set]
        all_failures = [i for i in method_index if i in all_failures_set]
        return merged, all_failures, max(cached_call_time, fresh_duration)

    def client_bulk_connection(
        self, state: State, method_index: list[int]
    ) -> tuple[list[requests.Response], list[int]]:

        assert (
            MAX_REPEATS >= 3
        ), "MAX_REPEATS must be at least 3 to allow for retries and a final attempt with sequential requests."

        assert (
            self.client is not None
        ), "Environment must be reset with a lemma before performing an action."
        assert self.theory_kind is not None
        assert self.lemma_config is not None
        for repeat in range(1, MAX_REPEATS + 1):
            try:
                if repeat < MAX_REPEATS:
                    # Apply the chosen proof method
                    responses: list[requests.Response] = (
                        self.client.apply_proofmethod_to_system_bulk(
                            self.theory_kind,
                            self.theory_index,
                            self.lemma_config.lemma_name,
                            [state.encoded_methods[i] for i in method_index],
                            state.encoded_sys,
                            timeout=(
                                self.timeout
                                if repeat < MAX_REPEATS - 1
                                else self.backup_timeout
                            ),
                        )
                    )
                    return responses, []
                else:
                    print(
                        f"Env: Final retry for bulk connection with method indices {method_index} for lemma {self.lemma_config.lemma_name} after {repeat} attempts. Trying to apply proof methods sequentially to identify failing methods..."
                    )
                    responses = []
                    failures = []
                    for method_idx in method_index:
                        try:
                            response = self.client.apply_proofmethod_to_system(
                                self.theory_kind,
                                self.theory_index,
                                self.lemma_config.lemma_name,
                                state.encoded_methods[method_idx],
                                state.encoded_sys,
                                timeout=self.backup_timeout,
                            )
                            print(
                                f"Env {self.id}: Successfully applied method index {method_idx} in sequential retry for lemma {self.lemma_config.lemma_name}"
                            )
                            responses.append(response)
                        except TamarinCallTimeoutError:
                            print(
                                f"Env {self.id}: Final timeout while applying proof methods, lemma name {self.lemma_config.lemma_name}, port {self.client.port}. Register failure."
                            )
                            print(
                                f"theory_kind: {self.theory_kind},\ntheory_index: {self.theory_index},\nlemma_name: {self.lemma_config.lemma_name},\nmethod_index: {method_index}"
                            )
                            failures.append(method_idx)
                            # Check memory consumption and reset if over 50GB
                            try:
                                process = psutil.Process(self.client.process.pid)
                                mem_usage_gb = process.memory_info().rss / (1024**3)
                                if mem_usage_gb > 50:
                                    print(
                                        f"WARNING: Env {self.id}: Memory usage {mem_usage_gb:.2f}GB exceeded 50GB. Resetting environment."
                                    )
                                    self.reset()
                            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                                print(
                                    f"WARNING: Env {self.id}: Failed to check memory usage: {e}. Resetting environment to be safe."
                                )
                                self.reset()

                    print(
                        f"Env {self.id}: Sequential retry completed for method indices {method_index} for lemma {self.lemma_config.lemma_name}. Resetting environment."
                    )
                    self.reset()
                    return responses, failures
            except TamarinCallTimeoutError:
                print(
                    f"Env {self.id}: {repeat}. timeout while applying proof methods, lemma name {self.lemma_config.lemma_name}, port {self.client.port}. Retrying with bigger timeout..."
                )
                # Check memory consumption and reset if over 50GB
                try:
                    process = psutil.Process(self.client.process.pid)
                    mem_usage_gb = process.memory_info().rss / (1024**3)
                    if mem_usage_gb > 50:
                        print(
                            f"WARNING: Env {self.id}: Memory usage {mem_usage_gb:.2f}GB exceeded 50GB. Resetting environment."
                        )
                        self.reset()
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    print(
                        f"WARNING: Env {self.id}: Failed to check memory usage: {e}. Resetting environment to be safe."
                    )
                    self.reset()
        return [], []

    def check_proof(self, proof: dict, size: int, status: str, timeout: int) -> str:
        """Checks whether a given proof is correct."""
        assert self.client is not None
        assert self.theory_kind is not None
        assert self.lemma_config is not None

        response = self.client.check_proof(
            self.theory_kind,
            self.theory_index,
            self.lemma_config.lemma_name,
            json.dumps(proof),
            timeout=timeout,
        )

        if not response.ok:
            print(
                f"WARNING: Env {self.id}: Failed to check proof for theory kind {self.theory_kind}, response not ok."
            )
            print(response)
            print(response.text)
            print(proof)
            return "FAILED"

        result = json.loads(response.text)

        if result["proofSize"] != size:
            print(
                f"WARNING: Env {self.id}: Proof size {result['proofSize']} does not match expected size {size}."
            )
            print(f"Proof: {proof}")
            print(f"Status: {status}")
            print(f"Result: {result}")
            print(
                f"theory_kind: {self.theory_kind},\ntheory_index: {self.theory_index},\nlemma_name: {self.lemma_config.lemma_name}"
            )
        if result["proofStatus"] != status:
            print(
                f"WARNING: Env {self.id}: Proof status  is {result['proofStatus']}, expected '{status}'."
            )
            raise RuntimeError(
                f"Proof status {result['proofStatus']} does not match expected status {status}, env {self.id}, theory kind {self.theory_kind}, theory index {self.theory_index}, lemma name {self.lemma_config.lemma_name}."
            )
        return result["proofFile"]

    def get_initial_state(self, timeout: int) -> State:
        assert self.client is not None
        assert self.lemma_config is not None
        if self.initial_state is not None:
            return self.initial_state

        r = self.client.get_initial_constraint_system(
            self.theory_kind,
            self.theory_index,
            self.lemma_config.lemma_name,
            timeout=timeout,
        )
        r = json.loads(r.text)
        self.initial_state = State(
            proof_methods=r[PROOF_METHODS],
            encoded_methods=tuple(r[ENCODED_METHODS]),
            encoded_sys=r[ENCODED_SYS],
        )
        return self.initial_state

    def close_client(self) -> None:
        if self.client:
            del self.client
            self.client = None
