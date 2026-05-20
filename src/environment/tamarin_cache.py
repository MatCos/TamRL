import sys
import threading
import traceback
from multiprocessing import Queue
from multiprocessing.connection import Connection, wait
from typing import Any

from src.environment.environment import State, TamarinEnvironment
from src.parser.lemma import LemmaConfig
from src.utils.cache import MemoryLRUCache


def _cache_value_size(
    value: tuple[dict[int, dict[str, TamarinEnvironment.ResponseState]], set[int], float],
) -> int:
    """Estimate byte size of a cache value from its dominant string fields."""
    responses, failed_set, _ = value
    size = sys.getsizeof(failed_set)
    for cases in responses.values():
        for rs in cases.values():
            size += sys.getsizeof(rs.encoded_sys)
            for s in rs.encoded_methods:
                size += sys.getsizeof(s)
    return size


class TamarinCache:

    def __init__(
        self,
        admin_conn: "Connection[tuple[str, int, Connection, Connection] | str]",
        fill_queue: "Queue[tuple[LemmaConfig, State, list[int], list[dict[str, TamarinEnvironment.ResponseState]], list[int], float]]",
        max_memory_mb: float = 100000.0,
    ) -> None:
        self.cache: MemoryLRUCache[
            tuple[LemmaConfig, State],
            tuple[dict[int, dict[str, TamarinEnvironment.ResponseState]], set[int], float],
        ] = MemoryLRUCache(
            max_size_bytes=int(max_memory_mb * 1024 * 1024),
            size_of=_cache_value_size,
        )
        self.admin_conn = admin_conn
        self.request_conns: dict[
            int, "Connection[tuple[str, LemmaConfig, State, list[int]]]"
        ] = {}
        self.response_conns: dict[
            int,
            "Connection[tuple[str, list[dict[str, TamarinEnvironment.ResponseState]] | None, float, float, float, list[int], list[int]]]",
        ] = {}
        assert len(self.request_conns) == len(self.response_conns)
        self.fill_queue = fill_queue
        self.lock = threading.Lock()

    def _consume_queue(self) -> None:
        """Consumes items from the fill_queue in a separate thread."""
        while True:
            try:
                (lemma_config, state, method_indices, responses, failures, call_time) = (
                    self.fill_queue.get()
                )

                to_add = {}
                resp_iter = iter(responses)
                for i in method_indices:
                    if i in failures:
                        continue
                    to_add[i] = next(resp_iter)

                remaining = list(resp_iter)
                assert remaining == [], f"Not all responses were used: {remaining}"

                with self.lock:
                    existing = self.cache.get((lemma_config, state))
                    if existing is not None:
                        merged_pfms = {**existing[0], **to_add}
                        merged_failures = existing[1] | set(failures)
                        self.cache[(lemma_config, state)] = (
                            merged_pfms,
                            merged_failures,
                            max(existing[2], call_time),
                        )
                    else:
                        self.cache[(lemma_config, state)] = (to_add, set(failures), call_time)
            except (OSError, EOFError):
                break

    def __call__(self) -> Any:
        threading.Thread(target=self._consume_queue, daemon=True).start()

        while True:
            readers = [self.admin_conn] + list(self.request_conns.values())

            ready = wait(readers)

            if self.admin_conn in ready:
                try:
                    command = self.admin_conn.recv()
                except EOFError:
                    break

                if isinstance(command, tuple) and command[0] == "UPDATE":
                    _, idx, req_conn, res_conn = command
                    try:
                        self.request_conns[idx].close()
                        self.response_conns[idx].close()
                    except Exception as e:
                        pass
                    self.request_conns[idx] = req_conn
                    self.response_conns[idx] = res_conn
                    print(f"Cache worker: Updated pipes for idx {idx}.", flush=True)
                elif isinstance(command, str) and command == "EXIT":
                    break

            for i, (req_conn, res_conn) in {
                idx: (req_conn, self.response_conns[idx])
                for idx, req_conn in self.request_conns.items()
            }.items():
                if req_conn in ready:
                    try:
                        (request_id, lemma_config, state, idx) = req_conn.recv()
                    except (EOFError, OSError) as e:
                        # Only close if the conn hasn't been replaced by an UPDATE
                        # processed earlier in this same wait() cycle.
                        if self.request_conns.get(i) is req_conn:
                            try:
                                self.request_conns[i].close()
                                self.response_conns[i].close()
                            except Exception:
                                pass
                            del self.request_conns[i]
                            del self.response_conns[i]
                        continue
                    except Exception as e:
                        print(f"WARNING: Cache: Unknown cache worker pipe error: {e}")
                        traceback.print_exc()
                        continue

                    try:
                        response_payload: tuple[
                            str,
                            list[dict[str, TamarinEnvironment.ResponseState]] | None,
                            float,
                            float,
                            float,
                            list[int],
                            list[int],
                        ]
                        with self.lock:
                            cache_hit = self.cache.get((lemma_config, state))
                            if cache_hit is None:
                                response_payload = (
                                    request_id,
                                    None,
                                    0.0,
                                    self.cache.hit_rate,
                                    self.cache.usage,
                                    [],
                                    [],
                                )
                            else:
                                pfms, failed_set, call_time = cache_hit

                                cached_failures = [i for i in idx if i in failed_set]
                                uncached = [i for i in idx if i not in pfms and i not in failed_set]
                                responses = [
                                    pfms[i] for i in idx if i in pfms
                                ]

                                response_payload = (
                                    request_id,
                                    responses,
                                    call_time,
                                    self.cache.hit_rate,
                                    self.cache.usage,
                                    cached_failures,
                                    uncached,
                                )

                        res_conn.send(response_payload)
                    except (BrokenPipeError, EOFError, OSError) as e:
                        print(
                            f"WARNING: Cache worker pipe error: {e}. Skipping",
                            flush=True,
                        )
                        traceback.print_exc()
                        continue
                    except Exception as e:
                        print(
                            f"WARNING: Cache: Unknown cache worker error: {e}. Skipping."
                        )
                        traceback.print_exc()
                        continue
