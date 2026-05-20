from collections import OrderedDict
from typing import Any, Callable, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(OrderedDict[K, V]):
    """
    A dictionary-like container with a fixed capacity that evicts the
    least recently used (LRU) items when it's full.
    """

    def __init__(self, capacity: int):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self.capacity = capacity
        super().__init__()
        self.hits = 0
        self.misses = 0

    def get(self, key: K, default: Any = None) -> V | Any:
        if self.check(key):
            return self.__getitem__(key)

        return default

    def check(self, key: K) -> bool:
        if key in self:
            self.hits += 1
            return True

        self.misses += 1
        return False

    def __getitem__(self, key: K) -> V:
        # Retrieve item and mark it as recently used by moving it to the end
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key: K, value: V) -> None:
        # If key exists, move it to the end before updating
        if key in self:
            self.move_to_end(key)

        super().__setitem__(key, value)

        # If capacity is exceeded, remove the oldest item (from the front)
        if len(self) > self.capacity:
            self.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        """Returns the hit rate of the cache (hits / (hits + misses))."""
        total_accesses = self.hits + self.misses
        if total_accesses == 0:
            return 0.0
        return self.hits / total_accesses

    @property
    def usage(self) -> float:
        """Returns the current usage of the cache as a percentage (0.0 to 1.0)."""
        return len(self) / self.capacity


class MemoryLRUCache(LRUCache[K, V]):
    """LRU cache that evicts by estimated memory usage instead of entry count.

    ``size_of`` is called on each value to estimate its byte size.
    ``capacity`` on the parent is set to a large sentinel so count-based
    eviction never fires; only the memory limit is the active constraint.
    """

    def __init__(self, max_size_bytes: int, size_of: Callable[[V], int]):
        super().__init__(capacity=10_000_000)
        self.max_size_bytes = max_size_bytes
        self._size_of = size_of
        self._total_bytes: int = 0

    def __setitem__(self, key: K, value: V) -> None:
        if key in self:
            # Bypass LRUCache.__getitem__ to avoid incrementing self.hits on an overwrite.
            self._total_bytes -= self._size_of(OrderedDict.__getitem__(self, key))
        super().__setitem__(key, value)
        self._total_bytes += self._size_of(value)
        while self._total_bytes > self.max_size_bytes and len(self) > 1:
            _, evicted = self.popitem(last=False)
            self._total_bytes -= self._size_of(evicted)

    @property
    def usage(self) -> float:
        """Returns memory usage as a fraction (0.0–1.0) of max_size_bytes."""
        if self.max_size_bytes == 0:
            return 0.0
        return self._total_bytes / self.max_size_bytes
