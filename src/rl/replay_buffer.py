import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

from src.environment.environment import State


@dataclass(frozen=True)
class TrainingExample(ABC):

    @property
    @abstractmethod
    def num_proof_methods(self) -> int:
        raise NotImplementedError()


@dataclass(frozen=True)
class ACTrainingExample(TrainingExample):
    state: State
    action: int
    value: float

    @property
    def num_proof_methods(self) -> int:
        return len(self.state.proof_methods)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, ACTrainingExample):
            return False
        return (
            self.state == value.state
            and self.action == value.action
            and self.value == value.value
        )

    def __hash__(self) -> int:
        return hash(self.state) ^ hash(self.action) ^ hash(self.value)

    def __repr__(self) -> str:
        return f"ACTrainingExample(state={self.state}, action={self.action}, value={self.value})"


T = TypeVar("T", bound=TrainingExample)


@dataclass(frozen=True)
class Batch(Generic[T]):

    items: tuple[T, ...]

    @classmethod
    def from_items(cls, items: list[T]) -> "Batch[T]":
        return cls(tuple(items))

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def to_list(self) -> list[T]:
        return list(self.items)

    def slice_batch(
        self,
        max_underlying_batch_size: int,
    ) -> list["Batch[T]"]:

        if (
            sum(item.num_proof_methods for item in self.items)
            <= max_underlying_batch_size
        ):
            return [self]  # early return if whole batch fits

        batches: list["Batch[T]"] = []
        current_batch: list[T] = []
        current_sum_of_pfms = 0
        for item in self.items:
            if item.num_proof_methods > max_underlying_batch_size:
                # oversized single item, put in its own batch
                batches.append(Batch.from_items([item]))
                continue

            if current_sum_of_pfms + item.num_proof_methods > max_underlying_batch_size:
                batches.append(Batch.from_items(current_batch))
                current_batch = [item]
                current_sum_of_pfms = item.num_proof_methods
            else:
                current_batch.append(item)
                current_sum_of_pfms += item.num_proof_methods

        if len(current_batch) > 0:
            batches.append(Batch.from_items(current_batch))

        return batches


class ReplayBuffer(Generic[T]):
    """A generic experience store for storing and sampling experiences."""

    def __init__(self, capacity: int, test_set_size: int = 0) -> None:
        """Initializes the ExperienceStore.

        Args:
            capacity (int): The maximum number of experiences to store.
            test_set_size (int): Number of first samples to keep as a fixed
                test set for tracking loss over time. 0 to disable.
        """
        self.memory: deque[T] = deque([], maxlen=capacity)
        self.usage_counts: deque[int] = deque([], maxlen=capacity)
        self.capacity = capacity
        self.fresh_samples = 0
        self.last_sample_mean_usage: float = 0.0
        self._usage_total: int = 0
        self._min_usage: int = 0
        # usage_count -> number of items with that count
        self._count_histogram: dict[int, int] = {}
        # Fixed test set: the first test_set_size samples that enter the buffer
        self.test_set: list[T] = []
        self.test_set_size = test_set_size

    @property
    def filled_ratio(self) -> float:
        """Returns the ratio of filled capacity.

        Returns:
            float: The ratio of filled capacity.
        """
        return len(self.memory) / self.capacity

    @property
    def mean_usage_count(self) -> float:
        """Returns the mean usage count across all items in the buffer."""
        if len(self.memory) == 0:
            return 0.0
        return self._usage_total / len(self.memory)

    @property
    def min_usage_count(self) -> int:
        """Returns the minimum usage count across all items in the buffer."""
        if len(self.memory) == 0:
            return 0
        return self._min_usage

    def _evict_count(self, count: int) -> None:
        self._usage_total -= count
        self._count_histogram[count] -= 1
        if self._count_histogram[count] == 0:
            del self._count_histogram[count]

    def _add_count(self, count: int) -> None:
        self._usage_total += count
        self._count_histogram[count] = (
            self._count_histogram.get(count, 0) + 1
        )

    def push(self, item: T) -> None:
        """Saves a single experience.

        Args:
            item: The experience to save.
        """
        if len(self.test_set) < self.test_set_size:
            self.test_set.append(item)
        if len(self.memory) == self.capacity:
            self._evict_count(self.usage_counts[0])
        self.memory.append(item)
        self.usage_counts.append(0)
        self._add_count(0)
        self._min_usage = 0
        self.fresh_samples += 1

    def extend(self, items: list[T]) -> None:
        """Saves multiple experiences.

        Args:
            items (list[T]): The experiences to save.
        """
        remaining = self.test_set_size - len(self.test_set)
        if remaining > 0:
            self.test_set.extend(items[:remaining])
        overflow = max(0, len(self.memory) + len(items) - self.capacity)
        for j in range(min(overflow, len(self.memory))):
            self._evict_count(self.usage_counts[j])
        self.memory.extend(items)
        self.usage_counts.extend([0] * len(items))
        num_added = min(len(items), self.capacity)
        self._count_histogram[0] = (
            self._count_histogram.get(0, 0) + num_added
        )
        self._min_usage = 0
        self.fresh_samples += len(items)

    def sample(self, batch_size: int) -> Batch[T]:
        """Randomly samples a batch of experiences, weighted by
        inverse usage count: weight = 1 / (1 + usage_count).

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Batch[T]: A batch of randomly selected experiences.
        """
        self.fresh_samples = 0
        # Weighted sampling without replacement
        weights = [
            1.0 / (1 + count) for count in self.usage_counts
        ]
        indices: list[int] = []
        for _ in range(batch_size):
            [idx] = random.choices(
                range(len(self.memory)), weights=weights, k=1
            )
            indices.append(idx)
            weights[idx] = 0.0
        items = [self.memory[i] for i in indices]
        usage_sum = 0
        for i in indices:
            old_count = self.usage_counts[i]
            new_count = old_count + 1
            self._evict_count(old_count)
            self._add_count(new_count)
            self.usage_counts[i] = new_count
            usage_sum += new_count
        # min can only increase; scan forward
        while self._min_usage not in self._count_histogram:
            self._min_usage += 1
        self.last_sample_mean_usage = usage_sum / batch_size
        return Batch.from_items(items)

    def __len__(self) -> int:
        """Returns the current number of experiences.

        Returns:
            int: The current size of the store.
        """
        return len(self.memory)
