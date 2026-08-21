"""Deterministic, framework-independent replay memory primitives.

The sampled Deep CFR trainer uses reservoir sampling so every observation in a
stage has an equal probability of remaining in memory.  This module purposely
does not import PyTorch: its state dictionary is a plain, pickle-compatible
mapping and can hold arbitrary Python objects, including dataclass samples.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
import random
from typing import Any, Generic, TypeVar


T = TypeVar("T")

_STATE_VERSION = 1


@dataclass(frozen=True, slots=True)
class ReservoirUpdate(Generic[T]):
    """Describe the effect of one :meth:`ReservoirBuffer.append` call.

    ``retained`` is false when Algorithm R elects not to keep the new item.
    ``evicted`` is populated when a retained item replaces an older sample.
    Callers that do not need lifetime bookkeeping may simply ignore this
    return value.
    """

    retained: bool
    index: int | None
    evicted: T | None = None

    def __bool__(self) -> bool:
        return self.retained


class ReservoirBuffer(Generic[T]):
    """A fixed-capacity, seeded Algorithm-R reservoir.

    Appending item number ``n`` (one based) retains it with probability
    ``capacity / n``.  Both replacement decisions and calls to :meth:`sample`
    consume the buffer's private RNG, whose complete state is included in
    :meth:`state_dict` for exact checkpoint/resume behavior.
    """

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._items: list[T] = []
        self._seen_count = 0
        self._rng = random.Random(seed)

    @property
    def capacity(self) -> int:
        """Maximum number of retained samples."""

        return self._capacity

    @property
    def seen_count(self) -> int:
        """Total number of samples offered to the reservoir."""

        return self._seen_count

    @property
    def items(self) -> tuple[T, ...]:
        """A read-only view of retained samples in reservoir-slot order."""

        return tuple(self._items)

    @property
    def values(self) -> tuple[T, ...]:
        """Alias for :attr:`items`, useful to generic replay-memory clients."""

        return self.items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def append(self, item: T) -> ReservoirUpdate[T]:
        """Offer ``item`` to the reservoir and report retention/replacement."""

        self._seen_count += 1
        if len(self._items) < self._capacity:
            index = len(self._items)
            self._items.append(item)
            return ReservoirUpdate(retained=True, index=index)

        index = self._rng.randrange(self._seen_count)
        if index >= self._capacity:
            return ReservoirUpdate(retained=False, index=None)

        evicted = self._items[index]
        self._items[index] = item
        return ReservoirUpdate(retained=True, index=index, evicted=evicted)

    def extend(self, items: Iterable[T]) -> list[ReservoirUpdate[T]]:
        """Offer several items in order, returning each insertion result."""

        return [self.append(item) for item in items]

    def sample(self, requested: int) -> list[T]:
        """Sample without replacement, capped at the number currently stored.

        Requesting more samples than are retained returns every retained item
        in a deterministic RNG-selected order.  A zero request returns an
        empty list.
        """

        if isinstance(requested, bool) or not isinstance(requested, int):
            raise TypeError("requested sample count must be an integer")
        if requested < 0:
            raise ValueError("requested sample count cannot be negative")
        return self._rng.sample(self._items, min(requested, len(self._items)))

    def state_dict(self) -> dict[str, Any]:
        """Return all data and RNG state required for an exact resume."""

        return {
            "version": _STATE_VERSION,
            "capacity": self._capacity,
            "seen_count": self._seen_count,
            "items": list(self._items),
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict`.

        Capacity is intentionally configuration-owned and must match the
        receiving buffer.  Validation is completed before this instance is
        mutated.
        """

        if not isinstance(state, Mapping):
            raise TypeError("reservoir state must be a mapping")
        version = state.get("version")
        if version != _STATE_VERSION:
            raise ValueError(f"unsupported reservoir state version: {version!r}")

        capacity = state.get("capacity")
        if capacity != self._capacity:
            raise ValueError(
                "reservoir capacity mismatch: "
                f"checkpoint has {capacity!r}, buffer has {self._capacity}"
            )

        seen_count = state.get("seen_count")
        if isinstance(seen_count, bool) or not isinstance(seen_count, int):
            raise ValueError("reservoir seen_count must be an integer")
        if seen_count < 0:
            raise ValueError("reservoir seen_count cannot be negative")

        raw_items = state.get("items")
        if not isinstance(raw_items, (list, tuple)):
            raise ValueError("reservoir items must be a list or tuple")
        items = list(raw_items)
        expected_size = min(seen_count, self._capacity)
        if len(items) != expected_size:
            raise ValueError(
                "reservoir item count is inconsistent with capacity and seen_count"
            )

        if "rng_state" not in state:
            raise ValueError("reservoir state is missing rng_state")
        restored_rng = random.Random()
        try:
            restored_rng.setstate(state["rng_state"])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid reservoir RNG state") from error

        self._items = items
        self._seen_count = seen_count
        self._rng = restored_rng

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ReservoirBuffer[Any]":
        """Construct a reservoir directly from serialized state."""

        if not isinstance(state, Mapping):
            raise TypeError("reservoir state must be a mapping")
        capacity = state.get("capacity")
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise ValueError("reservoir state has no valid capacity")
        buffer: ReservoirBuffer[Any] = cls(capacity)
        buffer.load_state_dict(state)
        return buffer


__all__ = ["ReservoirBuffer", "ReservoirUpdate"]
