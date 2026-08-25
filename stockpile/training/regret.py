"""Signed outcome-sampling regret telemetry and offline analysis.

This module is deliberately independent from the trainer and from PyTorch.  A
trainer may capture the signed targets it already computed, finish each whole
traversal, and atomically commit one compressed sidecar after an outer
iteration completes.  Capture, persistence, and analysis use their own local
state and never consume a learning RNG.

The stage statistic is sampled average regret.  For each outer iteration and
player, the signed contribution of each exact
perfect-recall information set/action is averaged across that stratum's N
complete traversals.  Those iteration estimates are accumulated with their
sign intact.  Only then is the best positive action selected at each
information set, summed, and divided by the number of complete outer stage
iterations.  The bootstrap resamples exactly N complete traversals with
replacement inside each fixed ``(iteration, player)`` stratum; neither outer
iterations nor individual observations are resampled.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .encoding import ACTION_COUNT, ENCODING_SCHEMA_VERSION


REGRET_SIDECAR_KIND = "stockpile_sampled_signed_regret_iteration"
REGRET_SIDECAR_SCHEMA_VERSION = 1
REGRET_ARCHIVE_KIND = "stockpile_sampled_signed_regret_archive"
REGRET_ARCHIVE_SCHEMA_VERSION = 1
REGRET_REPORT_KIND = "stockpile_sampled_average_regret_analysis"
REGRET_REPORT_SCHEMA_VERSION = 1

DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_CONFIDENCE = 0.90
DEFAULT_BOOTSTRAP_SEED = 0

_INTEGRITY_ALGORITHM = "sha256"
_INTEGRITY_ENCODING = "canonical_numpy_payload_v1"
_REPORT_FILENAME = "sampled_average_regret.json"
_MAX_MASK_ACTIONS = 32
_BOOTSTRAP_CHUNK_SIZE = 256
_HEX_DIGITS = frozenset("0123456789abcdef")

_TRAVERSAL_PLAYER = "traversal_player"
_TRAVERSAL_ORDINAL = "traversal_ordinal"
_TRAVERSAL_OBSERVATION_OFFSETS = "traversal_observation_offsets"
_PERFECT_RECALL_ID = "perfect_recall_id"
_LEGAL_MASK_BITS = "legal_mask_bits"
_OBSERVATION_ACTION_OFFSETS = "observation_action_offsets"
_ACTION_IDS = "action_ids"
_SIGNED_TARGETS = "signed_targets"
_METADATA_JSON = "metadata_json"

_PAYLOAD_NAMES = (
    _TRAVERSAL_PLAYER,
    _TRAVERSAL_ORDINAL,
    _TRAVERSAL_OBSERVATION_OFFSETS,
    _PERFECT_RECALL_ID,
    _LEGAL_MASK_BITS,
    _OBSERVATION_ACTION_OFFSETS,
    _ACTION_IDS,
    _SIGNED_TARGETS,
)
_ALL_ARRAY_NAMES = frozenset((*_PAYLOAD_NAMES, _METADATA_JSON))

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RegretAnalysisProgress:
    """One deterministic progress observation from offline regret analysis."""

    stage_index: int
    round_count: int
    stage_number: int
    stage_count: int
    completed_replicates: int
    total_replicates: int


class RegretFormatError(ValueError):
    """Raised when a regret record, sidecar, or report is incompatible."""


class RegretArchiveError(RuntimeError):
    """Raised when atomic archive commit or checkpoint restore cannot proceed."""


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise RegretFormatError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise RegretFormatError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise RegretFormatError(f"{name} must be at most {maximum}")
    return result


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise RegretFormatError(f"{name} must be a finite float") from error
    if not math.isfinite(result):
        raise RegretFormatError(f"{name} must be finite")
    return result


def _action_count(value: Any) -> int:
    return _integer(
        value,
        "action_count",
        minimum=1,
        maximum=_MAX_MASK_ACTIONS,
    )


def _perfect_recall_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise RegretFormatError(
            "perfect_recall_id must be an exact lowercase SHA-256 hex digest"
        )
    return value


def _mask_bits(legal_mask: ArrayLike, action_count: int) -> tuple[int, tuple[int, ...]]:
    raw = np.asarray(legal_mask)
    if raw.ndim != 1 or raw.shape != (action_count,):
        raise RegretFormatError(
            f"legal_mask must have shape ({action_count},)"
        )
    if raw.dtype != np.bool_:
        try:
            numeric = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise RegretFormatError(
                "legal_mask must contain booleans or zero/one values"
            ) from error
        if not np.all(np.isfinite(numeric)) or not np.all(
            (numeric == 0.0) | (numeric == 1.0)
        ):
            raise RegretFormatError(
                "legal_mask must contain booleans or zero/one values"
            )
    mask = raw.astype(np.bool_, copy=False)
    action_ids = tuple(int(value) for value in np.flatnonzero(mask))
    if not action_ids:
        raise RegretFormatError("legal_mask must contain at least one legal action")
    bits = 0
    for action_id in action_ids:
        bits |= 1 << action_id
    return bits, action_ids


def _ids_from_mask_bits(bits: Any, action_count: int) -> tuple[int, ...]:
    result = _integer(
        bits,
        "legal_mask_bits",
        minimum=1,
        maximum=(1 << action_count) - 1,
    )
    return tuple(
        action_id
        for action_id in range(action_count)
        if result & (1 << action_id)
    )


@dataclass(frozen=True, slots=True)
class RegretTargetObservation:
    """One sparse signed target at an exact perfect-recall information set."""

    perfect_recall_id: str
    legal_mask_bits: int
    action_ids: tuple[int, ...]
    target_values: tuple[np.float64, ...]
    action_count: int = ACTION_COUNT

    def __post_init__(self) -> None:
        count = _action_count(self.action_count)
        identifier = _perfect_recall_id(self.perfect_recall_id)
        bits = _integer(
            self.legal_mask_bits,
            "legal_mask_bits",
            minimum=1,
            maximum=(1 << count) - 1,
        )
        expected_ids = _ids_from_mask_bits(bits, count)
        try:
            action_ids = tuple(int(value) for value in self.action_ids)
        except (TypeError, ValueError, OverflowError) as error:
            raise RegretFormatError("action_ids must contain integers") from error
        if action_ids != expected_ids:
            raise RegretFormatError(
                "action_ids must be the sorted actions encoded by legal_mask_bits"
            )
        try:
            target_values = tuple(np.float64(value) for value in self.target_values)
        except (TypeError, ValueError, OverflowError) as error:
            raise RegretFormatError("target_values must be numeric") from error
        if len(target_values) != len(action_ids):
            raise RegretFormatError(
                "target_values must contain one value per legal action"
            )
        if not all(np.isfinite(value) for value in target_values):
            raise RegretFormatError("target_values must all be finite")

        object.__setattr__(self, "action_count", count)
        object.__setattr__(self, "perfect_recall_id", identifier)
        object.__setattr__(self, "legal_mask_bits", bits)
        object.__setattr__(self, "action_ids", action_ids)
        object.__setattr__(self, "target_values", target_values)

    @classmethod
    def from_dense(
        cls,
        *,
        perfect_recall_id: str,
        legal_mask: ArrayLike,
        target: ArrayLike,
        action_count: int = ACTION_COUNT,
    ) -> "RegretTargetObservation":
        """Copy the legal signed entries from a full action-indexed target."""

        count = _action_count(action_count)
        bits, action_ids = _mask_bits(legal_mask, count)
        try:
            values = np.asarray(target, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise RegretFormatError("target must be a numeric vector") from error
        if values.ndim != 1 or values.shape != (count,):
            raise RegretFormatError(f"target must have shape ({count},)")
        if not np.all(np.isfinite(values)):
            raise RegretFormatError("target must contain only finite values")
        illegal = np.ones(count, dtype=np.bool_)
        illegal[np.asarray(action_ids, dtype=np.intp)] = False
        if np.any(values[illegal] != 0.0):
            raise RegretFormatError("target must be zero on illegal actions")
        return cls(
            perfect_recall_id=perfect_recall_id,
            legal_mask_bits=bits,
            action_ids=action_ids,
            target_values=tuple(np.float64(values[action_id]) for action_id in action_ids),
            action_count=count,
        )

    @property
    def legal_mask(self) -> tuple[bool, ...]:
        return tuple(
            bool(self.legal_mask_bits & (1 << action_id))
            for action_id in range(self.action_count)
        )

    def dense_target(self) -> FloatArray:
        result = np.zeros(self.action_count, dtype=np.float64)
        result[np.asarray(self.action_ids, dtype=np.intp)] = np.asarray(
            self.target_values,
            dtype=np.float64,
        )
        return result


@dataclass(frozen=True, slots=True)
class TraversalRegretRecord:
    """All update-player regret observations from one completed traversal."""

    player_id: int
    traversal_ordinal: int
    observations: tuple[RegretTargetObservation, ...]

    def __post_init__(self) -> None:
        player = _integer(self.player_id, "player_id", minimum=0, maximum=1)
        ordinal = _integer(
            self.traversal_ordinal,
            "traversal_ordinal",
            minimum=0,
        )
        observations = tuple(self.observations)
        if not all(
            isinstance(value, RegretTargetObservation) for value in observations
        ):
            raise RegretFormatError(
                "traversal observations must be RegretTargetObservation values"
            )
        object.__setattr__(self, "player_id", player)
        object.__setattr__(self, "traversal_ordinal", ordinal)
        object.__setattr__(self, "observations", observations)


@dataclass(frozen=True, slots=True)
class IterationRegretRecord:
    """Both players' complete traversal strata for one outer iteration."""

    stage_index: int
    round_count: int
    stage_iteration: int
    global_iteration: int
    traversals: tuple[TraversalRegretRecord, ...]
    encoder_schema_version: str = ENCODING_SCHEMA_VERSION
    action_count: int = ACTION_COUNT

    def __post_init__(self) -> None:
        stage_index = _integer(self.stage_index, "stage_index", minimum=0)
        round_count = _integer(self.round_count, "round_count", minimum=1)
        stage_iteration = _integer(
            self.stage_iteration,
            "stage_iteration",
            minimum=1,
        )
        global_iteration = _integer(
            self.global_iteration,
            "global_iteration",
            minimum=1,
        )
        count = _action_count(self.action_count)
        if not isinstance(self.encoder_schema_version, str) or not (
            self.encoder_schema_version
        ):
            raise RegretFormatError("encoder_schema_version must be nonempty")
        raw_traversals = tuple(self.traversals)
        if not raw_traversals or not all(
            isinstance(value, TraversalRegretRecord) for value in raw_traversals
        ):
            raise RegretFormatError(
                "an iteration must contain completed traversal records"
            )
        traversals = tuple(
            sorted(
                raw_traversals,
                key=lambda value: (value.player_id, value.traversal_ordinal),
            )
        )

        identifiers: dict[str, int] = {}
        for player_id in (0, 1):
            player_traversals = [
                value for value in traversals if value.player_id == player_id
            ]
            if not player_traversals:
                raise RegretFormatError(
                    f"iteration is missing player {player_id} traversals"
                )
            ordinals = tuple(value.traversal_ordinal for value in player_traversals)
            if ordinals != tuple(range(len(player_traversals))):
                raise RegretFormatError(
                    f"player {player_id} traversal ordinals must be contiguous from zero"
                )
            for traversal in player_traversals:
                for observation in traversal.observations:
                    if observation.action_count != count:
                        raise RegretFormatError(
                            "observation action_count does not match iteration"
                        )
                    previous = identifiers.setdefault(
                        observation.perfect_recall_id,
                        observation.legal_mask_bits,
                    )
                    if previous != observation.legal_mask_bits:
                        raise RegretFormatError(
                            "one perfect_recall_id has inconsistent legal masks"
                        )

        object.__setattr__(self, "stage_index", stage_index)
        object.__setattr__(self, "round_count", round_count)
        object.__setattr__(self, "stage_iteration", stage_iteration)
        object.__setattr__(self, "global_iteration", global_iteration)
        object.__setattr__(self, "action_count", count)
        object.__setattr__(self, "traversals", traversals)


class RegretTraversalCapture:
    """Mutable capture handle that seals into one immutable traversal record."""

    def __init__(self, player_id: int, traversal_ordinal: int) -> None:
        self._player_id = _integer(player_id, "player_id", minimum=0, maximum=1)
        self._traversal_ordinal = _integer(
            traversal_ordinal,
            "traversal_ordinal",
            minimum=0,
        )
        self._observations: list[RegretTargetObservation] = []
        self._finished: TraversalRegretRecord | None = None

    def add_target(
        self,
        *,
        perfect_recall_id: str,
        legal_mask: ArrayLike,
        target: ArrayLike,
    ) -> None:
        """Copy one already-computed target without consuming any RNG."""

        if self._finished is not None:
            raise RegretArchiveError("cannot add a target after traversal finish")
        count = int(np.asarray(legal_mask).size)
        self._observations.append(
            RegretTargetObservation.from_dense(
                perfect_recall_id=perfect_recall_id,
                legal_mask=legal_mask,
                target=target,
                action_count=count,
            )
        )

    def finish(self) -> TraversalRegretRecord:
        """Seal and return the completed traversal; repeated calls are idempotent.

        Repeated observations of the same perfect-recall ID are compacted by
        signed summation.  This is lossless for the cumulative statistic and
        keeps sidecars sparse; an ID whose legal mask changes is corrupt.
        """

        if self._finished is None:
            order: list[str] = []
            grouped: dict[str, list[RegretTargetObservation]] = defaultdict(list)
            masks: dict[str, int] = {}
            for observation in self._observations:
                identifier = observation.perfect_recall_id
                if identifier not in grouped:
                    order.append(identifier)
                previous = masks.setdefault(identifier, observation.legal_mask_bits)
                if previous != observation.legal_mask_bits:
                    raise RegretFormatError(
                        "one perfect_recall_id has inconsistent legal masks "
                        "within a traversal"
                    )
                grouped[identifier].append(observation)

            compacted: list[RegretTargetObservation] = []
            for identifier in order:
                observations = grouped[identifier]
                exemplar = observations[0]
                try:
                    totals = tuple(
                        np.float64(
                            math.fsum(
                                float(observation.target_values[index])
                                for observation in observations
                            )
                        )
                        for index in range(len(exemplar.action_ids))
                    )
                except OverflowError as error:
                    raise FloatingPointError(
                        "within-traversal signed regret sum overflowed"
                    ) from error
                if not all(np.isfinite(value) for value in totals):
                    raise FloatingPointError(
                        "within-traversal signed regret sum became nonfinite"
                    )
                compacted.append(
                    RegretTargetObservation(
                        perfect_recall_id=identifier,
                        legal_mask_bits=exemplar.legal_mask_bits,
                        action_ids=exemplar.action_ids,
                        target_values=totals,
                        action_count=exemplar.action_count,
                    )
                )
            self._finished = TraversalRegretRecord(
                player_id=self._player_id,
                traversal_ordinal=self._traversal_ordinal,
                observations=tuple(compacted),
            )
        assert self._finished is not None
        return self._finished


class RegretIterationCapture:
    """Collect completed traversal records for one atomic iteration sidecar."""

    def __init__(
        self,
        *,
        stage_index: int,
        round_count: int,
        stage_iteration: int,
        global_iteration: int,
        encoder_schema_version: str = ENCODING_SCHEMA_VERSION,
        action_count: int = ACTION_COUNT,
    ) -> None:
        self._metadata = {
            "stage_index": _integer(stage_index, "stage_index", minimum=0),
            "round_count": _integer(round_count, "round_count", minimum=1),
            "stage_iteration": _integer(
                stage_iteration,
                "stage_iteration",
                minimum=1,
            ),
            "global_iteration": _integer(
                global_iteration,
                "global_iteration",
                minimum=1,
            ),
            "encoder_schema_version": encoder_schema_version,
            "action_count": _action_count(action_count),
        }
        self._traversals: dict[tuple[int, int], TraversalRegretRecord] = {}
        self._finished: IterationRegretRecord | None = None

    def add_traversal(self, traversal: TraversalRegretRecord) -> None:
        if self._finished is not None:
            raise RegretArchiveError("cannot add a traversal after iteration finish")
        if not isinstance(traversal, TraversalRegretRecord):
            raise TypeError("traversal must be a TraversalRegretRecord")
        key = (traversal.player_id, traversal.traversal_ordinal)
        if key in self._traversals:
            raise RegretArchiveError(
                f"duplicate traversal player={key[0]} ordinal={key[1]}"
            )
        for observation in traversal.observations:
            if observation.action_count != self._metadata["action_count"]:
                raise RegretFormatError(
                    "traversal observation action_count does not match iteration"
                )
        self._traversals[key] = traversal

    def finish(self) -> IterationRegretRecord:
        """Seal an iteration only after both complete traversal strata exist."""

        if self._finished is None:
            self._finished = IterationRegretRecord(
                traversals=tuple(self._traversals.values()),
                **self._metadata,
            )
        assert self._finished is not None
        return self._finished


def regret_sidecar_path(
    run_dir: str | Path,
    *,
    stage_index: int,
    round_count: int,
    stage_iteration: int,
) -> Path:
    """Return the canonical path for a completed iteration sidecar."""

    _integer(stage_index, "stage_index", minimum=0)
    rounds = _integer(round_count, "round_count", minimum=1)
    iteration = _integer(stage_iteration, "stage_iteration", minimum=1)
    return (
        Path(run_dir)
        / f"round_{rounds:02d}"
        / "sampled_regret"
        / f"iteration_{iteration:06d}.npz"
    )


def analysis_report_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "analysis" / _REPORT_FILENAME


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest_chunk(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _canonical_digest(
    metadata_core: Mapping[str, Any],
    arrays: Mapping[str, NDArray[Any]],
) -> str:
    digest = hashlib.sha256()
    _digest_chunk(digest, _INTEGRITY_ENCODING.encode("ascii"))
    _digest_chunk(digest, _json_bytes(metadata_core))
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        _digest_chunk(digest, name.encode("ascii"))
        _digest_chunk(digest, array.dtype.str.encode("ascii"))
        _digest_chunk(
            digest,
            json.dumps(array.shape, separators=(",", ":")).encode("ascii"),
        )
        _digest_chunk(digest, array.tobytes(order="C"))
    return digest.hexdigest()


def _record_arrays(record: IterationRegretRecord) -> dict[str, NDArray[Any]]:
    players: list[int] = []
    ordinals: list[int] = []
    traversal_offsets = [0]
    identifiers: list[str] = []
    mask_bits: list[int] = []
    observation_offsets = [0]
    action_ids: list[int] = []
    targets: list[np.float64] = []

    for traversal in record.traversals:
        players.append(traversal.player_id)
        ordinals.append(traversal.traversal_ordinal)
        for observation in traversal.observations:
            identifiers.append(observation.perfect_recall_id)
            mask_bits.append(observation.legal_mask_bits)
            action_ids.extend(observation.action_ids)
            targets.extend(observation.target_values)
            observation_offsets.append(len(action_ids))
        traversal_offsets.append(len(identifiers))

    return {
        _TRAVERSAL_PLAYER: np.asarray(players, dtype=np.dtype("u1")),
        _TRAVERSAL_ORDINAL: np.asarray(ordinals, dtype=np.dtype("<i8")),
        _TRAVERSAL_OBSERVATION_OFFSETS: np.asarray(
            traversal_offsets,
            dtype=np.dtype("<i8"),
        ),
        _PERFECT_RECALL_ID: np.asarray(identifiers, dtype=np.dtype("<U64")),
        _LEGAL_MASK_BITS: np.asarray(mask_bits, dtype=np.dtype("<u4")),
        _OBSERVATION_ACTION_OFFSETS: np.asarray(
            observation_offsets,
            dtype=np.dtype("<i8"),
        ),
        _ACTION_IDS: np.asarray(action_ids, dtype=np.dtype("u1")),
        _SIGNED_TARGETS: np.asarray(targets, dtype=np.dtype("<f8")),
    }


def _metadata_core(
    record: IterationRegretRecord,
    arrays: Mapping[str, NDArray[Any]],
) -> dict[str, Any]:
    return {
        "kind": REGRET_SIDECAR_KIND,
        "schema_version": REGRET_SIDECAR_SCHEMA_VERSION,
        "encoder_schema_version": record.encoder_schema_version,
        "action_count": record.action_count,
        "stage_index": record.stage_index,
        "round_count": record.round_count,
        "stage_iteration": record.stage_iteration,
        "global_iteration": record.global_iteration,
        "traversal_count": int(arrays[_TRAVERSAL_PLAYER].size),
        "observation_count": int(arrays[_PERFECT_RECALL_ID].size),
        "sparse_target_count": int(arrays[_SIGNED_TARGETS].size),
    }


def _serialized_sidecar(record: IterationRegretRecord) -> bytes:
    arrays = _record_arrays(record)
    core = _metadata_core(record, arrays)
    metadata = {
        **core,
        "integrity": {
            "algorithm": _INTEGRITY_ALGORITHM,
            "encoding": _INTEGRITY_ENCODING,
            "payload_sha256": _canonical_digest(core, arrays),
        },
    }
    rendered = _json_bytes(metadata).decode("utf-8")
    archive_arrays = {
        **arrays,
        _METADATA_JSON: np.asarray(
            rendered,
            dtype=np.dtype(f"<U{max(1, len(rendered))}"),
        ),
    }
    stream = io.BytesIO()
    np.savez_compressed(stream, **archive_arrays)
    return stream.getvalue()


def _record_fingerprint(record: IterationRegretRecord) -> str:
    arrays = _record_arrays(record)
    return _canonical_digest(_metadata_core(record, arrays), arrays)


def _strict_json(text: str, name: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise RegretFormatError(f"{name} contains nonstandard constant {value}")

    try:
        result = json.loads(text, parse_constant=reject_constant)
    except (json.JSONDecodeError, TypeError) as error:
        raise RegretFormatError(f"{name} is not valid strict JSON") from error
    if not isinstance(result, Mapping):
        raise RegretFormatError(f"{name} must decode to an object")
    return result


def _read_npz(source: str | Path | io.BytesIO) -> dict[str, NDArray[Any]]:
    try:
        with np.load(source, allow_pickle=False) as archive:
            if frozenset(archive.files) != _ALL_ARRAY_NAMES:
                missing = sorted(_ALL_ARRAY_NAMES.difference(archive.files))
                unexpected = sorted(set(archive.files).difference(_ALL_ARRAY_NAMES))
                raise RegretFormatError(
                    "sidecar arrays do not match schema; "
                    f"missing={missing}, unexpected={unexpected}"
                )
            result = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
    except RegretFormatError:
        raise
    except (OSError, ValueError, KeyError, EOFError) as error:
        raise RegretFormatError("cannot read regret sidecar without pickle") from error
    if any(array.dtype.hasobject for array in result.values()):
        raise RegretFormatError("regret sidecars cannot contain object arrays")
    return result


def _metadata_from_arrays(
    archive_arrays: Mapping[str, NDArray[Any]],
) -> tuple[Mapping[str, Any], dict[str, NDArray[Any]]]:
    metadata_array = archive_arrays[_METADATA_JSON]
    if metadata_array.ndim != 0 or metadata_array.dtype.kind != "U":
        raise RegretFormatError("metadata_json must be one Unicode scalar")
    metadata_value = metadata_array.item()
    if not isinstance(metadata_value, str):
        raise RegretFormatError("metadata_json must contain text")
    metadata = _strict_json(metadata_value, "metadata_json")
    payload = {name: archive_arrays[name] for name in _PAYLOAD_NAMES}

    if metadata.get("kind") != REGRET_SIDECAR_KIND:
        raise RegretFormatError("not a Stockpile signed-regret sidecar")
    if metadata.get("schema_version") != REGRET_SIDECAR_SCHEMA_VERSION:
        raise RegretFormatError("unsupported signed-regret sidecar schema")
    integrity = metadata.get("integrity")
    if not isinstance(integrity, Mapping):
        raise RegretFormatError("sidecar metadata is missing integrity")
    if integrity.get("algorithm") != _INTEGRITY_ALGORITHM:
        raise RegretFormatError("unsupported sidecar integrity algorithm")
    if integrity.get("encoding") != _INTEGRITY_ENCODING:
        raise RegretFormatError("unsupported sidecar integrity encoding")
    supplied_digest = integrity.get("payload_sha256")
    if not isinstance(supplied_digest, str):
        raise RegretFormatError("sidecar integrity digest is missing")
    core = dict(metadata)
    core.pop("integrity")
    expected_digest = _canonical_digest(core, payload)
    if not hmac.compare_digest(supplied_digest, expected_digest):
        raise RegretFormatError("sidecar integrity digest does not match payload")
    return metadata, payload


def _expected_array(
    payload: Mapping[str, NDArray[Any]],
    name: str,
    dtype: str,
    length: int | None = None,
) -> NDArray[Any]:
    array = payload[name]
    if array.ndim != 1 or array.dtype != np.dtype(dtype):
        raise RegretFormatError(
            f"{name} must be a one-dimensional {dtype} array"
        )
    if length is not None and array.size != length:
        raise RegretFormatError(f"{name} has inconsistent length")
    return array


def _offsets(
    payload: Mapping[str, NDArray[Any]],
    name: str,
    group_count: int,
    item_count: int,
) -> NDArray[np.int64]:
    values = _expected_array(payload, name, "<i8", group_count + 1)
    if (
        values[0] != 0
        or values[-1] != item_count
        or np.any(values[1:] < values[:-1])
    ):
        raise RegretFormatError(f"{name} is not a valid offset array")
    return values


def _record_from_arrays(
    metadata: Mapping[str, Any],
    payload: Mapping[str, NDArray[Any]],
) -> IterationRegretRecord:
    count = _action_count(metadata.get("action_count"))
    traversal_count = _integer(
        metadata.get("traversal_count"),
        "traversal_count",
        minimum=1,
    )
    observation_count = _integer(
        metadata.get("observation_count"),
        "observation_count",
        minimum=0,
    )
    sparse_count = _integer(
        metadata.get("sparse_target_count"),
        "sparse_target_count",
        minimum=0,
    )

    players = _expected_array(
        payload,
        _TRAVERSAL_PLAYER,
        "u1",
        traversal_count,
    )
    if np.any(players > 1):
        raise RegretFormatError("traversal_player values must be 0 or 1")
    ordinals = _expected_array(
        payload,
        _TRAVERSAL_ORDINAL,
        "<i8",
        traversal_count,
    )
    if np.any(ordinals < 0):
        raise RegretFormatError("traversal ordinals cannot be negative")
    traversal_offsets = _offsets(
        payload,
        _TRAVERSAL_OBSERVATION_OFFSETS,
        traversal_count,
        observation_count,
    )
    identifiers = _expected_array(
        payload,
        _PERFECT_RECALL_ID,
        "<U64",
        observation_count,
    )
    masks = _expected_array(
        payload,
        _LEGAL_MASK_BITS,
        "<u4",
        observation_count,
    )
    observation_offsets = _offsets(
        payload,
        _OBSERVATION_ACTION_OFFSETS,
        observation_count,
        sparse_count,
    )
    action_ids = _expected_array(payload, _ACTION_IDS, "u1", sparse_count)
    signed_targets = _expected_array(
        payload,
        _SIGNED_TARGETS,
        "<f8",
        sparse_count,
    )
    if not np.all(np.isfinite(signed_targets)):
        raise RegretFormatError("signed_targets must all be finite")

    observations: list[RegretTargetObservation] = []
    for observation_index in range(observation_count):
        start = int(observation_offsets[observation_index])
        stop = int(observation_offsets[observation_index + 1])
        bits = int(masks[observation_index])
        expected_ids = _ids_from_mask_bits(bits, count)
        stored_ids = tuple(int(value) for value in action_ids[start:stop])
        if stored_ids != expected_ids:
            raise RegretFormatError(
                "sparse action IDs do not match their legal mask"
            )
        observations.append(
            RegretTargetObservation(
                perfect_recall_id=str(identifiers[observation_index]),
                legal_mask_bits=bits,
                action_ids=stored_ids,
                target_values=tuple(
                    np.float64(value) for value in signed_targets[start:stop]
                ),
                action_count=count,
            )
        )

    traversals: list[TraversalRegretRecord] = []
    for traversal_index in range(traversal_count):
        start = int(traversal_offsets[traversal_index])
        stop = int(traversal_offsets[traversal_index + 1])
        traversals.append(
            TraversalRegretRecord(
                player_id=int(players[traversal_index]),
                traversal_ordinal=int(ordinals[traversal_index]),
                observations=tuple(observations[start:stop]),
            )
        )

    encoder = metadata.get("encoder_schema_version")
    if not isinstance(encoder, str) or not encoder:
        raise RegretFormatError("encoder_schema_version must be nonempty")
    return IterationRegretRecord(
        stage_index=_integer(metadata.get("stage_index"), "stage_index", minimum=0),
        round_count=_integer(metadata.get("round_count"), "round_count", minimum=1),
        stage_iteration=_integer(
            metadata.get("stage_iteration"),
            "stage_iteration",
            minimum=1,
        ),
        global_iteration=_integer(
            metadata.get("global_iteration"),
            "global_iteration",
            minimum=1,
        ),
        traversals=tuple(traversals),
        encoder_schema_version=encoder,
        action_count=count,
    )


def _load_regret_bytes(data: bytes) -> IterationRegretRecord:
    arrays = _read_npz(io.BytesIO(data))
    metadata, payload = _metadata_from_arrays(arrays)
    return _record_from_arrays(metadata, payload)


def load_regret_sidecar(path: str | Path) -> IterationRegretRecord:
    """Load and fully validate one no-pickle signed-regret sidecar."""

    arrays = _read_npz(Path(path))
    metadata, payload = _metadata_from_arrays(arrays)
    return _record_from_arrays(metadata, payload)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        assert temporary_name is not None
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _raw_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class RegretRestoreResult:
    kept: tuple[Path, ...]
    restored: tuple[Path, ...]
    archived: tuple[Path, ...]


def _manifest_record(
    run_dir: Path,
    path: Path,
    *,
    embed: bool,
) -> dict[str, Any]:
    record = load_regret_sidecar(path)
    data = path.read_bytes()
    result: dict[str, Any] = {
        "relative_path": path.relative_to(run_dir).as_posix(),
        "stage_index": record.stage_index,
        "round_count": record.round_count,
        "stage_iteration": record.stage_iteration,
        "global_iteration": record.global_iteration,
        "size_bytes": len(data),
        "sha256": _raw_sha256(data),
    }
    if embed:
        result["payload"] = data
    return result


def _manifest_sort_key(value: Mapping[str, Any]) -> tuple[int, int]:
    return int(value["stage_index"]), int(value["stage_iteration"])


def _cursor_from_manifest(value: Mapping[str, Any]) -> dict[str, int]:
    return {
        "stage_index": int(value["stage_index"]),
        "round_count": int(value["round_count"]),
        "stage_iteration": int(value["stage_iteration"]),
        "global_iteration": int(value["global_iteration"]),
    }


def _validated_checkpoint_manifest(
    state: Mapping[str, Any],
) -> tuple[bool, tuple[Mapping[str, Any], ...]]:
    if not isinstance(state, Mapping):
        raise RegretFormatError("regret archive checkpoint state must be a mapping")
    if state.get("kind") != REGRET_ARCHIVE_KIND:
        raise RegretFormatError("not a Stockpile regret archive checkpoint state")
    if state.get("schema_version") != REGRET_ARCHIVE_SCHEMA_VERSION:
        raise RegretFormatError("unsupported regret archive checkpoint schema")
    embedded = state.get("embedded")
    if not isinstance(embedded, (bool, np.bool_)):
        raise RegretFormatError("regret archive embedded flag must be boolean")
    raw_records = state.get("records")
    if not isinstance(raw_records, (list, tuple)):
        raise RegretFormatError("regret archive records must be a list or tuple")

    records: list[Mapping[str, Any]] = []
    seen_paths: set[str] = set()
    previous_global = 0
    previous_key: tuple[int, int] | None = None
    stage_rounds: dict[int, int] = {}
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise RegretFormatError("regret archive manifest entry must be a mapping")
        stage_index = _integer(raw.get("stage_index"), "stage_index", minimum=0)
        round_count = _integer(raw.get("round_count"), "round_count", minimum=1)
        stage_iteration = _integer(
            raw.get("stage_iteration"),
            "stage_iteration",
            minimum=1,
        )
        global_iteration = _integer(
            raw.get("global_iteration"),
            "global_iteration",
            minimum=1,
        )
        size = _integer(raw.get("size_bytes"), "size_bytes", minimum=1)
        digest = raw.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(value not in _HEX_DIGITS for value in digest)
        ):
            raise RegretFormatError("manifest sha256 is invalid")
        relative = raw.get("relative_path")
        if not isinstance(relative, str):
            raise RegretFormatError("manifest relative_path must be a string")
        relative_path = relative
        pure = PurePosixPath(relative_path)
        if pure.is_absolute() or ".." in pure.parts:
            raise RegretFormatError("manifest relative_path must stay inside run")
        canonical = regret_sidecar_path(
            Path(),
            stage_index=stage_index,
            round_count=round_count,
            stage_iteration=stage_iteration,
        ).as_posix()
        if relative_path != canonical:
            raise RegretFormatError("manifest relative_path is not canonical")
        if relative_path in seen_paths:
            raise RegretFormatError("manifest contains duplicate paths")
        seen_paths.add(relative_path)
        previous_round = stage_rounds.setdefault(stage_index, round_count)
        if previous_round != round_count:
            raise RegretFormatError("one stage_index has multiple round counts")
        key = (stage_index, stage_iteration)
        if previous_key is not None and key <= previous_key:
            raise RegretFormatError("manifest records are not canonically ordered")
        if global_iteration <= previous_global:
            raise RegretFormatError("manifest global iterations are not increasing")
        previous_key = key
        previous_global = global_iteration

        if bool(embedded):
            payload = raw.get("payload")
            if not isinstance(payload, bytes):
                raise RegretFormatError("embedded manifest payload must be bytes")
            if len(payload) != size or not hmac.compare_digest(
                _raw_sha256(payload), digest
            ):
                raise RegretFormatError("embedded manifest payload hash mismatch")
            record = _load_regret_bytes(payload)
            if (
                record.stage_index != stage_index
                or record.round_count != round_count
                or record.stage_iteration != stage_iteration
                or record.global_iteration != global_iteration
            ):
                raise RegretFormatError(
                    "embedded payload metadata does not match its manifest"
                )
        elif "payload" in raw:
            raise RegretFormatError(
                "nonembedded manifest entries cannot contain payload bytes"
            )
        records.append(raw)

    cursor = state.get("cursor")
    expected_cursor = None if not records else _cursor_from_manifest(records[-1])
    if cursor != expected_cursor:
        raise RegretFormatError("regret archive cursor does not match manifest")
    return bool(embedded), tuple(records)


class RegretSidecarArchive:
    """Atomic iteration sidecars plus self-contained checkpoint snapshots."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def commit(self, iteration: IterationRegretRecord) -> Path:
        if not isinstance(iteration, IterationRegretRecord):
            raise TypeError("iteration must be an IterationRegretRecord")
        path = regret_sidecar_path(
            self.run_dir,
            stage_index=iteration.stage_index,
            round_count=iteration.round_count,
            stage_iteration=iteration.stage_iteration,
        )
        data = _serialized_sidecar(iteration)
        if path.exists():
            try:
                existing = path.read_bytes()
                existing_record = _load_regret_bytes(existing)
            except (OSError, RegretFormatError) as error:
                raise RegretArchiveError(
                    f"existing regret sidecar is unreadable: {path}"
                ) from error
            if hmac.compare_digest(
                _record_fingerprint(existing_record),
                _record_fingerprint(iteration),
            ):
                return path
            raise RegretArchiveError(
                f"conflicting regret sidecar already exists: {path}"
            )
        try:
            _atomic_write_bytes(path, data)
        except OSError as error:
            raise RegretArchiveError(f"cannot commit regret sidecar: {path}") from error
        return path

    def sidecars(self) -> tuple[Path, ...]:
        if not self.run_dir.exists():
            return ()
        return tuple(
            sorted(
                path
                for path in self.run_dir.glob(
                    "round_*/sampled_regret/iteration_*.npz"
                )
                if path.is_file()
            )
        )

    def checkpoint_state(self, embed_records: bool = True) -> dict[str, Any]:
        """Return a manifest and, by default, exact compressed NPZ bytes."""

        if not isinstance(embed_records, (bool, np.bool_)):
            raise TypeError("embed_records must be boolean")
        entries = [
            _manifest_record(self.run_dir, path, embed=bool(embed_records))
            for path in self.sidecars()
        ]
        entries.sort(key=_manifest_sort_key)
        state = {
            "kind": REGRET_ARCHIVE_KIND,
            "schema_version": REGRET_ARCHIVE_SCHEMA_VERSION,
            "embedded": bool(embed_records),
            "cursor": None if not entries else _cursor_from_manifest(entries[-1]),
            "records": entries,
        }
        self.validate_checkpoint_state(state)
        return state

    @staticmethod
    def validate_checkpoint_state(state: Mapping[str, Any]) -> None:
        """Purely validate checkpoint telemetry before trainer state mutates."""

        _validated_checkpoint_manifest(state)

    def _archive_existing(self, path: Path) -> Path:
        data = path.read_bytes()
        relative = path.relative_to(self.run_dir)
        base = self.run_dir / "analysis" / "sampled_regret_archive" / relative
        candidate = base.with_name(f"{base.name}.{_raw_sha256(data)[:12]}.bak")
        suffix = 1
        while candidate.exists():
            candidate = base.with_name(
                f"{base.name}.{_raw_sha256(data)[:12]}.{suffix}.bak"
            )
            suffix += 1
        candidate.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, candidate)
        return candidate

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
    ) -> RegretRestoreResult:
        """Reconcile to a validated manifest and rehydrate embedded sidecars."""

        embedded, entries = _validated_checkpoint_manifest(state)
        expected = {
            str(entry["relative_path"]): entry for entry in entries
        }
        local = {
            path.relative_to(self.run_dir).as_posix(): path
            for path in self.sidecars()
        }

        kept: list[Path] = []
        replacements: list[tuple[Path, bytes]] = []
        to_archive: list[Path] = []
        for relative, entry in expected.items():
            path = self.run_dir / PurePosixPath(relative)
            existing = local.get(relative)
            if existing is not None:
                try:
                    data = existing.read_bytes()
                    valid = _raw_sha256(data) == entry["sha256"]
                    if valid:
                        _load_regret_bytes(data)
                except (OSError, RegretFormatError):
                    valid = False
                if valid:
                    kept.append(existing)
                    continue
                if not embedded:
                    raise RegretArchiveError(
                        f"checkpoint sidecar conflicts and has no embedded copy: {path}"
                    )
                to_archive.append(existing)
            elif not embedded:
                raise RegretArchiveError(
                    f"checkpoint sidecar is missing and has no embedded copy: {path}"
                )
            payload = entry.get("payload")
            assert isinstance(payload, bytes)
            replacements.append((path, payload))

        for relative, path in local.items():
            if relative not in expected:
                to_archive.append(path)

        archived: list[Path] = []
        restored: list[Path] = []
        try:
            for path in sorted(set(to_archive)):
                archived.append(self._archive_existing(path))
            for path, payload in replacements:
                _atomic_write_bytes(path, payload)
                restored.append(path)
        except OSError as error:
            raise RegretArchiveError(
                "failed while reconciling regret checkpoint archive"
            ) from error
        return RegretRestoreResult(
            kept=tuple(sorted(kept)),
            restored=tuple(sorted(restored)),
            archived=tuple(sorted(archived)),
        )

    # A concise alias for trainer resume integration.
    restore = restore_checkpoint_state


@dataclass(frozen=True, slots=True)
class _IterationOccurrence:
    traversal_ordinal: int
    action_ids: NDArray[np.intp]
    target_values: FloatArray


@dataclass(frozen=True, slots=True)
class _PreparedPlayerSeries:
    """One player's fixed iteration strata, ready for prefix reconstruction."""

    iteration_groups: tuple[
        Mapping[str, tuple[_IterationOccurrence, ...]], ...
    ]
    traversal_counts: tuple[int, ...]
    last_iteration_by_id: Mapping[str, int]
    action_count: int

    def point_series(self) -> FloatArray:
        cumulative: dict[str, FloatArray] = {}
        scores: dict[str, float] = {}
        result = np.empty(len(self.iteration_groups), dtype=np.float64)
        for iteration_index, groups in enumerate(self.iteration_groups):
            traversal_count = self.traversal_counts[iteration_index]
            scale = 1.0 / traversal_count
            for identifier in sorted(groups):
                accumulated = cumulative.setdefault(
                    identifier,
                    np.zeros(self.action_count, dtype=np.float64),
                )
                for occurrence in groups[identifier]:
                    accumulated[occurrence.action_ids] += (
                        occurrence.target_values * scale
                    )
                if not np.all(np.isfinite(accumulated)):
                    raise FloatingPointError(
                        "signed regret prefix accumulation became nonfinite"
                    )
                scores[identifier] = max(0.0, float(np.max(accumulated)))
            try:
                numerator = math.fsum(scores.values())
            except OverflowError as error:
                raise FloatingPointError(
                    "sampled average regret prefix overflowed"
                ) from error
            result[iteration_index] = numerator / (iteration_index + 1)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("sampled average regret prefix is nonfinite")
        return np.maximum(result, 0.0)

    def bootstrap_series_chunk(
        self,
        rng: np.random.Generator,
        sample_count: int,
    ) -> FloatArray:
        """Draw every fixed iteration stratum once and emit all prefixes."""

        cumulative: dict[str, FloatArray] = {}
        scores: dict[str, FloatArray] = {}
        numerator = np.zeros(sample_count, dtype=np.float64)
        result = np.empty(
            (sample_count, len(self.iteration_groups)),
            dtype=np.float64,
        )
        for iteration_index, groups in enumerate(self.iteration_groups):
            traversal_count = self.traversal_counts[iteration_index]
            counts = rng.multinomial(
                traversal_count,
                np.full(
                    traversal_count,
                    1.0 / traversal_count,
                    dtype=np.float64,
                ),
                size=sample_count,
            ).astype(np.float64, copy=False)
            scale = 1.0 / traversal_count
            for identifier in sorted(groups):
                old_score = scores.get(identifier)
                if old_score is None:
                    old_score = np.zeros(sample_count, dtype=np.float64)
                accumulated = cumulative.get(identifier)
                if accumulated is None:
                    accumulated = np.zeros(
                        (sample_count, self.action_count),
                        dtype=np.float64,
                    )
                for occurrence in groups[identifier]:
                    accumulated[:, occurrence.action_ids] += (
                        counts[:, occurrence.traversal_ordinal, None]
                        * occurrence.target_values[None, :]
                        * scale
                    )
                if not np.all(np.isfinite(accumulated)):
                    raise FloatingPointError(
                        "bootstrap signed regret prefix became nonfinite"
                    )
                new_score = np.maximum(0.0, np.max(accumulated, axis=1))
                numerator += new_score - old_score
                if self.last_iteration_by_id[identifier] == iteration_index:
                    cumulative.pop(identifier, None)
                    scores.pop(identifier, None)
                else:
                    cumulative[identifier] = accumulated
                    scores[identifier] = new_score
            result[:, iteration_index] = numerator / (iteration_index + 1)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("bootstrap sampled regret series is nonfinite")
        return np.maximum(result, 0.0)


def _prepare_player_series(
    iterations: Sequence[IterationRegretRecord],
    player_id: int,
) -> _PreparedPlayerSeries:
    iteration_groups: list[Mapping[str, tuple[_IterationOccurrence, ...]]] = []
    traversal_counts: list[int] = []
    last_iteration: dict[str, int] = {}
    masks: dict[str, int] = {}
    for iteration_index, iteration in enumerate(iterations):
        traversals = [
            value for value in iteration.traversals if value.player_id == player_id
        ]
        if not traversals:
            raise RegretFormatError(
                f"iteration {iteration.stage_iteration} has no player {player_id} traversals"
            )
        grouped: dict[str, list[_IterationOccurrence]] = defaultdict(list)
        for traversal in traversals:
            for observation in traversal.observations:
                previous = masks.setdefault(
                    observation.perfect_recall_id,
                    observation.legal_mask_bits,
                )
                if previous != observation.legal_mask_bits:
                    raise RegretFormatError(
                        "one perfect_recall_id has inconsistent legal masks "
                        "across sidecars"
                    )
                grouped[observation.perfect_recall_id].append(
                    _IterationOccurrence(
                        traversal_ordinal=traversal.traversal_ordinal,
                        action_ids=np.asarray(
                            observation.action_ids,
                            dtype=np.intp,
                        ),
                        target_values=np.asarray(
                            observation.target_values,
                            dtype=np.float64,
                        ),
                    )
                )
                last_iteration[observation.perfect_recall_id] = iteration_index
        iteration_groups.append(
            {
                identifier: tuple(occurrences)
                for identifier, occurrences in grouped.items()
            }
        )
        traversal_counts.append(len(traversals))
    return _PreparedPlayerSeries(
        iteration_groups=tuple(iteration_groups),
        traversal_counts=tuple(traversal_counts),
        last_iteration_by_id=last_iteration,
        action_count=iterations[0].action_count,
    )


def _percentile_interval(
    values: FloatArray,
    confidence: float,
) -> list[float]:
    tail = (1.0 - confidence) / 2.0
    quantiles = np.quantile(values, [tail, 1.0 - tail])
    if not np.all(np.isfinite(quantiles)):
        raise FloatingPointError("bootstrap confidence interval is nonfinite")
    return [float(quantiles[0]), float(quantiles[1])]


def _stage_report(
    records: Sequence[IterationRegretRecord],
    *,
    rng: np.random.Generator,
    replicates: int,
    confidence: float,
    stage_number: int,
    stage_count: int,
    progress: Callable[[RegretAnalysisProgress], None] | None,
) -> dict[str, Any]:
    ordered = tuple(sorted(records, key=lambda value: value.stage_iteration))
    first = ordered[0]
    actual_iterations = tuple(value.stage_iteration for value in ordered)
    expected_iterations = tuple(range(1, actual_iterations[-1] + 1))
    contiguous_count = 0
    for expected, actual in enumerate(actual_iterations, start=1):
        if actual != expected:
            break
        contiguous_count += 1
    complete_prefix = actual_iterations == expected_iterations
    traversal_counts = {
        f"player_{player}": sum(
            traversal.player_id == player
            for iteration in ordered
            for traversal in iteration.traversals
        )
        for player in (0, 1)
    }
    target_counts = {
        f"player_{player}": sum(
            len(traversal.observations)
            for iteration in ordered
            for traversal in iteration.traversals
            if traversal.player_id == player
        )
        for player in (0, 1)
    }
    base: dict[str, Any] = {
        "stage_index": first.stage_index,
        "round_count": first.round_count,
        "available": contiguous_count > 0,
        "complete_prefix": complete_prefix,
        "reason": (
            None
            if complete_prefix
            else (
                "incomplete_outer_iteration_prefix"
                if contiguous_count
                else "no_reconstructible_outer_iteration_prefix"
            )
        ),
        "iteration_count": len(ordered),
        "first_stage_iteration": actual_iterations[0],
        "last_stage_iteration": actual_iterations[-1],
        "normalizer_outer_iterations": (
            len(ordered) if complete_prefix else None
        ),
        "traversal_count": traversal_counts,
        "target_count": target_counts,
        "point": {"player_0": None, "player_1": None, "maximum": None},
        "confidence_interval": {
            "player_0": None,
            "player_1": None,
            "maximum": None,
        },
        "series": [],
    }

    point_series: tuple[FloatArray, FloatArray] | None = None
    replicate_series: tuple[FloatArray, FloatArray] | None = None
    if contiguous_count:
        if progress is not None:
            progress(
                RegretAnalysisProgress(
                    stage_index=first.stage_index,
                    round_count=first.round_count,
                    stage_number=stage_number,
                    stage_count=stage_count,
                    completed_replicates=0,
                    total_replicates=replicates,
                )
            )
        prefix = ordered[:contiguous_count]
        prepared = tuple(
            _prepare_player_series(prefix, player) for player in (0, 1)
        )
        point_series = (
            prepared[0].point_series(),
            prepared[1].point_series(),
        )
        mutable_replicates = [
            np.empty((contiguous_count, replicates), dtype=np.float64),
            np.empty((contiguous_count, replicates), dtype=np.float64),
        ]
        for start in range(0, replicates, _BOOTSTRAP_CHUNK_SIZE):
            stop = min(start + _BOOTSTRAP_CHUNK_SIZE, replicates)
            for player in (0, 1):
                chunk = prepared[player].bootstrap_series_chunk(
                    rng,
                    stop - start,
                )
                mutable_replicates[player][:, start:stop] = chunk.T
            if progress is not None:
                progress(
                    RegretAnalysisProgress(
                        stage_index=first.stage_index,
                        round_count=first.round_count,
                        stage_number=stage_number,
                        stage_count=stage_count,
                        completed_replicates=stop,
                        total_replicates=replicates,
                    )
                )
        replicate_series = (mutable_replicates[0], mutable_replicates[1])

    series: list[dict[str, Any]] = []
    for index, iteration in enumerate(ordered):
        available = index < contiguous_count
        entry: dict[str, Any] = {
            "stage_iteration": iteration.stage_iteration,
            "global_iteration": iteration.global_iteration,
            "available": available,
            "reason": None if available else "incomplete_outer_iteration_prefix",
            "normalizer_outer_iterations": index + 1 if available else None,
            "point": {"player_0": None, "player_1": None, "maximum": None},
            "confidence_interval": {
                "player_0": None,
                "player_1": None,
                "maximum": None,
            },
        }
        if available:
            assert point_series is not None
            assert replicate_series is not None
            player_0 = float(point_series[0][index])
            player_1 = float(point_series[1][index])
            maximum_replicates = np.maximum(
                replicate_series[0][index],
                replicate_series[1][index],
            )
            entry["point"] = {
                "player_0": player_0,
                "player_1": player_1,
                "maximum": max(player_0, player_1),
            }
            entry["confidence_interval"] = {
                "player_0": _percentile_interval(
                    replicate_series[0][index],
                    confidence,
                ),
                "player_1": _percentile_interval(
                    replicate_series[1][index],
                    confidence,
                ),
                # This is deliberately the percentile of the replicatewise
                # maximum, not a maximum of the two marginal endpoints.
                "maximum": _percentile_interval(
                    maximum_replicates,
                    confidence,
                ),
            }
        series.append(entry)

    base["series"] = series
    final_entry = series[-1]
    base["point"] = final_entry["point"]
    base["confidence_interval"] = final_entry["confidence_interval"]
    return base


def _analysis_paths(
    run_dir_or_paths: str | Path | Iterable[str | Path],
) -> tuple[Path, ...]:
    if isinstance(run_dir_or_paths, (str, Path)):
        path = Path(run_dir_or_paths)
        if path.is_dir():
            return RegretSidecarArchive(path).sidecars()
        if path.suffix == ".npz":
            return (path,)
        # A CLI may be handed ``round_XX/full.pt``; locating its run directory
        # does not require importing or trusting the Torch checkpoint.
        if path.name == "full.pt" and path.parent.name.startswith("round_"):
            return RegretSidecarArchive(path.parent.parent).sidecars()
        return ()
    return tuple(sorted(Path(value) for value in run_dir_or_paths))


def _validated_analysis_settings(
    replicates: Any,
    confidence: Any,
    seed: Any,
) -> tuple[int, float, int, dict[str, Any]]:
    replicate_count = _integer(replicates, "replicates", minimum=1)
    level = _finite_float(confidence, "confidence")
    if not 0.0 < level < 1.0:
        raise RegretFormatError("confidence must be in (0, 1)")
    bootstrap_seed = _integer(seed, "seed", minimum=0)
    return (
        replicate_count,
        level,
        bootstrap_seed,
        {
            "method": "stratified_complete_traversal_percentile",
            "replicates": replicate_count,
            "confidence": level,
            "seed": bootstrap_seed,
            "bit_generator": "PCG64",
        },
    )


def _empty_report(
    bootstrap_metadata: Mapping[str, Any],
    *,
    telemetry_declared: bool,
    reason: str,
    source: str | None,
) -> dict[str, Any]:
    return {
        "kind": REGRET_REPORT_KIND,
        "schema_version": REGRET_REPORT_SCHEMA_VERSION,
        "sidecar_schema_version": REGRET_SIDECAR_SCHEMA_VERSION,
        "availability": {
            "available": False,
            "reason": reason,
            "sidecar_count": 0,
            "source": source,
            "telemetry_declared": telemetry_declared,
        },
        "statistic": {
            "name": "sampled_average_regret",
            "time_index": "complete_outer_stage_iterations",
            "traversal_aggregation": "mean_within_iteration_player",
            "information_key": "exact_perfect_recall_id",
        },
        "bootstrap": dict(bootstrap_metadata),
        "stages": [],
        "latest_stage": None,
    }


def _analyze_records(
    records: Sequence[IterationRegretRecord],
    *,
    record_count: int,
    replicate_count: int,
    level: float,
    bootstrap_seed: int,
    bootstrap_metadata: Mapping[str, Any],
    source: str,
    progress: Callable[[RegretAnalysisProgress], None] | None,
) -> dict[str, Any]:
    signatures = {
        (record.encoder_schema_version, record.action_count) for record in records
    }
    if len(signatures) != 1:
        raise RegretFormatError(
            "sidecars use incompatible encoder schemas or action counts"
        )
    grouped: dict[int, list[IterationRegretRecord]] = defaultdict(list)
    seen_iterations: set[tuple[int, int]] = set()
    stage_rounds: dict[int, int] = {}
    seen_global: set[int] = set()
    for record in records:
        key = (record.stage_index, record.stage_iteration)
        if key in seen_iterations:
            raise RegretFormatError("duplicate sidecars for one stage iteration")
        seen_iterations.add(key)
        if record.global_iteration in seen_global:
            raise RegretFormatError("duplicate global iteration in sidecars")
        seen_global.add(record.global_iteration)
        prior_round = stage_rounds.setdefault(record.stage_index, record.round_count)
        if prior_round != record.round_count:
            raise RegretFormatError("one stage_index has multiple round counts")
        grouped[record.stage_index].append(record)

    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    ordered_stage_indexes = sorted(grouped)
    stage_count = len(ordered_stage_indexes)
    stages = [
        _stage_report(
            grouped[stage_index],
            rng=rng,
            replicates=replicate_count,
            confidence=level,
            stage_number=stage_number,
            stage_count=stage_count,
            progress=progress,
        )
        for stage_number, stage_index in enumerate(ordered_stage_indexes, start=1)
    ]
    available = any(bool(stage["available"]) for stage in stages)
    all_complete = all(bool(stage["complete_prefix"]) for stage in stages)
    return {
        "kind": REGRET_REPORT_KIND,
        "schema_version": REGRET_REPORT_SCHEMA_VERSION,
        "sidecar_schema_version": REGRET_SIDECAR_SCHEMA_VERSION,
        "availability": {
            "available": available,
            "reason": (
                None
                if available and all_complete
                else (
                    "partial_outer_iteration_prefix"
                    if available
                    else "no_reconstructible_outer_iteration_prefix"
                )
            ),
            "sidecar_count": record_count,
            "source": source,
            "telemetry_declared": True,
        },
        "statistic": {
            "name": "sampled_average_regret",
            "time_index": "complete_outer_stage_iterations",
            "traversal_aggregation": "mean_within_iteration_player",
            "information_key": "exact_perfect_recall_id",
        },
        "bootstrap": dict(bootstrap_metadata),
        "stages": stages,
        "latest_stage": stages[-1],
    }


def analyze_regret(
    run_dir_or_paths: str | Path | Iterable[str | Path],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    progress: Callable[[RegretAnalysisProgress], None] | None = None,
) -> dict[str, Any]:
    """Analyze sidecars without importing a policy or mutating global RNGs."""

    (
        replicate_count,
        level,
        bootstrap_seed,
        bootstrap_metadata,
    ) = _validated_analysis_settings(replicates, confidence, seed)
    paths = _analysis_paths(run_dir_or_paths)
    if not paths:
        return _empty_report(
            bootstrap_metadata,
            telemetry_declared=False,
            reason="no_sampled_regret_sidecars",
            source=None,
        )

    records = [load_regret_sidecar(path) for path in paths]
    return _analyze_records(
        records,
        record_count=len(paths),
        replicate_count=replicate_count,
        level=level,
        bootstrap_seed=bootstrap_seed,
        bootstrap_metadata=bootstrap_metadata,
        source="iteration_sidecars",
        progress=progress,
    )


def _analysis_run_dir(path: Path) -> Path:
    if path.is_dir() and path.name.startswith("round_"):
        return path.parent
    if path.is_dir():
        return path
    if path.parent.name.startswith("round_"):
        return path.parent.parent
    return path.parent


def _declared_telemetry(run_dir: Path) -> bool:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return False
    try:
        config = _strict_json(
            config_path.read_text(encoding="utf-8"),
            "config.json",
        )
    except OSError as error:
        raise RegretFormatError("cannot read run config.json") from error
    marker = config.get("sampled_regret_telemetry")
    if marker is None:
        training = config.get("training")
        if isinstance(training, Mapping):
            marker = training.get("sampled_regret_telemetry")
    if marker is None:
        return False
    if not isinstance(marker, Mapping):
        raise RegretFormatError("sampled_regret_telemetry marker must be an object")
    if marker.get("record_schema_version") != REGRET_SIDECAR_SCHEMA_VERSION:
        raise RegretFormatError("unsupported declared regret record schema")
    return True


def _latest_full_checkpoint(run_dir: Path, requested: Path) -> Path | None:
    if requested.name == "full.pt" and requested.is_file():
        return requested
    candidates = tuple(sorted(run_dir.glob("round_*/full.pt")))
    return candidates[-1] if candidates else None


def _embedded_checkpoint_records(path: Path) -> list[IterationRegretRecord] | None:
    # This import is intentionally isolated behind either an explicit full.pt
    # request or a run-level telemetry marker and a no-sidecar check.  Normal
    # legacy directory analysis therefore never loads a potentially huge
    # checkpoint merely to discover that telemetry does not exist, while a
    # copied v2 full checkpoint remains self-contained for analysis.
    try:
        import torch
    except ImportError as error:  # pragma: no cover - training installs Torch
        raise RegretArchiveError(
            "Torch is required to read embedded regret telemetry"
        ) from error
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise RegretFormatError(
            f"cannot read embedded telemetry checkpoint: {path}"
        ) from error
    if not isinstance(payload, Mapping) or payload.get("kind") != (
        "stockpile_deep_cfr_training"
    ):
        raise RegretFormatError("not a Stockpile Deep CFR training checkpoint")
    schema = payload.get("schema_version")
    if schema == 1:
        return None
    if schema != 2:
        raise RegretFormatError("unsupported Deep CFR checkpoint schema")
    state = payload.get("sampled_regret_telemetry")
    if not isinstance(state, Mapping):
        raise RegretFormatError("checkpoint is missing sampled regret telemetry")
    embedded, entries = _validated_checkpoint_manifest(state)
    if not embedded:
        raise RegretFormatError(
            "checkpoint regret manifest is not self-contained"
        )
    return [
        _load_regret_bytes(entry["payload"])
        for entry in entries
        if isinstance(entry.get("payload"), bytes)
    ]


def _records_have_contiguous_prefixes(
    records: Sequence[IterationRegretRecord],
) -> bool:
    if not records:
        return False
    grouped: dict[int, list[int]] = defaultdict(list)
    for record in records:
        grouped[record.stage_index].append(record.stage_iteration)
    return all(
        tuple(sorted(values)) == tuple(range(1, max(values) + 1))
        for values in grouped.values()
    )


def _merged_records(
    sidecars: Sequence[IterationRegretRecord],
    embedded: Sequence[IterationRegretRecord],
) -> list[IterationRegretRecord]:
    result: dict[tuple[int, int], IterationRegretRecord] = {}
    for record in (*embedded, *sidecars):
        key = (record.stage_index, record.stage_iteration)
        previous = result.get(key)
        if previous is not None and not hmac.compare_digest(
            _record_fingerprint(previous),
            _record_fingerprint(record),
        ):
            raise RegretFormatError(
                "sidecar conflicts with embedded checkpoint regret record"
            )
        result[key] = record
    return [result[key] for key in sorted(result)]


def analyze_run(
    path: str | Path,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    progress: Callable[[RegretAnalysisProgress], None] | None = None,
) -> dict[str, Any]:
    """Analyze a run, falling back to marked v2 checkpoint telemetry only."""

    requested = Path(path)
    (
        replicate_count,
        level,
        bootstrap_seed,
        bootstrap_metadata,
    ) = _validated_analysis_settings(
        bootstrap_replicates,
        confidence,
        seed,
    )

    if requested.suffix == ".npz":
        return analyze_regret(
            requested,
            replicates=bootstrap_replicates,
            confidence=confidence,
            seed=seed,
            progress=progress,
        )
    # A compact policy contains no signed targets.  Its siblings must not
    # silently broaden an explicit policy-only analysis request.
    if requested.name == "policy.pt":
        return _empty_report(
            bootstrap_metadata,
            telemetry_declared=False,
            reason="policy_has_no_sampled_regret",
            source=None,
        )

    run_dir = _analysis_run_dir(requested)
    # Likewise, an explicit full checkpoint is evaluated from its own embedded
    # archive only. Schema-v1 is exactly N/A and is never inferred from memory.
    if requested.name == "full.pt":
        embedded_records = _embedded_checkpoint_records(requested)
        if embedded_records is None:
            return _empty_report(
                bootstrap_metadata,
                telemetry_declared=False,
                reason="legacy_checkpoint_has_no_sampled_regret",
                source=None,
            )
        if not embedded_records:
            return _empty_report(
                bootstrap_metadata,
                telemetry_declared=True,
                reason="no_completed_outer_iterations",
                source="embedded_checkpoint",
            )
        return _analyze_records(
            embedded_records,
            record_count=len(embedded_records),
            replicate_count=replicate_count,
            level=level,
            bootstrap_seed=bootstrap_seed,
            bootstrap_metadata=bootstrap_metadata,
            source="embedded_checkpoint",
            progress=progress,
        )

    sidecar_paths = RegretSidecarArchive(run_dir).sidecars()
    sidecar_records = [load_regret_sidecar(path) for path in sidecar_paths]
    if sidecar_records and _records_have_contiguous_prefixes(sidecar_records):
        # Fast normal path: do not touch a large full checkpoint when all
        # observed stages already form reconstructible prefixes.
        return _analyze_records(
            sidecar_records,
            record_count=len(sidecar_records),
            replicate_count=replicate_count,
            level=level,
            bootstrap_seed=bootstrap_seed,
            bootstrap_metadata=bootstrap_metadata,
            source="iteration_sidecars",
            progress=progress,
        )

    declared = _declared_telemetry(run_dir)
    if sidecar_records and declared:
        checkpoint = _latest_full_checkpoint(run_dir, requested)
        if checkpoint is not None:
            embedded_records = _embedded_checkpoint_records(checkpoint)
            if embedded_records is not None:
                merged = _merged_records(sidecar_records, embedded_records)
                return _analyze_records(
                    merged,
                    record_count=len(merged),
                    replicate_count=replicate_count,
                    level=level,
                    bootstrap_seed=bootstrap_seed,
                    bootstrap_metadata=bootstrap_metadata,
                    source="iteration_sidecars_plus_embedded_checkpoint",
                    progress=progress,
                )
    if sidecar_records:
        return _analyze_records(
            sidecar_records,
            record_count=len(sidecar_records),
            replicate_count=replicate_count,
            level=level,
            bootstrap_seed=bootstrap_seed,
            bootstrap_metadata=bootstrap_metadata,
            source="iteration_sidecars",
            progress=progress,
        )

    if not declared:
        return _empty_report(
            bootstrap_metadata,
            telemetry_declared=False,
            reason="no_sampled_regret_sidecars",
            source=None,
        )
    checkpoint = _latest_full_checkpoint(run_dir, requested)
    if checkpoint is None:
        return _empty_report(
            bootstrap_metadata,
            telemetry_declared=True,
            reason="no_full_checkpoint",
            source="declared_telemetry",
        )
    embedded_records = _embedded_checkpoint_records(checkpoint)
    if embedded_records is None:
        # Explicit legacy behavior: never infer telemetry from replay memory.
        return _empty_report(
            bootstrap_metadata,
            telemetry_declared=False,
            reason="legacy_checkpoint_has_no_sampled_regret",
            source=None,
        )
    if not embedded_records:
        return _empty_report(
            bootstrap_metadata,
            telemetry_declared=True,
            reason="no_completed_outer_iterations",
            source="embedded_checkpoint",
        )
    return _analyze_records(
        embedded_records,
        record_count=len(embedded_records),
        replicate_count=replicate_count,
        level=level,
        bootstrap_seed=bootstrap_seed,
        bootstrap_metadata=bootstrap_metadata,
        source="embedded_checkpoint",
        progress=progress,
    )


def write_analysis_report(
    run_dir: str | Path,
    report: Mapping[str, Any],
) -> Path:
    """Write ``<run>/analysis/sampled_average_regret.json`` atomically."""

    return _write_report_to_path(analysis_report_path(run_dir), report)


def _write_report_to_path(
    destination: Path,
    report: Mapping[str, Any],
) -> Path:
    """Atomically write strict JSON; unavailable values remain JSON null."""

    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    try:
        rendered = json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as error:
        raise RegretFormatError("report is not strict JSON serializable") from error
    try:
        _atomic_write_bytes(destination, rendered.encode("utf-8"))
    except OSError as error:
        raise RegretArchiveError(
            f"cannot write sampled-regret analysis report: {destination}"
        ) from error
    return destination


def write_regret_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Low-level report-first writer for an explicit destination path."""

    return _write_report_to_path(Path(path), report)


__all__ = [
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE",
    "REGRET_ARCHIVE_KIND",
    "REGRET_ARCHIVE_SCHEMA_VERSION",
    "REGRET_REPORT_KIND",
    "REGRET_REPORT_SCHEMA_VERSION",
    "REGRET_SIDECAR_KIND",
    "REGRET_SIDECAR_SCHEMA_VERSION",
    "IterationRegretRecord",
    "RegretArchiveError",
    "RegretAnalysisProgress",
    "RegretFormatError",
    "RegretIterationCapture",
    "RegretRestoreResult",
    "RegretSidecarArchive",
    "RegretTargetObservation",
    "RegretTraversalCapture",
    "TraversalRegretRecord",
    "analysis_report_path",
    "analyze_regret",
    "analyze_run",
    "load_regret_sidecar",
    "regret_sidecar_path",
    "write_analysis_report",
    "write_regret_report",
]
