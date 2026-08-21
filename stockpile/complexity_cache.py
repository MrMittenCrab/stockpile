"""Persistent, versioned memory for Stockpile complexity traversals.

The cache stores summaries produced by
``stockpile_platform.compute_information_set_complexity``; it never stores an
estimate in place of that traversal result.  Cache identity is based on the
effective rules and the actual action-codec ranges, so interface-only choices
such as the preset label and compact/shared padding can share a traversal when
their reachable trees are identical.

Two current stores are used by default:

* a generated preset seed distributed in ``stockpile/data``; and
* a learned store in the same folder for configurations calculated on demand.

Both files are independently validated. Version-one and version-two stores
are searched read-only as migration sources. A legacy entry is reusable only
when its validated semantic payload, after applying documented historical
defaults, exactly matches the requested game. Neighbouring configurations are
never treated as evidence. A missing, stale, or malformed file is a cache
miss, allowing callers to fall back to a live traversal.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import stockpile_platform as platform


CACHE_SCHEMA_VERSION = 3
"""On-disk JSON schema version."""

COMPLEXITY_SEMANTICS_VERSION = 1
"""Version of fingerprinting, traversal, and information-state semantics."""

LEGACY_CACHE_SCHEMA_VERSIONS = (1, 2)
"""Read-only schemas eligible for exact-semantic migration."""

_LEGACY_V1_STARTING_SHARES_PER_PLAYER = 1
_LEGACY_SEQUENTIAL_OBSERVABLE_SELLING = True

_PACKAGE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_PRESET_CACHE_PATH = (
    _PACKAGE_DIRECTORY / "data" / f"preset_complexity_v{CACHE_SCHEMA_VERSION}.json"
)
DEFAULT_LEARNED_CACHE_PATH = (
    _PACKAGE_DIRECTORY / "data" / f"complexity_memory_v{CACHE_SCHEMA_VERSION}.json"
)
DEFAULT_LEGACY_PRESET_CACHE_PATHS = tuple(
    _PACKAGE_DIRECTORY / "data" / f"preset_complexity_v{version}.json"
    for version in LEGACY_CACHE_SCHEMA_VERSIONS
)
DEFAULT_LEGACY_LEARNED_CACHE_PATHS = tuple(
    _PACKAGE_DIRECTORY / "data" / f"complexity_memory_v{version}.json"
    for version in LEGACY_CACHE_SCHEMA_VERSIONS
)

_MAX_CACHE_BYTES = 64 * 1024 * 1024
_DIGEST_LENGTH = 64

CacheSource = Literal["preset", "learned", "live"]
StoredCacheSource = Literal["preset", "learned"]


class ComplexityProvenance(BaseModel):
    """Where and under which semantics a complexity result was produced."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source: CacheSource
    generated_at: datetime
    schema_version: int = Field(ge=1)
    semantics_version: int = Field(ge=1)
    platform_digest: str = Field(min_length=_DIGEST_LENGTH, max_length=_DIGEST_LENGTH)

    @field_validator("generated_at")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return value.astimezone(timezone.utc)

    @field_validator("platform_digest")
    @classmethod
    def _sha256_hex(cls, value: str) -> str:
        lowered = value.lower()
        if any(character not in "0123456789abcdef" for character in lowered):
            raise ValueError("platform_digest must be a SHA-256 hexadecimal digest")
        return lowered


class CachedComplexity(BaseModel):
    """One validated traversal record stored under its semantic fingerprint."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    fingerprint: str = Field(min_length=_DIGEST_LENGTH, max_length=_DIGEST_LENGTH)
    semantic_rules: dict[str, Any]
    result: platform.InformationSetComplexity
    provenance: ComplexityProvenance

    @field_validator("fingerprint")
    @classmethod
    def _fingerprint_sha256(cls, value: str) -> str:
        lowered = value.lower()
        if any(character not in "0123456789abcdef" for character in lowered):
            raise ValueError("fingerprint must be a SHA-256 hexadecimal digest")
        return lowered

    @model_validator(mode="after")
    def _consistent_result(self) -> "CachedComplexity":
        result = self.result
        if result.exact != (result.count_kind == "exact"):
            raise ValueError("result exact and count_kind fields disagree")
        scalar_counts = (
            result.information_sets,
            result.information_set_actions,
            result.max_actions_per_information_set,
            result.states_visited,
            result.terminal_states,
            result.chance_nodes,
        )
        if any(value < 0 for value in scalar_counts):
            raise ValueError("complexity counts must be non-negative")
        if any(value < 0 for value in result.per_player_information_sets.values()):
            raise ValueError("per-player information-set counts must be non-negative")
        if any(
            value < 0
            for value in result.per_player_information_set_actions.values()
        ):
            raise ValueError(
                "per-player information-set-action counts must be non-negative"
            )
        if sum(result.per_player_information_sets.values()) != result.information_sets:
            raise ValueError("per-player information sets do not sum to the total")
        if (
            sum(result.per_player_information_set_actions.values())
            != result.information_set_actions
        ):
            raise ValueError(
                "per-player information-set actions do not sum to the total"
            )
        if result.max_actions_per_information_set > result.information_set_actions:
            raise ValueError("maximum actions cannot exceed the total action count")
        if result.states_visited > result.max_states:
            raise ValueError("states_visited cannot exceed max_states")
        return self


class ComplexityBounds(BaseModel):
    """Observed lower bounds and a finite structural upper bound.

    Non-exact structural ceilings remain symbolic.  This prevents an unusual
    custom configuration from materialising an integer with millions of digits
    merely so a terminal can display its order of magnitude.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    exact: bool
    lower_information_sets: int = Field(ge=0)
    lower_information_set_actions: int = Field(ge=0)
    upper_information_sets: int | None = Field(default=None, ge=0)
    upper_information_set_actions: int | None = Field(default=None, ge=0)
    upper_information_sets_expression: str
    upper_information_set_actions_expression: str
    upper_information_sets_log10: float = Field(ge=0)
    upper_information_set_actions_log10: float = Field(ge=0)
    player_decision_bound: int = Field(ge=0)
    chance_node_bound: int = Field(ge=0)
    max_legal_actions: int = Field(ge=1)
    max_chance_outcomes: int = Field(ge=1)

    @model_validator(mode="after")
    def _exact_ceiling_consistency(self) -> "ComplexityBounds":
        if self.exact:
            if self.upper_information_sets != self.lower_information_sets:
                raise ValueError("an exact infoset result must collapse its bounds")
            if (
                self.upper_information_set_actions
                != self.lower_information_set_actions
            ):
                raise ValueError(
                    "an exact infoset-action result must collapse its bounds"
                )
        elif (
            self.upper_information_sets is not None
            or self.upper_information_set_actions is not None
        ):
            raise ValueError("non-exact structural ceilings must remain symbolic")
        return self


class _ComplexityCacheDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: int
    semantics_version: int
    platform_digest: str
    entries: dict[str, CachedComplexity]


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("semantic rule values must be finite")
        return value
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, range):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("semantic rule mappings must use string keys")
            converted[key] = _canonical_value(item)
        return {key: converted[key] for key in sorted(converted)}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if is_dataclass(value):
        return {
            item.name: _canonical_value(getattr(value, item.name))
            for item in fields(value)
        }
    raise TypeError(f"unsupported semantic rule value: {type(value).__name__}")


def _coerce_rule_set(
    configuration: platform.ConfiguredGame | platform.StockpileGame | platform.RuleSet,
) -> platform.RuleSet:
    if isinstance(configuration, platform.ConfiguredGame):
        return configuration.rule_set
    if isinstance(configuration, platform.StockpileGame):
        return configuration.rule_set
    if isinstance(configuration, platform.RuleSet):
        return configuration
    raise TypeError("configuration must be a ConfiguredGame, StockpileGame, or RuleSet")


def semantic_rule_payload(
    configuration: platform.ConfiguredGame | platform.StockpileGame | platform.RuleSet,
) -> dict[str, Any]:
    """Return the canonical effective rules that determine traversal counts.

    The profile label, requested action-space mode, and padded policy-head sizes
    are interface metadata.  Actual codec namespaces and their ranges remain in
    the payload because action identifiers appear in histories and infosets.
    """

    rule_set = _coerce_rule_set(configuration)
    payload: dict[str, Any] = {
        "semantic_payload_version": COMPLEXITY_SEMANTICS_VERSION,
    }
    excluded = {"profile", "action_space_mode", "action_codec"}
    for item in fields(rule_set):
        if item.name not in excluded:
            payload[item.name] = _canonical_value(getattr(rule_set, item.name))
    payload["action_codec_ranges"] = [
        {
            "namespace": namespace,
            "start": action_range.start,
            "stop": action_range.stop,
            "step": action_range.step,
        }
        for namespace, action_range in rule_set.action_codec.ranges.items()
    ]
    return _canonical_value(payload)


def _fingerprint_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _canonical_value(payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _migrate_legacy_semantic_payload(
    payload: Mapping[str, Any],
    *,
    schema_version: int,
) -> dict[str, Any]:
    """Normalize historical semantics without inventing configurable values.

    Schema v1 predates the explicit starting-share field, but that engine
    always dealt one starting share per player. Schemas v1 and v2 predate the
    explicit selling-order field, but both engines exposed each sale before
    the next player acted. Treating absence as those known historical values
    permits exact reuse while rejecting configurations with different setup
    or selling observability.
    """

    migrated = dict(_canonical_value(payload))
    if schema_version == 1 and "starting_shares_per_player" not in migrated:
        migrated["starting_shares_per_player"] = (
            _LEGACY_V1_STARTING_SHARES_PER_PLAYER
        )
    if (
        schema_version in LEGACY_CACHE_SCHEMA_VERSIONS
        and "sequential_observable_selling" not in migrated
    ):
        migrated["sequential_observable_selling"] = (
            _LEGACY_SEQUENTIAL_OBSERVABLE_SELLING
        )
    return _canonical_value(migrated)


def semantic_fingerprint(
    configuration: platform.ConfiguredGame | platform.StockpileGame | platform.RuleSet,
) -> str:
    """Return a stable SHA-256 identity for a reachable game tree."""

    return _fingerprint_payload(semantic_rule_payload(configuration))


def platform_source_digest() -> str:
    """Hash the platform implementation used to produce traversal records."""

    if platform.__file__ is None:
        raise RuntimeError("the Stockpile platform has no source file")
    source_path = Path(platform.__file__).resolve()
    return hashlib.sha256(source_path.read_bytes()).hexdigest()


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def live_complexity_provenance() -> ComplexityProvenance:
    """Create provenance for a live result that has not been stored."""

    return ComplexityProvenance(
        source="live",
        generated_at=_now_utc(),
        schema_version=CACHE_SCHEMA_VERSION,
        semantics_version=COMPLEXITY_SEMANTICS_VERSION,
        platform_digest=platform_source_digest(),
    )


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _is_sha256_hex(value: str) -> bool:
    return len(value) == _DIGEST_LENGTH and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _validate_document_integrity(
    document: _ComplexityCacheDocument,
    *,
    expected_source: StoredCacheSource,
    accepted_schema_versions: Sequence[int] = (CACHE_SCHEMA_VERSION,),
) -> None:
    if document.schema_version not in accepted_schema_versions:
        raise ValueError("cache schema version is stale")
    if document.semantics_version != COMPLEXITY_SEMANTICS_VERSION:
        raise ValueError("cache semantics version is stale")
    if not _is_sha256_hex(document.platform_digest):
        raise ValueError("cache platform digest is invalid")

    for key, entry in document.entries.items():
        if key != entry.fingerprint:
            raise ValueError("cache entry key does not match its fingerprint")
        if _fingerprint_payload(entry.semantic_rules) != entry.fingerprint:
            raise ValueError(
                "cache entry semantic payload does not match its fingerprint"
            )
        provenance = entry.provenance
        if provenance.source != expected_source:
            raise ValueError("cache entry is stored in the wrong source file")
        if provenance.schema_version != document.schema_version:
            raise ValueError("cache entry schema version does not match its document")
        if provenance.semantics_version != document.semantics_version:
            raise ValueError(
                "cache entry semantics version does not match its document"
            )
        # Source digests are provenance, not cache identity. A document may
        # legitimately contain entries calculated against different platform
        # revisions after a stronger result is retained during a later save.


def _load_document(
    path: Path | None,
    *,
    expected_source: StoredCacheSource,
    accepted_schema_versions: Sequence[int] = (CACHE_SCHEMA_VERSION,),
) -> _ComplexityCacheDocument | None:
    if path is None:
        return None
    try:
        size = path.stat().st_size
        if size > _MAX_CACHE_BYTES:
            return None
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="strict")
        json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
        document = _ComplexityCacheDocument.model_validate_json(raw, strict=True)
        _validate_document_integrity(
            document,
            expected_source=expected_source,
            accepted_schema_versions=accepted_schema_versions,
        )
        return document
    except (OSError, UnicodeError, ValueError, TypeError):
        return None


def _load_legacy_document(
    path: Path | None,
    *,
    expected_source: StoredCacheSource,
) -> _ComplexityCacheDocument | None:
    """Load a legacy store without treating its old source digest as current.

    Legacy data remains read-only and is not trusted by location or label. A
    caller must still compare both the stored fingerprint and semantic payload
    with the currently requested configuration before using an entry.
    """

    return _load_document(
        path,
        expected_source=expected_source,
        accepted_schema_versions=LEGACY_CACHE_SCHEMA_VERSIONS,
    )


def _empty_document() -> _ComplexityCacheDocument:
    return _ComplexityCacheDocument(
        schema_version=CACHE_SCHEMA_VERSION,
        semantics_version=COMPLEXITY_SEMANTICS_VERSION,
        platform_digest=platform_source_digest(),
        entries={},
    )


@contextmanager
def _exclusive_cache_lock(path: Path) -> Iterator[None]:
    """Lock a stable temp-file inode without leaving artifacts in the package."""

    identity = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    lock_directory = Path(tempfile.gettempdir()) / "stockpile-complexity-locks"
    lock_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = lock_directory / f"{identity}.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_write_document(path: Path, document: _ComplexityCacheDocument) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = document.model_dump(mode="json")
    text = json.dumps(
        serializable,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.write(text)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _result_rank(entry: CachedComplexity) -> tuple[int, int, int, int]:
    result = entry.result
    return (
        int(result.exact),
        result.states_visited,
        result.information_sets,
        result.information_set_actions,
    )


class ComplexityCache:
    """Read and update the preset seed and learned complexity stores."""

    def __init__(
        self,
        *,
        seed_path: str | os.PathLike[str] | None = DEFAULT_PRESET_CACHE_PATH,
        learned_path: str | os.PathLike[str] | None = DEFAULT_LEARNED_CACHE_PATH,
        legacy_seed_paths: Sequence[str | os.PathLike[str]] | None = None,
        legacy_learned_paths: Sequence[str | os.PathLike[str]] | None = None,
    ) -> None:
        self.seed_path = Path(seed_path) if seed_path is not None else None
        self.learned_path = Path(learned_path) if learned_path is not None else None

        # Custom stores are isolated by default. Package defaults opt into the
        # bundled legacy locations, while tests and applications can explicitly
        # supply their own migration sources.
        if legacy_seed_paths is None:
            legacy_seed_paths = (
                DEFAULT_LEGACY_PRESET_CACHE_PATHS
                if self.seed_path == DEFAULT_PRESET_CACHE_PATH
                else ()
            )
        if legacy_learned_paths is None:
            legacy_learned_paths = (
                DEFAULT_LEGACY_LEARNED_CACHE_PATHS
                if self.learned_path == DEFAULT_LEARNED_CACHE_PATH
                else ()
            )
        self.legacy_seed_paths = tuple(Path(path) for path in legacy_seed_paths)
        self.legacy_learned_paths = tuple(
            Path(path) for path in legacy_learned_paths
        )

    def lookup(self, configured: platform.ConfiguredGame) -> CachedComplexity | None:
        """Return the strongest valid record for ``configured``, if one exists."""

        if not isinstance(configured, platform.ConfiguredGame):
            raise TypeError("configured must be a ConfiguredGame")
        fingerprint = semantic_fingerprint(configured)
        requested_payload = semantic_rule_payload(configured)
        candidates: list[CachedComplexity] = []
        learned = _load_document(self.learned_path, expected_source="learned")
        if learned is not None and fingerprint in learned.entries:
            entry = learned.entries[fingerprint]
            if entry.semantic_rules == requested_payload:
                candidates.append(entry)
        seed = _load_document(self.seed_path, expected_source="preset")
        if seed is not None and fingerprint in seed.entries:
            entry = seed.entries[fingerprint]
            if entry.semantic_rules == requested_payload:
                candidates.append(entry)

        for source, paths in (
            ("learned", self.legacy_learned_paths),
            ("preset", self.legacy_seed_paths),
        ):
            for path in paths:
                legacy = _load_legacy_document(path, expected_source=source)
                if legacy is None:
                    continue
                for entry in legacy.entries.values():
                    migrated_payload = _migrate_legacy_semantic_payload(
                        entry.semantic_rules,
                        schema_version=legacy.schema_version,
                    )
                    if (
                        migrated_payload == requested_payload
                        and _fingerprint_payload(migrated_payload) == fingerprint
                    ):
                        candidates.append(entry)
        if not candidates:
            return None

        strongest = max(candidates, key=_result_rank)
        adapted_result = strongest.result.model_copy(
            update={"parameters": configured.parameters},
            deep=True,
        )
        return strongest.model_copy(update={"result": adapted_result}, deep=True)

    def save(
        self,
        configured: platform.ConfiguredGame,
        result: platform.InformationSetComplexity,
        *,
        source: StoredCacheSource = "learned",
    ) -> CachedComplexity:
        """Atomically save a traversal, preserving a stronger existing record."""

        if not isinstance(configured, platform.ConfiguredGame):
            raise TypeError("configured must be a ConfiguredGame")
        if not isinstance(result, platform.InformationSetComplexity):
            raise TypeError("result must be an InformationSetComplexity")
        if source not in {"preset", "learned"}:
            raise ValueError("stored cache source must be preset or learned")
        target = self.seed_path if source == "preset" else self.learned_path
        if target is None:
            raise ValueError(f"the {source} cache path is disabled")

        result_configuration = platform.configure_game(result.parameters)
        if semantic_fingerprint(result_configuration) != semantic_fingerprint(
            configured
        ):
            raise ValueError("result parameters do not match the configured game tree")

        digest = platform_source_digest()
        payload = semantic_rule_payload(configured)
        fingerprint = _fingerprint_payload(payload)
        adapted_result = result.model_copy(
            update={"parameters": configured.parameters},
            deep=True,
        )
        candidate = CachedComplexity(
            fingerprint=fingerprint,
            semantic_rules=payload,
            result=adapted_result,
            provenance=ComplexityProvenance(
                source=source,
                generated_at=_now_utc(),
                schema_version=CACHE_SCHEMA_VERSION,
                semantics_version=COMPLEXITY_SEMANTICS_VERSION,
                platform_digest=digest,
            ),
        )

        with _exclusive_cache_lock(target):
            document = _load_document(target, expected_source=source)
            if document is None:
                document = _empty_document()
            entries = dict(document.entries)
            existing = entries.get(fingerprint)
            if existing is None or _result_rank(candidate) >= _result_rank(existing):
                entries[fingerprint] = candidate
            else:
                candidate = existing
            updated = document.model_copy(
                update={
                    "platform_digest": digest,
                    "entries": entries,
                },
                deep=True,
            )
            _atomic_write_document(target, updated)

        returned_result = candidate.result.model_copy(
            update={"parameters": configured.parameters},
            deep=True,
        )
        return candidate.model_copy(update={"result": returned_result}, deep=True)


def default_complexity_cache() -> ComplexityCache:
    """Return a cache using current stores plus exact-match legacy readers."""

    return ComplexityCache()


def compute_complexity_bounds(
    configured: platform.ConfiguredGame,
    result: platform.InformationSetComplexity | None = None,
) -> ComplexityBounds:
    """Compute rigorous bounds without treating a neighbouring game as evidence.

    ``max_game_length`` is the platform's conservative player-decision bound.
    The chance-node bound counts setup draws and, for every round, information
    pairing plus Market draws.  If ``M`` and ``Q`` are the maximum player and
    chance branching factors, every decision infoset is bounded by
    ``Lp * M^Lp * Q^Lc`` and every infoset-action pair by one further factor
    of ``M``.
    """

    if not isinstance(configured, platform.ConfiguredGame):
        raise TypeError("configured must be a ConfiguredGame")
    if result is not None and not isinstance(result, platform.InformationSetComplexity):
        raise TypeError("result must be an InformationSetComplexity or None")
    if result is not None:
        result_configuration = platform.configure_game(result.parameters)
        if semantic_fingerprint(result_configuration) != semantic_fingerprint(
            configured
        ):
            raise ValueError("result parameters do not match the configured game tree")

    rules = configured.rule_set
    player_decisions = rules.max_game_length
    investor_deals = 0
    if rules.investors:
        investor_deals = rules.player_count * (4 if rules.player_count == 2 else 2)
    starting_share_draws = (
        rules.player_count * rules.starting_shares_per_player
    )
    chance_nodes = (
        starting_share_draws
        + 1
        + investor_deals
        + rules.round_count
        * (
            2 * rules.company_count
            + rules.stockpile_count
            + 2 * rules.player_count * rules.supply_batches
        )
    )
    max_actions = max(1, rules.max_legal_actions)
    max_chance = max(1, rules.max_chance_outcomes)

    lower_sets = 0 if result is None else result.information_sets
    lower_actions = 0 if result is None else result.information_set_actions
    exact = bool(result is not None and result.exact)
    if exact:
        return ComplexityBounds(
            exact=True,
            lower_information_sets=lower_sets,
            lower_information_set_actions=lower_actions,
            upper_information_sets=lower_sets,
            upper_information_set_actions=lower_actions,
            upper_information_sets_expression=str(lower_sets),
            upper_information_set_actions_expression=str(lower_actions),
            upper_information_sets_log10=(
                math.log10(lower_sets) if lower_sets > 0 else 0.0
            ),
            upper_information_set_actions_log10=(
                math.log10(lower_actions) if lower_actions > 0 else 0.0
            ),
            player_decision_bound=player_decisions,
            chance_node_bound=chance_nodes,
            max_legal_actions=max_actions,
            max_chance_outcomes=max_chance,
        )

    set_expression = (
        f"{player_decisions} * {max_actions}^{player_decisions} * "
        f"{max_chance}^{chance_nodes}"
    )
    action_expression = (
        f"{player_decisions} * {max_actions}^{player_decisions + 1} * "
        f"{max_chance}^{chance_nodes}"
    )
    set_log10 = (
        (math.log10(player_decisions) if player_decisions > 0 else 0.0)
        + player_decisions * math.log10(max_actions)
        + chance_nodes * math.log10(max_chance)
    )
    return ComplexityBounds(
        exact=False,
        lower_information_sets=lower_sets,
        lower_information_set_actions=lower_actions,
        upper_information_sets=None,
        upper_information_set_actions=None,
        upper_information_sets_expression=set_expression,
        upper_information_set_actions_expression=action_expression,
        upper_information_sets_log10=set_log10,
        upper_information_set_actions_log10=(
            set_log10 + math.log10(max_actions)
        ),
        player_decision_bound=player_decisions,
        chance_node_bound=chance_nodes,
        max_legal_actions=max_actions,
        max_chance_outcomes=max_chance,
    )


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "COMPLEXITY_SEMANTICS_VERSION",
    "DEFAULT_LEGACY_LEARNED_CACHE_PATHS",
    "DEFAULT_LEGACY_PRESET_CACHE_PATHS",
    "DEFAULT_LEARNED_CACHE_PATH",
    "DEFAULT_PRESET_CACHE_PATH",
    "LEGACY_CACHE_SCHEMA_VERSIONS",
    "CachedComplexity",
    "ComplexityBounds",
    "ComplexityCache",
    "ComplexityProvenance",
    "compute_complexity_bounds",
    "default_complexity_cache",
    "live_complexity_provenance",
    "platform_source_digest",
    "semantic_fingerprint",
    "semantic_rule_payload",
]
