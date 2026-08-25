"""Torch-free discovery and reservation of Deep CFR artifact runs.

Managed runs live below the package ``artifacts/deep_cfr`` directory.
Normal runs use ``<mode>/run_XX`` while smoke runs share the separate
``smoke/run_XX`` namespace.  A small versioned ``run.json`` file makes a
reserved directory discoverable without loading an untrusted PyTorch
checkpoint.

The historical ``default`` and ``smoke`` directories remain read-only legacy
sources.  Discovery never moves, registers, or modifies them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Literal, Mapping
import uuid


RUN_SCHEMA_VERSION = 1
RUN_MANIFEST_NAME = "run.json"
RESUME_PROVENANCE_SCHEMA_VERSION = 1
RESUME_PROVENANCE_NAME = "resume_provenance.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "deep_cfr"

RunKind = Literal["normal", "smoke"]
ArtifactSource = Literal["managed", "unmanaged", "legacy"]
RunState = Literal["reserved", "active", "completed"]

_RUN_NAME = re.compile(r"^run_([0-9]{2,})$")
_MODE_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
_ROUND_NAME = re.compile(r"^round_[0-9]{2,}$")
_RESERVED_MODE_DIRECTORIES = frozenset({"default", "smoke"})
_MAX_JSON_BYTES = 1024 * 1024
_RUN_STATES: tuple[RunState, ...] = ("reserved", "active", "completed")
_SHA256_CHUNK_SIZE = 1024 * 1024
_CURRENT_CHECKPOINT_SCHEMA_VERSION = 2
_CURRENT_REGRET_RECORD_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RunRef:
    """One managed, unmanaged, or historical artifact directory."""

    path: Path
    mode: str
    kind: RunKind
    source: ArtifactSource
    run: int | None
    manifest: Path | None
    run_id: str | None = None
    state: RunState | None = None
    created_at: str | None = None
    resume_provenance: Mapping[str, object] | None = None
    manifest_data: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).resolve(strict=False))
        object.__setattr__(
            self,
            "manifest",
            None
            if self.manifest is None
            else Path(self.manifest).resolve(strict=False),
        )
        _validate_mode(self.mode)
        if self.kind not in {"normal", "smoke"}:
            raise ValueError(f"unknown artifact run kind: {self.kind!r}")
        if self.source not in {"managed", "unmanaged", "legacy"}:
            raise ValueError(f"unknown artifact source: {self.source!r}")
        if self.source == "managed":
            _validate_run_number(self.run)
            if self.manifest is None:
                raise ValueError("managed runs require a manifest path")
            _validate_run_id(self.run_id)
            _validate_run_state(self.state)
            _validate_created_at(self.created_at)
            _validate_provenance(self.resume_provenance)
            if self.manifest_data is None:
                raise ValueError("managed runs require retained manifest data")
        elif self.run is not None or self.manifest is not None:
            raise ValueError("only managed runs have a run number and manifest")
        elif any(
            value is not None
            for value in (
                self.run_id,
                self.state,
                self.created_at,
                self.resume_provenance,
                self.manifest_data,
            )
        ):
            raise ValueError("only managed runs have manifest metadata")

    @property
    def output_dir(self) -> Path:
        return self.path

    @property
    def smoke(self) -> bool:
        return self.kind == "smoke"

    @property
    def managed(self) -> bool:
        return self.source == "managed"

    @property
    def legacy(self) -> bool:
        return self.source == "legacy"


@dataclass(frozen=True, slots=True)
class ResumePlan:
    """A stable checkpoint source and the directory a resume may write."""

    checkpoint: Path
    source: RunRef
    destination: RunRef
    fork: bool
    provenance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checkpoint",
            Path(self.checkpoint).resolve(strict=False),
        )

    @property
    def output_dir(self) -> Path:
        return self.destination.path

    @property
    def in_place(self) -> bool:
        return not self.fork


def default_artifact_root() -> Path:
    """Return the project-root managed artifact directory."""

    return DEFAULT_ARTIFACT_ROOT


def _raw_artifact_root(value: str | Path | None) -> Path:
    return DEFAULT_ARTIFACT_ROOT if value is None else Path(value).expanduser()


def _artifact_root(
    value: str | Path | None,
    *,
    reject_symlink: bool = False,
) -> Path:
    raw = _raw_artifact_root(value)
    if reject_symlink and raw.is_symlink():
        raise ValueError(f"artifact root cannot be a symlink: {raw}")
    return raw.resolve(strict=False)


def _resolve_explicit_path(value: str | Path, *, label: str) -> Path:
    raw = Path(value).expanduser()
    if raw.is_symlink():
        raise ValueError(f"{label} cannot select a symlink alias: {raw}")
    return raw.resolve(strict=False)


def _validate_mode(mode: object) -> str:
    if not isinstance(mode, str) or not _MODE_NAME.fullmatch(mode):
        raise ValueError("mode must be a lowercase artifact namespace name")
    if mode in _RESERVED_MODE_DIRECTORIES:
        raise ValueError(f"mode name is reserved for artifacts: {mode}")
    return mode


def _validate_run_number(run: object) -> int:
    if isinstance(run, bool) or not isinstance(run, int) or run < 1:
        raise ValueError("run must be a positive integer")
    return run


def _validate_run_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("managed run_id must be a UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise ValueError("managed run_id must be a UUID") from error
    if parsed.version != 4 or str(parsed) != value.lower():
        raise ValueError("managed run_id must be a canonical UUID4")
    return value.lower()


def _validate_run_state(value: object) -> RunState:
    if value not in _RUN_STATES:
        raise ValueError(f"unknown managed run state: {value!r}")
    return value  # type: ignore[return-value]


def _validate_created_at(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("managed run created_at must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("managed run created_at must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("managed run created_at must include a timezone")
    return value


def _validate_provenance(
    value: object,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("resume provenance must be a JSON object")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("resume provenance must be JSON serializable") from error
    return dict(value)


def run_name(run: int) -> str:
    """Format a positive run index with a minimum two-digit width."""

    return f"run_{_validate_run_number(run):02d}"


def parse_run_name(name: str) -> int | None:
    """Return a canonical run index, or ``None`` for unrelated names."""

    matched = _RUN_NAME.fullmatch(name)
    if matched is None:
        return None
    run = int(matched.group(1))
    if run < 1 or name != run_name(run):
        return None
    return run


def managed_parent(
    mode: str,
    *,
    smoke: bool = False,
    artifact_root: str | Path | None = None,
) -> Path:
    """Return the namespace directory for one managed run kind."""

    mode = _validate_mode(mode)
    root = _artifact_root(artifact_root, reject_symlink=True)
    return root / ("smoke" if smoke else mode)


def _manifest_document(ref: RunRef) -> dict[str, object]:
    assert ref.manifest_data is not None
    return dict(ref.manifest_data)


def _write_manifest(ref: RunRef) -> None:
    assert ref.manifest is not None
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".run.",
        suffix=".tmp",
        dir=ref.path,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                _manifest_document(ref),
                stream,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, ref.manifest)
    finally:
        if temporary.exists():
            temporary.unlink()


def _matching_run_numbers(parent: Path) -> tuple[int, ...]:
    if not parent.is_dir():
        return ()
    numbers: list[int] = []
    for entry in parent.iterdir():
        parsed = parse_run_name(entry.name)
        if parsed is not None:
            numbers.append(parsed)
    return tuple(numbers)


def reserve_run(
    mode: str,
    *,
    run: int | None = None,
    smoke: bool = False,
    artifact_root: str | Path | None = None,
) -> RunRef:
    """Atomically reserve and describe a new managed run directory.

    Automatic allocation uses the largest occupied canonical name plus one.
    Directory creation is the uniqueness primitive: concurrent allocators that
    collide rescan and retry, while an explicit collision is always an error.
    """

    mode = _validate_mode(mode)
    if run is not None:
        _validate_run_number(run)
    root = _artifact_root(artifact_root, reject_symlink=True)
    if root.is_symlink():
        raise ValueError(f"artifact root cannot be a symlink: {root}")
    parent = managed_parent(mode, smoke=smoke, artifact_root=root)
    if parent.is_symlink():
        raise ValueError(f"artifact namespace cannot be a symlink: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError(f"artifact root cannot be a symlink: {root}")
    if parent.is_symlink():
        raise ValueError(f"artifact namespace cannot be a symlink: {parent}")

    while True:
        selected = (
            run
            if run is not None
            else max(_matching_run_numbers(parent), default=0) + 1
        )
        assert selected is not None
        path = parent / run_name(selected)
        try:
            path.mkdir()
        except FileExistsError as error:
            if run is not None:
                raise FileExistsError(
                    f"artifact run already exists: {path}"
                ) from error
            continue

        run_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        document = {
            "schema_version": RUN_SCHEMA_VERSION,
            "source": "managed",
            "mode": mode,
            "kind": "smoke" if smoke else "normal",
            "run": selected,
            "run_id": run_id,
            "state": "reserved",
            "created_at": created_at,
            "resume_provenance": None,
        }
        ref = RunRef(
            path=path,
            mode=mode,
            kind="smoke" if smoke else "normal",
            source="managed",
            run=selected,
            manifest=path / RUN_MANIFEST_NAME,
            run_id=run_id,
            state="reserved",
            created_at=created_at,
            resume_provenance=None,
            manifest_data=document,
        )
        try:
            _write_manifest(ref)
        except BaseException:
            # This process created the directory and manifest is its first
            # durable payload.  Remove only that still-empty reservation on a
            # failed manifest write; if anything else appeared, leave it
            # untouched and let future allocation treat the name as occupied.
            try:
                path.rmdir()
            except OSError:
                pass
            raise
        if run is not None and ref.run != run:
            raise RuntimeError(
                f"explicit run {run} resolved to unexpected run {ref.run}"
            )
        return ref


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        if path.stat().st_size > _MAX_JSON_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _managed_layout(
    path: Path,
    root: Path,
) -> tuple[str | None, RunKind, int] | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if len(relative.parts) != 2:
        return None
    namespace, name = relative.parts
    run = parse_run_name(name)
    if run is None:
        return None
    if namespace == "smoke":
        return None, "smoke", run
    try:
        mode = _validate_mode(namespace)
    except ValueError:
        return None
    return mode, "normal", run


def read_run(
    path: str | Path,
    *,
    artifact_root: str | Path | None = None,
) -> RunRef | None:
    """Read one valid v1 managed run without loading model artifacts."""

    root = _artifact_root(artifact_root, reject_symlink=True)
    raw = Path(path).expanduser()
    if raw.is_symlink() or raw.parent.is_symlink():
        return None
    resolved = raw.resolve(strict=False)
    layout = _managed_layout(resolved, root)
    if layout is None or not resolved.is_dir():
        return None
    layout_mode, layout_kind, layout_run = layout
    manifest = resolved / RUN_MANIFEST_NAME
    document = _read_json(manifest)
    if document is None:
        return None
    if document.get("schema_version") != RUN_SCHEMA_VERSION:
        return None
    if document.get("source") != "managed":
        return None
    mode = document.get("mode")
    kind = document.get("kind")
    run = document.get("run")
    run_id = document.get("run_id")
    state = document.get("state")
    created_at = document.get("created_at")
    resume_provenance = document.get("resume_provenance")
    try:
        mode = _validate_mode(mode)
        run = _validate_run_number(run)
        run_id = _validate_run_id(run_id)
        state = _validate_run_state(state)
        created_at = _validate_created_at(created_at)
        resume_provenance = _validate_provenance(resume_provenance)
    except ValueError:
        return None
    if kind not in {"normal", "smoke"}:
        return None
    if run != layout_run or kind != layout_kind:
        return None
    if layout_mode is not None and mode != layout_mode:
        return None
    return RunRef(
        path=resolved,
        mode=mode,
        kind=kind,
        source="managed",
        run=run,
        manifest=manifest,
        run_id=run_id,
        state=state,
        created_at=created_at,
        resume_provenance=resume_provenance,
        manifest_data=dict(document),
    )


def update_run_manifest(
    ref: RunRef,
    *,
    state: RunState | None = None,
    provenance: Mapping[str, object] | None = None,
) -> RunRef:
    """Atomically advance lifecycle state or attach resume provenance."""

    if not ref.managed or ref.manifest is None:
        raise ValueError("only managed runs have an updatable manifest")
    document = _read_json(ref.manifest)
    if document is None:
        raise ValueError(f"managed run manifest is missing or malformed: {ref.manifest}")
    identity = {
        "schema_version": RUN_SCHEMA_VERSION,
        "source": "managed",
        "mode": ref.mode,
        "kind": ref.kind,
        "run": ref.run,
        "run_id": ref.run_id,
        "created_at": ref.created_at,
    }
    if any(document.get(name) != value for name, value in identity.items()):
        raise ValueError("managed run manifest identity changed on disk")

    current_state = _validate_run_state(document.get("state"))
    next_state = current_state if state is None else _validate_run_state(state)
    if _RUN_STATES.index(next_state) < _RUN_STATES.index(current_state):
        raise ValueError(
            f"run state cannot move backward from {current_state} to {next_state}"
        )
    current_provenance = _validate_provenance(
        document.get("resume_provenance")
    )
    next_provenance = (
        current_provenance
        if provenance is None
        else _validate_provenance(provenance)
    )

    updated_document = dict(document)
    updated_document["state"] = next_state
    if next_provenance is not None:
        updated_document["resume_provenance"] = dict(next_provenance)
    updated = RunRef(
        path=ref.path,
        mode=ref.mode,
        kind=ref.kind,
        source="managed",
        run=ref.run,
        manifest=ref.manifest,
        run_id=ref.run_id,
        state=next_state,
        created_at=ref.created_at,
        resume_provenance=next_provenance,
        manifest_data=updated_document,
    )
    _write_manifest(updated)
    return updated


def _saved_mode(path: Path) -> str | None:
    document = _read_json(path / "config.json")
    if document is None:
        return None
    base_game = document.get("base_game")
    if not isinstance(base_game, Mapping):
        return None
    mode = base_game.get("mode")
    try:
        return _validate_mode(mode)
    except ValueError:
        return None


def _config_marks_current_v2(path: Path) -> bool:
    """Whether an unmanaged run's additive config marks current artifacts."""

    document = _read_json(path / "config.json")
    if document is None:
        return False
    telemetry = document.get("sampled_regret_telemetry")
    if isinstance(telemetry, Mapping) and telemetry.get(
        "record_schema_version"
    ) == _CURRENT_REGRET_RECORD_SCHEMA_VERSION:
        return True

    # Accept explicit additive version markers as well.  The telemetry marker
    # is what current Stockpile writes; the named schema markers keep the
    # resolver forward-compatible with config-only tooling.
    version_keys = (
        "artifact_schema_version",
        "checkpoint_schema_version",
        "checkpoint_schema",
    )
    if any(
        document.get(key) == _CURRENT_CHECKPOINT_SCHEMA_VERSION
        for key in version_keys
    ):
        return True
    training = document.get("training")
    return isinstance(training, Mapping) and any(
        training.get(key) == _CURRENT_CHECKPOINT_SCHEMA_VERSION
        for key in version_keys
    )


def discover_legacy_runs(
    *,
    mode: str | None = None,
    smoke: bool | None = None,
    artifact_root: str | Path | None = None,
) -> tuple[RunRef, ...]:
    """Discover historical fixed directories without writing to them."""

    if mode is not None:
        mode = _validate_mode(mode)
    root = _artifact_root(artifact_root, reject_symlink=True)
    candidates: list[tuple[Path, RunKind]] = []
    if smoke in {None, False}:
        candidates.append((root / "default", "normal"))
    if smoke in {None, True}:
        candidates.append((root / "smoke", "smoke"))

    runs: list[RunRef] = []
    for path, kind in candidates:
        if path.is_symlink() or not path.is_dir():
            continue
        saved_mode = _saved_mode(path)
        if saved_mode is None or (mode is not None and saved_mode != mode):
            continue
        runs.append(
            RunRef(
                path=path,
                mode=saved_mode,
                kind=kind,
                source="legacy",
                run=None,
                manifest=None,
            )
        )
    return tuple(runs)


def _managed_runs(
    *,
    mode: str | None,
    smoke: bool | None,
    artifact_root: Path,
) -> list[RunRef]:
    parents: list[Path] = []
    if smoke in {None, False}:
        if mode is not None:
            parents.append(artifact_root / mode)
        elif artifact_root.is_dir():
            for entry in artifact_root.iterdir():
                if entry.name in _RESERVED_MODE_DIRECTORIES:
                    continue
                try:
                    _validate_mode(entry.name)
                except ValueError:
                    continue
                if entry.is_dir() and not entry.is_symlink():
                    parents.append(entry)
    if smoke in {None, True}:
        parents.append(artifact_root / "smoke")

    runs: list[RunRef] = []
    for parent in parents:
        if parent.is_symlink() or not parent.is_dir():
            continue
        for entry in parent.iterdir():
            if parse_run_name(entry.name) is None:
                continue
            ref = read_run(entry, artifact_root=artifact_root)
            if ref is None:
                continue
            if mode is not None and ref.mode != mode:
                continue
            runs.append(ref)
    return runs


def discover_runs(
    *,
    mode: str | None = None,
    smoke: bool | None = None,
    artifact_root: str | Path | None = None,
    include_legacy: bool = True,
) -> tuple[RunRef, ...]:
    """Return deterministic managed and optionally legacy run references."""

    if mode is not None:
        mode = _validate_mode(mode)
    root = _artifact_root(artifact_root, reject_symlink=True)
    runs = _managed_runs(mode=mode, smoke=smoke, artifact_root=root)
    if include_legacy:
        runs.extend(
            discover_legacy_runs(
                mode=mode,
                smoke=smoke,
                artifact_root=root,
            )
        )
    source_order = {"managed": 0, "legacy": 1, "unmanaged": 2}
    runs.sort(
        key=lambda ref: (
            ref.mode,
            ref.kind,
            source_order[ref.source],
            -1 if ref.run is None else ref.run,
            str(ref.path),
        )
    )
    return tuple(runs)


def resolve_run(
    mode: str,
    *,
    run: int | None = None,
    smoke: bool | None = False,
    artifact_root: str | Path | None = None,
    include_legacy: bool = False,
) -> RunRef:
    """Resolve an explicit run or the highest managed run.

    ``smoke=None`` searches both namespaces and rejects an ambiguous explicit
    run number instead of silently preferring one.
    """

    mode = _validate_mode(mode)
    root = _artifact_root(artifact_root, reject_symlink=True)
    if run is not None:
        run = _validate_run_number(run)
        kinds = (False, True) if smoke is None else (smoke,)
        matches: list[RunRef] = []
        searched: list[Path] = []
        for is_smoke in kinds:
            path = (
                managed_parent(mode, smoke=is_smoke, artifact_root=root)
                / run_name(run)
            )
            searched.append(path)
            ref = read_run(path, artifact_root=root)
            if (
                ref is not None
                and ref.mode == mode
                and ref.smoke == is_smoke
                and ref.run == run
            ):
                matches.append(ref)
        if len(matches) > 1:
            raise ValueError(
                f"run {run} is ambiguous between normal and smoke artifacts; "
                "select an explicit --output-dir"
            )
        if matches:
            return matches[0]
        raise FileNotFoundError(
            "artifact run does not exist or is invalid: "
            + ", ".join(str(path) for path in searched)
        )

    managed = [
        ref
        for ref in discover_runs(
            mode=mode,
            smoke=smoke,
            artifact_root=root,
            include_legacy=False,
        )
        if ref.managed
    ]
    if managed:
        maximum = max(ref.run or 0 for ref in managed)
        latest = [ref for ref in managed if ref.run == maximum]
        if len(latest) > 1:
            raise ValueError(
                f"latest run {maximum} is ambiguous between normal and smoke "
                "artifacts"
            )
        return latest[0]
    if include_legacy:
        legacy = discover_legacy_runs(
            mode=mode,
            smoke=smoke,
            artifact_root=root,
        )
        if len(legacy) > 1:
            raise ValueError("legacy normal and smoke artifacts are ambiguous")
        if legacy:
            return legacy[0]
    namespace = "normal or smoke" if smoke is None else ("smoke" if smoke else mode)
    raise FileNotFoundError(f"no artifact runs found for {namespace}")


def _legacy_ref_for_path(path: Path, root: Path) -> RunRef | None:
    for ref in discover_legacy_runs(artifact_root=root):
        if path == ref.path or path.is_relative_to(ref.path):
            return ref
    return None


def _nearest_saved_mode(path: Path) -> tuple[Path, str] | None:
    start = path if path.is_dir() else path.parent
    for directory in (start, *start.parents):
        saved = _saved_mode(directory)
        if saved is not None:
            return directory, saved
    return None


def find_run_for_path(
    path: str | Path,
    *,
    mode: str | None = None,
    smoke: bool = False,
    artifact_root: str | Path | None = None,
) -> RunRef | None:
    """Identify the provenance root containing an artifact path."""

    if mode is not None:
        mode = _validate_mode(mode)
    root = _artifact_root(artifact_root, reject_symlink=True)
    raw = Path(path).expanduser()
    if raw.is_symlink():
        return None
    resolved = raw.resolve(strict=False)
    start = resolved if resolved.is_dir() else resolved.parent
    for directory in (start, *start.parents):
        ref = read_run(directory, artifact_root=root)
        if ref is not None:
            if mode is not None and ref.mode != mode:
                return None
            return ref
        if directory == root:
            break

    legacy = _legacy_ref_for_path(resolved, root)
    if legacy is not None:
        if mode is not None and legacy.mode != mode:
            return None
        return legacy

    saved = _nearest_saved_mode(resolved)
    if saved is not None:
        output_dir, saved_mode = saved
        if mode is not None and saved_mode != mode:
            return None
        return RunRef(
            path=output_dir,
            mode=saved_mode,
            kind="smoke" if smoke else "normal",
            source="unmanaged",
            run=None,
            manifest=None,
        )
    if mode is None:
        return None
    output_dir = (
        resolved.parent.parent
        if _ROUND_NAME.fullmatch(resolved.parent.name)
        else start
    )
    return RunRef(
        path=output_dir,
        mode=mode,
        kind="smoke" if smoke else "normal",
        source="unmanaged",
        run=None,
        manifest=None,
    )


def provenance_dict(ref: RunRef) -> dict[str, object]:
    """Return a JSON-safe description for logs and command output."""

    return {
        "source": ref.source,
        "mode": ref.mode,
        "kind": ref.kind,
        "run": ref.run,
        "run_id": ref.run_id,
        "state": ref.state,
        "created_at": ref.created_at,
        "path": str(ref.path),
        "manifest_schema": RUN_SCHEMA_VERSION if ref.managed else None,
        "resume_provenance": (
            None
            if ref.resume_provenance is None
            else dict(ref.resume_provenance)
        ),
    }


def checkpoint_sha256(path: str | Path) -> str:
    """Hash one stable checkpoint without interpreting its pickle payload."""

    checkpoint = _checkpoint_path(path)
    before = checkpoint.stat()
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        while block := stream.read(_SHA256_CHUNK_SIZE):
            digest.update(block)
    after = checkpoint.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise RuntimeError("checkpoint changed while its provenance was recorded")
    return digest.hexdigest()


def resume_provenance(
    checkpoint: str | Path,
    source: RunRef,
) -> dict[str, object]:
    """Describe the immutable source of a forked training run."""

    checkpoint_path = _checkpoint_path(checkpoint)
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "source": source.source,
        "source_path": str(source.path),
        "source_mode": source.mode,
        "source_kind": source.kind,
        "source_run": source.run,
        "source_run_id": source.run_id,
        "legacy": source.legacy,
    }


def _write_unmanaged_resume_provenance(
    destination: RunRef,
    provenance: Mapping[str, object],
) -> None:
    """Atomically persist fork provenance beside an unmanaged destination."""

    if destination.source != "unmanaged":
        raise ValueError("unmanaged provenance requires an unmanaged destination")
    destination.path.mkdir(parents=True, exist_ok=True)
    if destination.path.is_symlink():
        raise ValueError(
            f"resume destination cannot be a symlink: {destination.path}"
        )
    document = {
        "schema_version": RESUME_PROVENANCE_SCHEMA_VERSION,
        "kind": "stockpile_deep_cfr_resume_provenance",
        "resume": dict(provenance),
    }
    target = destination.path / RESUME_PROVENANCE_NAME
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".resume_provenance.",
        suffix=".tmp",
        dir=destination.path,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                document,
                stream,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _path_classification(
    path: Path,
    *,
    mode: str,
    smoke: bool,
    root: Path,
) -> RunRef:
    if path == root or path.is_relative_to(root):
        raise ValueError(
            "--output-dir must be an explicit unmanaged path outside the "
            "managed artifact root; use --run for managed runs and do not "
            "write into legacy artifacts"
        )
    return RunRef(
        path=path,
        mode=mode,
        kind="smoke" if smoke else "normal",
        source="unmanaged",
        run=None,
        manifest=None,
    )


def _validate_unmanaged_output(
    ref: RunRef,
    *,
    overwrite: bool,
) -> None:
    if ref.source != "unmanaged":
        raise ValueError(
            "--output-dir must be an unmanaged path; use --run for managed runs "
            "and do not overwrite legacy artifacts"
        )
    if ref.path.exists() and not ref.path.is_dir():
        raise ValueError(f"output path is not a directory: {ref.path}")
    nonempty = ref.path.is_dir() and any(ref.path.iterdir())
    if nonempty and not overwrite:
        raise ValueError(
            f"output directory is not empty: {ref.path}; pass --overwrite "
            "only with an explicit unmanaged --output-dir"
        )


def resolve_fresh_output(
    mode: str,
    *,
    output_dir: str | Path | None = None,
    run: int | None = None,
    smoke: bool = False,
    overwrite: bool = False,
    artifact_root: str | Path | None = None,
) -> RunRef:
    """Resolve ``solve`` output semantics and reserve managed defaults."""

    mode = _validate_mode(mode)
    root = _artifact_root(artifact_root, reject_symlink=True)
    if output_dir is None:
        if overwrite:
            raise ValueError(
                "--overwrite requires an explicit unmanaged --output-dir"
            )
        return reserve_run(mode, run=run, smoke=smoke, artifact_root=root)
    if run is not None:
        raise ValueError("--run cannot be combined with --output-dir")
    path = _resolve_explicit_path(output_dir, label="--output-dir")
    ref = _path_classification(path, mode=mode, smoke=smoke, root=root)
    _validate_unmanaged_output(ref, overwrite=overwrite)
    return ref


def _checkpoint_path(path: str | Path) -> Path:
    checkpoint = _resolve_explicit_path(path, label="checkpoint")
    if checkpoint.name.endswith(".tmp"):
        raise ValueError("temporary checkpoints cannot be resumed")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    return checkpoint


def plan_resume_destination(
    checkpoint: str | Path,
    *,
    mode: str | None = None,
    output_dir: str | Path | None = None,
    run: int | None = None,
    smoke: bool | None = None,
    overwrite: bool = False,
    artifact_root: str | Path | None = None,
) -> ResumePlan:
    """Plan an in-place resume or an explicit, non-destructive fork.

    Reserved and active managed runs resume in place. Completed managed runs,
    legacy fixed directories, and unmarked unmanaged artifacts fork by
    default. An unmanaged config carrying a current-v2 schema marker may
    resume in place. A distinct explicit destination always forks.
    """

    if mode is not None:
        mode = _validate_mode(mode)
    if output_dir is not None and run is not None:
        raise ValueError("--run cannot be combined with --output-dir")
    if overwrite and output_dir is None:
        raise ValueError(
            "--overwrite requires an explicit unmanaged --output-dir"
        )
    root = _artifact_root(artifact_root, reject_symlink=True)
    requested_output = (
        None
        if output_dir is None
        else _resolve_explicit_path(output_dir, label="--output-dir")
    )
    checkpoint_path = _checkpoint_path(checkpoint)
    checkpoint_parent = checkpoint_path.parent
    managed_ancestor: Path | None = None
    for directory in (checkpoint_parent, *checkpoint_parent.parents):
        if _managed_layout(directory, root) is not None:
            managed_ancestor = directory
            break
        if directory == root:
            break
    if managed_ancestor is not None and read_run(
        managed_ancestor,
        artifact_root=root,
    ) is None:
        raise ValueError("checkpoint is inside a malformed managed run")
    if managed_ancestor is None:
        for legacy_root in (root / "default", root / "smoke"):
            if checkpoint_path.is_relative_to(legacy_root) and (
                _legacy_ref_for_path(checkpoint_path, root) is None
            ):
                raise ValueError("checkpoint is inside malformed legacy artifacts")
    source = find_run_for_path(
        checkpoint_path,
        mode=None,
        smoke=False if smoke is None else smoke,
        artifact_root=root,
    )
    if source is None:
        if mode is None:
            raise ValueError("checkpoint mode cannot be determined; specify --mode")
        source = find_run_for_path(
            checkpoint_path,
            mode=mode,
            smoke=False if smoke is None else smoke,
            artifact_root=root,
        )
        assert source is not None
    if mode is not None and source.mode != mode:
        raise ValueError(
            f"checkpoint mode is {source.mode}, not requested mode {mode}"
        )
    if smoke is not None and source.smoke != smoke:
        raise ValueError("checkpoint smoke provenance does not match the request")

    selector_differs = False
    if requested_output is not None:
        selector_differs = requested_output != source.path
    if run is not None:
        requested_run = _validate_run_number(run)
        selector_differs = selector_differs or (
            not source.managed or source.run != requested_run
        )

    unmanaged_is_legacy_like = (
        source.source == "unmanaged"
        and not _config_marks_current_v2(source.path)
    )
    completed_managed = source.managed and source.state == "completed"
    must_fork = (
        source.legacy
        or unmanaged_is_legacy_like
        or completed_managed
        or selector_differs
    )
    if not must_fork:
        if overwrite:
            raise ValueError("--overwrite is not valid for an in-place resume")
        if requested_output is not None:
            if requested_output != source.path:
                raise AssertionError("different output was not planned as a fork")
        if run is not None:
            if not source.managed or source.run != _validate_run_number(run):
                raise AssertionError("different run was not planned as a fork")
        return ResumePlan(
            checkpoint=checkpoint_path,
            source=source,
            destination=source,
            fork=False,
        )

    destination_smoke = source.smoke if smoke is None else smoke
    # Automatic forks must not consume a numbered reservation merely because
    # hashing the immutable source checkpoint fails or detects concurrent
    # mutation.  Explicit destinations are validated first so a bad path does
    # not trigger an unnecessary large-checkpoint hash.
    provenance: dict[str, object] | None = None
    if output_dir is None:
        provenance = resume_provenance(checkpoint_path, source)
    if output_dir is None:
        destination = reserve_run(
            source.mode,
            run=run,
            smoke=destination_smoke,
            artifact_root=root,
        )
    else:
        assert requested_output is not None
        destination_path = requested_output
        destination = _path_classification(
            destination_path,
            mode=source.mode,
            smoke=destination_smoke,
            root=root,
        )
        _validate_unmanaged_output(destination, overwrite=overwrite)
    if destination.path == source.path:
        raise ValueError(
            "a fork destination must differ from its checkpoint source"
        )
    if destination.source == "unmanaged" and (
        destination.path.is_relative_to(source.path)
        or source.path.is_relative_to(destination.path)
    ):
        raise ValueError(
            "an unmanaged fork destination must not overlap its checkpoint source"
        )
    if run is not None and (
        not destination.managed
        or destination.run != _validate_run_number(run)
    ):
        raise RuntimeError(
            f"explicit run {run} resolved to unexpected run {destination.run}"
        )
    if provenance is None:
        provenance = resume_provenance(checkpoint_path, source)
    if destination.managed:
        destination = update_run_manifest(
            destination,
            provenance=provenance,
        )
    else:
        _write_unmanaged_resume_provenance(destination, provenance)
    return ResumePlan(
        checkpoint=checkpoint_path,
        source=source,
        destination=destination,
        fork=True,
        provenance=provenance,
    )


__all__ = [
    "ArtifactSource",
    "DEFAULT_ARTIFACT_ROOT",
    "PROJECT_ROOT",
    "RUN_MANIFEST_NAME",
    "RUN_SCHEMA_VERSION",
    "RESUME_PROVENANCE_NAME",
    "RESUME_PROVENANCE_SCHEMA_VERSION",
    "ResumePlan",
    "RunKind",
    "RunRef",
    "RunState",
    "checkpoint_sha256",
    "default_artifact_root",
    "discover_legacy_runs",
    "discover_runs",
    "find_run_for_path",
    "managed_parent",
    "parse_run_name",
    "plan_resume_destination",
    "provenance_dict",
    "read_run",
    "reserve_run",
    "resolve_fresh_output",
    "resolve_run",
    "resume_provenance",
    "run_name",
    "update_run_manifest",
]
