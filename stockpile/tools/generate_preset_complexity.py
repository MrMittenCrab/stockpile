"""Generate the versioned Lite/Classic/Deluxe complexity seed.

Run from the project root with::

    python -m stockpile.tools.generate_preset_complexity

Every tree is traversed through the public platform algorithm.  Compact and
shared policy-head configurations are verified to have the same semantic
fingerprint before one traversal is reused between them.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import tempfile
from typing import Literal, Sequence

from .. import stockpile_platform as platform
from ..complexity_cache import (
    DEFAULT_LEGACY_PRESET_CACHE_PATHS,
    DEFAULT_PRESET_CACHE_PATH,
    CachedComplexity,
    ComplexityCache,
    _exclusive_cache_lock,
    _load_document,
    semantic_fingerprint,
)


PRESET_MATRIX: tuple[tuple[str, int], ...] = (
    ("lite", 2),
    ("lite", 3),
    ("lite", 4),
    ("lite", 5),
    ("classic", 2),
    ("classic", 3),
    ("classic", 4),
    ("classic", 5),
    ("deluxe", 2),
    ("deluxe", 3),
    ("deluxe", 4),
    ("deluxe", 5),
)


def _configured_preset(
    profile: str,
    player_count: int,
    action_space_mode: Literal["compact", "shared"],
) -> platform.ConfiguredGame:
    parameters = platform.get_parameter_preset(
        profile,
        player_count=player_count,
        round_count=6,
        action_space_mode=action_space_mode,
    )
    return platform.configure_game(parameters)


def generate_preset_complexity(
    output_path: str | Path = DEFAULT_PRESET_CACHE_PATH,
    *,
    max_states: int = 10_000,
    max_seconds: float = 120.0,
    resume: bool = True,
    legacy_paths: Sequence[str | Path] | None = None,
) -> dict[str, CachedComplexity]:
    """Calculate and persist the twelve canonical six-round preset trees.

    A complete valid seed whose entries meet ``max_states`` is returned
    without rewriting it.  Partial seeds are reused where possible, rebuilt
    into a clean staging document, validated, and atomically promoted.  Thus a
    failed or time-truncated generation never leaves a partial package seed.
    """

    if isinstance(max_states, bool) or max_states <= 0:
        raise ValueError("max_states must be positive")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be positive and finite")

    output = Path(output_path)
    if legacy_paths is None and output == DEFAULT_PRESET_CACHE_PATH:
        legacy_paths = DEFAULT_LEGACY_PRESET_CACHE_PATHS
    output_cache = ComplexityCache(
        seed_path=output,
        learned_path=None,
        legacy_seed_paths=() if legacy_paths is None else legacy_paths,
        legacy_learned_paths=(),
    )
    existing_document = (
        _load_document(output, expected_source="preset") if resume else None
    )
    configurations: list[
        tuple[str, platform.ConfiguredGame, platform.ConfiguredGame, str]
    ] = []
    for profile, player_count in PRESET_MATRIX:
        compact = _configured_preset(
            profile,
            player_count,
            "compact",
        )
        shared = _configured_preset(
            profile,
            player_count,
            "shared",
        )
        compact_fingerprint = semantic_fingerprint(compact)
        if semantic_fingerprint(shared) != compact_fingerprint:
            raise RuntimeError(
                f"{profile}/{player_count} compact and shared trees differ"
            )
        key = f"{profile}:{player_count}"
        configurations.append((key, compact, shared, compact_fingerprint))

    expected_fingerprints = {item[3] for item in configurations}
    if len(expected_fingerprints) != len(PRESET_MATRIX):
        raise RuntimeError("canonical preset matrix contains duplicate game trees")

    def qualifies(cached: CachedComplexity | None) -> bool:
        if cached is None:
            return False
        result = cached.result
        return result.exact or (
            result.truncation_reason == "max_states"
            and result.states_visited >= max_states
        )

    existing_entries = (
        existing_document.entries if existing_document is not None else {}
    )
    complete = (
        set(existing_entries) == expected_fingerprints
        and all(qualifies(existing_entries.get(item[3])) for item in configurations)
    )
    if complete:
        unchanged: dict[str, CachedComplexity] = {}
        for key, compact, shared, fingerprint in configurations:
            compact_hit = output_cache.lookup(compact)
            shared_hit = output_cache.lookup(shared)
            if (
                compact_hit is None
                or shared_hit is None
                or compact_hit.fingerprint != fingerprint
                or shared_hit.fingerprint != fingerprint
            ):
                raise RuntimeError(f"{key} resume verification failed")
            unchanged[key] = compact_hit
        return unchanged

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".staging",
    )
    os.close(descriptor)
    staging_path = Path(staging_name)
    staging_path.unlink()
    staging_cache = ComplexityCache(seed_path=staging_path, learned_path=None)

    try:
        for key, compact, shared, compact_fingerprint in configurations:
            reusable = existing_entries.get(compact_fingerprint)
            if not qualifies(reusable):
                reusable = output_cache.lookup(compact)
            if qualifies(reusable):
                assert reusable is not None
                result = reusable.result
            else:
                result = platform.compute_information_set_complexity(
                    compact,
                    max_states=max_states,
                    max_seconds=max_seconds,
                    require_exact=False,
                )
                if not result.exact and not (
                    result.truncation_reason == "max_states"
                    and result.states_visited == max_states
                ):
                    raise RuntimeError(
                        f"{key} did not reach its fixed {max_states}-state budget; "
                        f"truncated by {result.truncation_reason or 'unknown'} after "
                        f"{result.states_visited} states"
                    )

            stored = staging_cache.save(compact, result, source="preset")
            shared_hit = staging_cache.lookup(shared)
            if (
                shared_hit is None
                or shared_hit.fingerprint != compact_fingerprint
                or shared_hit.result.information_sets
                != stored.result.information_sets
                or shared_hit.result.information_set_actions
                != stored.result.information_set_actions
            ):
                raise RuntimeError(f"{key} shared cache verification failed")

        staged_document = _load_document(staging_path, expected_source="preset")
        if (
            staged_document is None
            or set(staged_document.entries) != expected_fingerprints
            or len(staged_document.entries) != len(PRESET_MATRIX)
        ):
            raise RuntimeError("generated preset cache failed final validation")

        with _exclusive_cache_lock(output):
            os.replace(staging_path, output)

        final_document = _load_document(output, expected_source="preset")
        if (
            final_document is None
            or set(final_document.entries) != expected_fingerprints
        ):
            raise RuntimeError("promoted preset cache failed final validation")

        # Re-read from the final path so returned values are exactly those that
        # terminal clients will subsequently resolve.
        verified: dict[str, CachedComplexity] = {}
        for key, compact, shared, fingerprint in configurations:
            compact_hit = output_cache.lookup(compact)
            shared_hit = output_cache.lookup(shared)
            if (
                compact_hit is None
                or shared_hit is None
                or compact_hit.fingerprint != fingerprint
                or shared_hit.fingerprint != fingerprint
            ):
                raise RuntimeError(f"{key} promoted lookup verification failed")
            verified[key] = compact_hit
        return verified
    finally:
        if staging_path.exists():
            staging_path.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Stockpile's preset information-set complexity cache."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PRESET_CACHE_PATH,
        help="destination JSON file",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=10_000,
        help="fixed traversal-state budget per unique preset (default: 10000)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=120.0,
        help="safety timeout per unique preset (default: 120)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recalculate every preset even when the existing seed is deep enough",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    generated = generate_preset_complexity(
        arguments.output,
        max_states=arguments.max_states,
        max_seconds=arguments.max_seconds,
        resume=not arguments.force,
    )
    print(f"Wrote {len(generated)} unique preset trees to {arguments.output}")
    for key, cached in generated.items():
        result = cached.result
        qualifier = "=" if result.exact else ">="
        print(
            f"  {key}: infosets {qualifier} {result.information_sets:,}; "
            f"infoset-actions {qualifier} {result.information_set_actions:,}; "
            f"states {result.states_visited:,}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PRESET_MATRIX", "generate_preset_complexity", "main"]
