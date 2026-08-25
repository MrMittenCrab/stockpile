"""Replaceable, player-scoped policies for the browser's computer seat."""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
from typing import Protocol, Sequence

from .. import stockpile_platform as platform

COMPUTER_POLICY_ENV = "STOCKPILE_COMPUTER_POLICY"
RANDOM_POLICY_TOKEN = "random"


class ComputerPolicy(Protocol):
    """Choose from actions visible in one player's information state only."""

    def choose_action(
        self,
        state: platform.GameState,
        information: platform.InformationState,
        legal_actions: Sequence[platform.LegalAction],
        rng: random.Random,
    ) -> int:
        """Return one action ID from ``legal_actions``."""


class RandomComputerPolicy:
    """Uniform legal-action policy kept as a fallback and test double."""

    def choose_action(
        self,
        state: platform.GameState,
        information: platform.InformationState,
        legal_actions: Sequence[platform.LegalAction],
        rng: random.Random,
    ) -> int:
        del state, information
        if not legal_actions:
            raise ValueError("computer policy requires at least one legal action")
        return int(rng.choice(tuple(legal_actions)).action_id)


class DeepCFRComputerPolicy:
    """Sample the exported Deep CFR average strategy for the computer seat.

    Canonical two-player Lite matches the solve engine. Optional Lite+ rule
    layers fall back to the uniform policy because the trained encoder rejects
    them.
    """

    def __init__(self, policy, *, fallback: ComputerPolicy | None = None) -> None:
        self.policy = policy
        self.fallback = fallback or RandomComputerPolicy()
        self.path = getattr(policy, "metadata", {}).get("source_path")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        fallback: ComputerPolicy | None = None,
    ) -> "DeepCFRComputerPolicy":
        from ..training.policy import DeepCFRPolicy

        checkpoint = Path(path).expanduser().resolve()
        loaded = DeepCFRPolicy.load(checkpoint, device=device)
        loaded.metadata = dict(loaded.metadata)
        loaded.metadata["source_path"] = str(checkpoint)
        return cls(loaded, fallback=fallback)

    def choose_action(
        self,
        state: platform.GameState,
        information: platform.InformationState,
        legal_actions: Sequence[platform.LegalAction],
        rng: random.Random,
    ) -> int:
        if not legal_actions:
            raise ValueError("computer policy requires at least one legal action")
        legal_ids = tuple(int(action.action_id) for action in legal_actions)
        if len(legal_ids) == 1:
            return legal_ids[0]
        try:
            probabilities = self.policy.action_probabilities(
                state,
                int(information.player_id),
            )
        except (TypeError, ValueError, RuntimeError):
            return int(
                self.fallback.choose_action(state, information, legal_actions, rng)
            )
        weights = [float(probabilities.get(action_id, 0.0)) for action_id in legal_ids]
        total = sum(weights)
        if not math.isfinite(total) or total <= 0.0:
            return int(
                self.fallback.choose_action(state, information, legal_actions, rng)
            )
        threshold = rng.random() * total
        cumulative = 0.0
        last_positive = legal_ids[0]
        for action_id, weight in zip(legal_ids, weights):
            if weight > 0.0:
                last_positive = action_id
            cumulative += weight
            if threshold < cumulative:
                return int(action_id)
        return int(last_positive)


def latest_policy_checkpoint(
    run_dir: str | Path, *, rounds: int | None = None
) -> Path:
    """Return a ``round_XX/policy.pt`` under one Deep CFR run directory.

    When ``rounds`` is set, select that exact curriculum stage. Otherwise return
    the highest numbered checkpoint present under the run.
    """

    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Deep CFR run directory does not exist: {root}")
    if rounds is not None:
        checkpoint = root / f"round_{int(rounds):02d}" / "policy.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"no policy.pt checkpoint for round {int(rounds)} under {root}"
            )
        return checkpoint.resolve()
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("round_"):
            continue
        suffix = name[6:]
        if not suffix.isdigit():
            continue
        checkpoint = child / "policy.pt"
        if checkpoint.is_file():
            candidates.append((int(suffix), checkpoint.resolve()))
    if not candidates:
        raise FileNotFoundError(f"no policy.pt checkpoints found under {root}")
    return max(candidates, key=lambda item: item[0])[1]


def resolve_computer_policy_path(
    *,
    policy: str | Path | None = None,
    mode: str = "lite",
    run: int | None = None,
    rounds: int | None = None,
    artifact_root: str | Path | None = None,
) -> Path:
    """Resolve an explicit policy file or a managed-run curriculum checkpoint."""

    if policy is not None:
        path = Path(policy).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"computer policy does not exist: {path}")
        return path
    from ..training.artifacts import resolve_run
    from .v2_schemas import BROWSER_ROUND_COUNT

    target_rounds = BROWSER_ROUND_COUNT if rounds is None else int(rounds)
    ref = resolve_run(mode, run=run, smoke=False, artifact_root=artifact_root)
    return latest_policy_checkpoint(ref.path, rounds=target_rounds)


def load_computer_policy(
    *,
    policy: str | Path | None = None,
    mode: str = "lite",
    run: int | None = None,
    rounds: int | None = None,
    artifact_root: str | Path | None = None,
    device: str = "cpu",
    allow_random_fallback: bool = True,
) -> ComputerPolicy:
    """Load Deep CFR for the computer seat, optionally falling back to random."""

    explicit = os.environ.get(COMPUTER_POLICY_ENV) if policy is None and run is None else None
    if policy is None and run is None and explicit is not None:
        token = explicit.strip()
        if token.casefold() == RANDOM_POLICY_TOKEN:
            return RandomComputerPolicy()
        policy = token
    try:
        checkpoint = resolve_computer_policy_path(
            policy=policy,
            mode=mode,
            run=run,
            rounds=rounds,
            artifact_root=artifact_root,
        )
        return DeepCFRComputerPolicy.load(checkpoint, device=device)
    except Exception:
        if not allow_random_fallback:
            raise
        return RandomComputerPolicy()


RandomPolicy = RandomComputerPolicy


__all__ = [
    "COMPUTER_POLICY_ENV",
    "ComputerPolicy",
    "DeepCFRComputerPolicy",
    "RANDOM_POLICY_TOKEN",
    "RandomComputerPolicy",
    "RandomPolicy",
    "latest_policy_checkpoint",
    "load_computer_policy",
    "resolve_computer_policy_path",
]
