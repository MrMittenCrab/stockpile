"""Validated configuration for Stockpile's sampled Deep CFR trainer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence


DEFAULT_CURRICULUM = (1, 2, 3, 4, 6)


def parse_curriculum(value: str | Sequence[int]) -> tuple[int, ...]:
    """Parse and validate a strictly increasing round curriculum."""

    if isinstance(value, str):
        try:
            rounds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        except ValueError as error:
            raise ValueError("curriculum must be a comma-delimited list of rounds") from error
    else:
        rounds = tuple(int(item) for item in value)
    if not rounds:
        raise ValueError("curriculum must contain at least one round")
    if rounds[0] != 1:
        raise ValueError("curriculum must start at round 1")
    if any(round_count < 1 or round_count > 6 for round_count in rounds):
        raise ValueError("Deep CFR curriculum rounds must be between 1 and 6")
    if any(left >= right for left, right in zip(rounds, rounds[1:])):
        raise ValueError("curriculum rounds must be strictly increasing")
    return rounds


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    """The ordered game horizons used for weight-transfer pretraining."""

    rounds: tuple[int, ...] = DEFAULT_CURRICULUM

    def __post_init__(self) -> None:
        object.__setattr__(self, "rounds", parse_curriculum(self.rounds))

    @classmethod
    def for_target(
        cls,
        target_rounds: int,
        requested: str | Sequence[int] | None = None,
    ) -> "CurriculumConfig":
        if requested is not None:
            resolved = parse_curriculum(requested)
        elif target_rounds == 6:
            resolved = DEFAULT_CURRICULUM
        else:
            resolved = tuple(range(1, target_rounds + 1))
        if resolved[-1] != target_rounds:
            raise ValueError("curriculum must end at the configured --rounds value")
        return cls(resolved)


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    """Shape-stable network dimensions shared by every curriculum stage."""

    observation_size: int = 256
    horizon_size: int = 3
    event_size: int = 96
    action_count: int = 18
    observation_hidden: int = 128
    history_hidden: int = 128
    event_hidden: int = 128
    fusion_hidden: int = 128


@dataclass(frozen=True, slots=True)
class DeepCFRConfig:
    """Compute, optimization, persistence, and evaluation settings."""

    curriculum: CurriculumConfig = CurriculumConfig()
    network: NetworkConfig = NetworkConfig()
    iterations_per_stage: int = 100
    traversals_per_player: int = 20
    advantage_train_steps: int = 1
    strategy_train_steps: int = 1
    # Strict history makes every retained sample substantially larger than a
    # flat observation. These conservative defaults keep six-round batches and
    # full-resume checkpoints practical; callers can raise them after a local
    # memory benchmark.
    batch_size: int = 32
    memory_capacity: int = 2_000
    learning_rate: float = 1e-4
    exploration: float = 0.6
    gradient_clip: float = 5.0
    checkpoint_every: int = 10
    evaluation_pairs: int = 100
    learning_curve_pairs: int = 500
    learning_curve_bootstrap_resamples: int = 10_000
    learning_curve_checkpoint_count: int = 10
    until_win_rate: float | None = None
    eval_every_iterations: int | None = None
    eval_games: int | None = None
    max_iterations: int | None = None
    until_win_rate_consecutive: int = 2
    seed: int = 42
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    output_dir: Path = Path("stockpile/artifacts/deep_cfr/default")
    algorithm: Literal["outcome_sampled_deep_cfr_v1"] = (
        "outcome_sampled_deep_cfr_v1"
    )

    def __post_init__(self) -> None:
        positive_integers = {
            "iterations_per_stage": self.iterations_per_stage,
            "traversals_per_player": self.traversals_per_player,
            "advantage_train_steps": self.advantage_train_steps,
            "strategy_train_steps": self.strategy_train_steps,
            "batch_size": self.batch_size,
            "memory_capacity": self.memory_capacity,
            "checkpoint_every": self.checkpoint_every,
            "evaluation_pairs": self.evaluation_pairs,
            "learning_curve_pairs": self.learning_curve_pairs,
            "learning_curve_bootstrap_resamples": (
                self.learning_curve_bootstrap_resamples
            ),
            "learning_curve_checkpoint_count": self.learning_curve_checkpoint_count,
            "until_win_rate_consecutive": self.until_win_rate_consecutive,
        }
        for name, value in positive_integers.items():
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 < self.learning_rate:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.exploration <= 1.0:
            raise ValueError("exploration must be in (0, 1]")
        if not 0.0 < self.gradient_clip:
            raise ValueError("gradient_clip must be positive")
        if self.batch_size > self.memory_capacity:
            raise ValueError("batch_size cannot exceed memory_capacity")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        self._validate_until_win_rate_options()

    def traversals_per_iteration(self) -> int:
        """Return sampled trajectories completed by one full training iteration."""

        return int(self.traversals_per_player) * 2

    def _validate_until_win_rate_options(self) -> None:
        until = self.until_win_rate
        eval_every = self.eval_every_iterations
        eval_games = self.eval_games
        max_iterations = self.max_iterations
        any_until_option = any(
            value is not None
            for value in (until, eval_every, eval_games, max_iterations)
        )
        if until is None:
            if any_until_option:
                raise ValueError(
                    "eval-every, eval-games, and max-iterations require --until-win-rate"
                )
            return
        if isinstance(until, bool) or not isinstance(until, (int, float)):
            raise ValueError("until_win_rate must be a number")
        until = float(until)
        if not 0.0 < until <= 1.0:
            raise ValueError("until_win_rate must be in (0, 1]")
        object.__setattr__(self, "until_win_rate", until)

        if eval_every is None:
            eval_every = 100
        if (
            isinstance(eval_every, bool)
            or not isinstance(eval_every, int)
            or eval_every < 1
        ):
            raise ValueError("eval_every_iterations must be a positive integer")
        object.__setattr__(self, "eval_every_iterations", int(eval_every))

        if eval_games is None:
            eval_games = 2_000
        if (
            isinstance(eval_games, bool)
            or not isinstance(eval_games, int)
            or eval_games < 2
            or eval_games % 2 != 0
        ):
            raise ValueError("eval_games must be a positive even integer")
        object.__setattr__(self, "eval_games", int(eval_games))

        if max_iterations is None:
            max_iterations = 10_000
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        if max_iterations < int(eval_every):
            raise ValueError(
                "max_iterations must be at least eval_every_iterations "
                f"({int(eval_every)})"
            )
        object.__setattr__(self, "max_iterations", int(max_iterations))

    @property
    def until_win_rate_enabled(self) -> bool:
        return self.until_win_rate is not None

    @property
    def learning_curve_evaluation_pairs(self) -> int:
        """Pairs used for in-training learning-curve / until-win-rate evaluation."""

        if self.eval_games is not None:
            return int(self.eval_games) // 2
        return int(self.learning_curve_pairs)

    @classmethod
    def smoke(
        cls,
        *,
        output_dir: str | Path,
        seed: int = 42,
    ) -> "DeepCFRConfig":
        """Return the fixed one-round end-to-end validation preset."""

        return cls(
            curriculum=CurriculumConfig((1,)),
            iterations_per_stage=2,
            traversals_per_player=2,
            advantage_train_steps=1,
            strategy_train_steps=1,
            batch_size=32,
            memory_capacity=512,
            checkpoint_every=1,
            evaluation_pairs=8,
            learning_curve_pairs=2,
            learning_curve_bootstrap_resamples=32,
            learning_curve_checkpoint_count=10,
            seed=seed,
            device="cpu",
            output_dir=Path(output_dir),
        )

    def with_curriculum(self, curriculum: CurriculumConfig) -> "DeepCFRConfig":
        return replace(self, curriculum=curriculum)


__all__ = [
    "DEFAULT_CURRICULUM",
    "CurriculumConfig",
    "DeepCFRConfig",
    "NetworkConfig",
    "parse_curriculum",
]
