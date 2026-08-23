"""Learning-curve evaluation against a uniform-random benchmark.

Evaluation checkpoints pause training briefly, score the current in-memory
average policy against uniform legal play, then resume without mutating
networks, memories, optimizers, or training RNG state. History is persisted
separately from model-save checkpoints and can regenerate the win-rate graph
without rerunning training or evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import math
from pathlib import Path
import random
from typing import Any

from .evaluation import play_evaluation_game

LEARNING_CURVE_SCHEMA_VERSION = 1
LEARNING_CURVE_KIND = "stockpile_deep_cfr_learning_curve"
LEARNING_CURVE_JSON_NAME = "learning_curve.json"
LEARNING_CURVE_CSV_NAME = "learning_curve.csv"
LEARNING_CURVE_PLOT_NAME = "learning_curve.png"
EVALUATION_HISTORY_CSV_NAME = "evaluation_history.csv"

_OUTCOME_SCORE = {"win": 1.0, "tie": 0.0, "loss": 0.0}

_EVALUATION_HISTORY_FIELDS = (
    "traversals",
    "games",
    "wins",
    "losses",
    "ties",
    "win_rate",
    "mean_utility",
    "ci_low",
    "ci_high",
)


def evaluation_checkpoint_iterations(
    iterations_per_stage: int,
    *,
    checkpoint_count: int = 10,
) -> tuple[int, ...]:
    """Return approximately evenly spaced stage iterations ending at ``I``."""

    if (
        isinstance(iterations_per_stage, bool)
        or not isinstance(iterations_per_stage, int)
        or iterations_per_stage < 1
    ):
        raise ValueError("iterations_per_stage must be a positive integer")
    if (
        isinstance(checkpoint_count, bool)
        or not isinstance(checkpoint_count, int)
        or checkpoint_count < 1
    ):
        raise ValueError("checkpoint_count must be a positive integer")

    points = {
        max(1, min(iterations_per_stage, round(k * iterations_per_stage / checkpoint_count)))
        for k in range(1, checkpoint_count + 1)
    }
    points.add(iterations_per_stage)
    return tuple(sorted(points))


def stage_traversals(stage_iteration: int, traversals_per_player: int) -> int:
    """Return traversals completed within the current stage through ``stage_iteration``."""

    if (
        isinstance(stage_iteration, bool)
        or not isinstance(stage_iteration, int)
        or stage_iteration < 1
    ):
        raise ValueError("stage_iteration must be a positive integer")
    if (
        isinstance(traversals_per_player, bool)
        or not isinstance(traversals_per_player, int)
        or traversals_per_player < 1
    ):
        raise ValueError("traversals_per_player must be a positive integer")
    return int(stage_iteration) * int(traversals_per_player) * 2


def cumulative_traversals(
    *,
    stage_index: int,
    stage_iteration: int,
    iterations_per_stage: int,
    traversals_per_player: int,
) -> int:
    """Return curriculum-wide traversals through the given stage iteration."""

    if isinstance(stage_index, bool) or not isinstance(stage_index, int) or stage_index < 0:
        raise ValueError("stage_index must be a nonnegative integer")
    prior = int(stage_index) * int(iterations_per_stage) * int(traversals_per_player) * 2
    return prior + stage_traversals(stage_iteration, traversals_per_player)


def stage_evaluation_seed(run_seed: int, stage_index: int) -> int:
    """Return the held-out evaluation seed base fixed for one curriculum stage."""

    return int(run_seed) + int(stage_index) * 1_000_000 + 17


def checkpoint_evaluation_seed(
    run_seed: int,
    *,
    stage_index: int,
    stage_iteration: int,
) -> int:
    """Return a fresh reproducible evaluation seed base for one checkpoint."""

    mixed = (
        int(run_seed)
        + int(stage_index) * 1_000_000
        + int(stage_iteration) * 10_000
        + 17
    )
    return mixed & 0x7FFFFFFF


def bootstrap_seed(
    run_seed: int,
    *,
    stage_index: int,
    stage_iteration: int,
) -> int:
    """Derive a deterministic bootstrap RNG seed from run and checkpoint identity."""

    mixed = (
        int(run_seed) * 1_000_003
        + int(stage_index) * 97_367
        + int(stage_iteration) * 1_039
        + 0xC0FFEE
    )
    return mixed & 0xFFFFFFFF


def outcome_score(outcome: str) -> float:
    """Map a single-game outcome onto the win-rate score scale (ties count as 0)."""

    try:
        return _OUTCOME_SCORE[outcome]
    except KeyError as error:
        raise ValueError(f"unknown evaluation outcome: {outcome!r}") from error


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
    lower_percentile: float = 2.5,
    upper_percentile: float = 97.5,
) -> tuple[float, float]:
    """Return percentile CI of the mean by resampling complete units with replacement."""

    if not values:
        raise ValueError("bootstrap requires at least one sampling unit")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if not (0.0 <= lower_percentile < upper_percentile <= 100.0):
        raise ValueError("bootstrap percentiles must satisfy 0 <= lower < upper <= 100")

    population = [float(value) for value in values]
    count = len(population)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(count):
            total += population[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    return (
        _percentile_sorted(means, lower_percentile),
        _percentile_sorted(means, upper_percentile),
    )


def _percentile_sorted(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(sorted_values[low])
    weight = rank - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def evaluate_learning_curve_checkpoint(
    configuration: Any,
    policy: Any,
    *,
    pairs: int,
    evaluation_seed: int,
    bootstrap_resamples: int,
    bootstrap_rng_seed: int,
    round_horizon: int,
    stage_index: int,
    stage_iteration: int,
    global_iteration: int,
    stage_traversal_count: int,
    cumulative_traversal_count: int,
) -> dict[str, Any]:
    """Evaluate the frozen policy with fixed seat-swapped pairs and bootstrap CI."""

    if isinstance(pairs, bool) or not isinstance(pairs, int) or pairs < 1:
        raise ValueError("pairs must be a positive integer")

    pair_scores: list[float] = []
    utilities: list[float] = []
    differentials: list[float] = []
    wins = losses = ties = 0

    for pair_index in range(pairs):
        pair_seed = int(evaluation_seed) + pair_index
        seat_scores: list[float] = []
        for trained_seat in (0, 1):
            game = play_evaluation_game(
                configuration,
                policy,
                trained_seat=trained_seat,
                seed=pair_seed,
            )
            outcome = str(game["outcome"])
            score = outcome_score(outcome)
            seat_scores.append(score)
            utilities.append(float(game["trained_utility"]))
            differentials.append(float(game["final_cash_differential"]))
            if outcome == "win":
                wins += 1
            elif outcome == "loss":
                losses += 1
            elif outcome == "tie":
                ties += 1
            else:
                raise ValueError(f"unknown evaluation outcome: {outcome!r}")
        pair_scores.append(sum(seat_scores) / len(seat_scores))

    game_count = pairs * 2
    score = wins / game_count
    lower, upper = bootstrap_mean_interval(
        pair_scores,
        resamples=bootstrap_resamples,
        seed=bootstrap_rng_seed,
    )
    return {
        "round_horizon": int(round_horizon),
        "stage_index": int(stage_index),
        "stage_iteration": int(stage_iteration),
        "global_iteration": int(global_iteration),
        "stage_traversals": int(stage_traversal_count),
        "cumulative_traversals": int(cumulative_traversal_count),
        "evaluation_pairs": int(pairs),
        "evaluation_games": int(game_count),
        "wins": int(wins),
        "losses": int(losses),
        "ties": int(ties),
        "win_rate": float(score),
        "win_rate_ci95_lower": float(lower),
        "win_rate_ci95_upper": float(upper),
        "score": float(score),
        "score_ci95_lower": float(lower),
        "score_ci95_upper": float(upper),
        "mean_utility": float(sum(utilities) / game_count),
        "mean_final_cash_differential": float(sum(differentials) / game_count),
    }


def checkpoint_key(record: Mapping[str, Any]) -> tuple[int, int]:
    """Identity used to skip already-evaluated learning-curve checkpoints."""

    return (int(record["stage_index"]), int(record["stage_iteration"]))


class LearningCurveStore:
    """Append-only JSON/CSV learning-curve history with resume deduplication."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        run_seed: int,
        evaluation_pairs: int,
        bootstrap_resamples: int,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.run_seed = int(run_seed)
        self.evaluation_pairs = int(evaluation_pairs)
        self.bootstrap_resamples = int(bootstrap_resamples)
        self.checkpoints: list[dict[str, Any]] = []
        self._keys: set[tuple[int, int]] = set()
        self.load()

    @property
    def json_path(self) -> Path:
        return self.output_dir / LEARNING_CURVE_JSON_NAME

    @property
    def csv_path(self) -> Path:
        return self.output_dir / LEARNING_CURVE_CSV_NAME

    @property
    def evaluation_history_path(self) -> Path:
        return self.output_dir / EVALUATION_HISTORY_CSV_NAME

    def load(self) -> None:
        path = self.json_path
        if not path.exists():
            return
        if not path.is_file():
            raise OSError(f"learning curve path is not a file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"learning curve file must contain an object: {path}")
        if payload.get("kind") not in {None, LEARNING_CURVE_KIND}:
            raise ValueError(f"unexpected learning curve kind in {path}")
        raw_checkpoints = payload.get("checkpoints", [])
        if not isinstance(raw_checkpoints, list):
            raise ValueError(f"learning curve checkpoints must be a list: {path}")
        checkpoints: list[dict[str, Any]] = []
        keys: set[tuple[int, int]] = set()
        for index, item in enumerate(raw_checkpoints):
            if not isinstance(item, dict):
                raise ValueError(
                    f"learning curve checkpoint {index} must be an object: {path}"
                )
            key = checkpoint_key(item)
            if key in keys:
                raise ValueError(
                    f"duplicate learning curve checkpoint {key} in {path}"
                )
            keys.add(key)
            normalized = dict(item)
            if "win_rate" not in normalized:
                games = int(normalized["evaluation_games"])
                wins = int(normalized["wins"])
                normalized["win_rate"] = wins / games if games else 0.0
            if "win_rate_ci95_lower" not in normalized:
                normalized["win_rate_ci95_lower"] = float(
                    normalized.get("score_ci95_lower", normalized["win_rate"])
                )
            if "win_rate_ci95_upper" not in normalized:
                normalized["win_rate_ci95_upper"] = float(
                    normalized.get("score_ci95_upper", normalized["win_rate"])
                )
            checkpoints.append(normalized)
        self.checkpoints = checkpoints
        self._keys = keys

    def reset(self) -> None:
        """Clear in-memory history and remove persisted learning-curve files."""

        self.checkpoints = []
        self._keys.clear()
        self.json_path.unlink(missing_ok=True)
        self.csv_path.unlink(missing_ok=True)
        self.evaluation_history_path.unlink(missing_ok=True)

    def contains(self, stage_index: int, stage_iteration: int) -> bool:
        return (int(stage_index), int(stage_iteration)) in self._keys

    def append(self, record: Mapping[str, Any]) -> bool:
        """Append one checkpoint when new; return whether it was written."""

        key = checkpoint_key(record)
        if key in self._keys:
            return False
        payload = {field: record[field] for field in _CHECKPOINT_FIELDS}
        self.checkpoints.append(payload)
        self._keys.add(key)
        self.save()
        return True

    def consecutive_win_rate_streak(self, threshold: float) -> int:
        """Return how many trailing checkpoints meet ``threshold`` win rate."""

        streak = 0
        for checkpoint in reversed(self.checkpoints):
            if float(checkpoint["win_rate"]) >= float(threshold):
                streak += 1
            else:
                break
        return streak

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": LEARNING_CURVE_SCHEMA_VERSION,
            "kind": LEARNING_CURVE_KIND,
            "seed": self.run_seed,
            "evaluation_pairs": self.evaluation_pairs,
            "evaluation_games_per_checkpoint": self.evaluation_pairs * 2,
            "bootstrap_resamples": self.bootstrap_resamples,
            "checkpoints": self.checkpoints,
        }
        rendered = (
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        temporary = self.json_path.with_suffix(".json.tmp")
        temporary.write_bytes(rendered)
        temporary.replace(self.json_path)

        with self.csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=_CHECKPOINT_FIELDS)
            writer.writeheader()
            for checkpoint in self.checkpoints:
                writer.writerow(
                    {field: checkpoint[field] for field in _CHECKPOINT_FIELDS}
                )

        with self.evaluation_history_path.open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=_EVALUATION_HISTORY_FIELDS)
            writer.writeheader()
            for checkpoint in self.checkpoints:
                writer.writerow(
                    {
                        "traversals": checkpoint["cumulative_traversals"],
                        "games": checkpoint["evaluation_games"],
                        "wins": checkpoint["wins"],
                        "losses": checkpoint["losses"],
                        "ties": checkpoint["ties"],
                        "win_rate": checkpoint["win_rate"],
                        "mean_utility": checkpoint["mean_utility"],
                        "ci_low": checkpoint["win_rate_ci95_lower"],
                        "ci_high": checkpoint["win_rate_ci95_upper"],
                    }
                )


_CHECKPOINT_FIELDS = (
    "round_horizon",
    "stage_index",
    "stage_iteration",
    "global_iteration",
    "stage_traversals",
    "cumulative_traversals",
    "evaluation_pairs",
    "evaluation_games",
    "wins",
    "losses",
    "ties",
    "score",
    "score_ci95_lower",
    "score_ci95_upper",
    "mean_utility",
    "mean_final_cash_differential",
    "win_rate",
    "win_rate_ci95_lower",
    "win_rate_ci95_upper",
)


def load_learning_curve_history(path: str | Path) -> dict[str, Any]:
    """Load a persisted learning-curve JSON document."""

    history_path = Path(path)
    if history_path.is_dir():
        history_path = history_path / LEARNING_CURVE_JSON_NAME
    if not history_path.is_file():
        raise FileNotFoundError(f"learning curve history not found: {history_path}")
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"learning curve history must be an object: {history_path}")
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError(f"learning curve history has no checkpoints: {history_path}")
    return payload


def plot_learning_curve(
    history: str | Path | Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Render the win-rate-vs-training graph from saved evaluation history."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "matplotlib is required to plot Deep CFR learning curves; "
            "install requirements-training.txt"
        ) from error

    document = (
        dict(history)
        if isinstance(history, Mapping)
        else load_learning_curve_history(history)
    )
    checkpoints = list(document["checkpoints"])
    checkpoints.sort(key=lambda item: int(item["cumulative_traversals"]))

    traversals = [int(item["cumulative_traversals"]) for item in checkpoints]
    scores = [
        100.0 * float(item.get("win_rate", item["score"])) for item in checkpoints
    ]
    lowers = [
        100.0
        * float(item.get("win_rate_ci95_lower", item["score_ci95_lower"]))
        for item in checkpoints
    ]
    uppers = [
        100.0
        * float(item.get("win_rate_ci95_upper", item["score_ci95_upper"]))
        for item in checkpoints
    ]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.fill_between(
        traversals,
        lowers,
        uppers,
        color="#9bbcff",
        alpha=0.45,
        linewidth=0,
        label="95% CI",
    )
    axis.plot(
        traversals,
        scores,
        color="#002fa7",
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        label="Deep CFR",
    )
    axis.axhline(
        50.0,
        color="#70747a",
        linestyle="--",
        linewidth=1.2,
        label="Random benchmark",
    )
    axis.set_title("Deep CFR Performance vs Random")
    axis.set_xlabel("Training traversals")
    axis.set_ylabel("Win rate vs random (%)")
    axis.set_ylim(0.0, 100.0)
    axis.legend(frameon=False)
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination.resolve()


__all__ = [
    "EVALUATION_HISTORY_CSV_NAME",
    "LEARNING_CURVE_CSV_NAME",
    "LEARNING_CURVE_JSON_NAME",
    "LEARNING_CURVE_KIND",
    "LEARNING_CURVE_PLOT_NAME",
    "LEARNING_CURVE_SCHEMA_VERSION",
    "LearningCurveStore",
    "bootstrap_mean_interval",
    "bootstrap_seed",
    "checkpoint_evaluation_seed",
    "checkpoint_key",
    "cumulative_traversals",
    "evaluate_learning_curve_checkpoint",
    "evaluation_checkpoint_iterations",
    "load_learning_curve_history",
    "outcome_score",
    "plot_learning_curve",
    "stage_evaluation_seed",
    "stage_traversals",
]
