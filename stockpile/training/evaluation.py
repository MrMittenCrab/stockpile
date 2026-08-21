"""Paired evaluation for canonical two-player Stockpile Lite policies."""

from __future__ import annotations

from collections.abc import Mapping
import math
import random
import statistics
from typing import TYPE_CHECKING, Any

from ..stockpile_platform import ConfiguredGame, score_game
from .encoding import TraceSession

if TYPE_CHECKING:
    from .policy import DeepCFRPolicy


_POLICY_STREAM_XOR = 0x9E3779B97F4A7C15
_UNIFORM_STREAM_XOR = 0xD1B54A32D192ED03
_SEED_MASK = (1 << 64) - 1

# Two-sided 95% Student-t critical values for 1..30 degrees of freedom.
_T_975 = (
    0.0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def _confidence_critical_value(sample_count: int) -> float:
    degrees_of_freedom = sample_count - 1
    if degrees_of_freedom <= 0:
        return 0.0
    if degrees_of_freedom < len(_T_975):
        return _T_975[degrees_of_freedom]
    return 1.96


def _coerce_configured_game(configuration: Any) -> ConfiguredGame:
    configured = getattr(configuration, "configured_game", configuration)
    if not isinstance(configured, ConfiguredGame):
        raise TypeError("configuration must be a ConfiguredGame or GameConfig")
    # TraceSession owns the canonical Deep CFR game contract: two-player Lite,
    # compact actions, no optional layers, and sealed selling.
    TraceSession(configured.game, 0)
    return configured


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return seed


def _sample_weighted(
    probabilities: Mapping[int, float] | list[tuple[int, float]],
    rng: random.Random,
) -> int:
    """Sample one action in iteration order from its exact supplied weight."""

    items = (
        list(probabilities.items())
        if isinstance(probabilities, Mapping)
        else list(probabilities)
    )
    if not items:
        raise ValueError("cannot sample an empty probability distribution")
    normalized: list[tuple[int, float]] = []
    total = 0.0
    for action, raw_weight in items:
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("probability weights must be finite and nonnegative")
        normalized.append((int(action), weight))
        total += weight
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("probability distribution must have positive total weight")

    threshold = rng.random() * total
    cumulative = 0.0
    last_positive = normalized[0][0]
    for action, weight in normalized:
        if weight > 0.0:
            last_positive = action
        cumulative += weight
        if threshold < cumulative:
            return action
    # Floating-point addition can finish infinitesimally below ``total``.
    return last_positive


def _policy_action(
    policy: DeepCFRPolicy,
    state: Any,
    player: int,
    trace_session: TraceSession,
    rng: random.Random,
) -> int:
    legal = tuple(int(action) for action in state.legal_actions(player))
    if not legal:
        raise RuntimeError("player decision node has no legal actions")
    if len(legal) == 1:
        return legal[0]
    probabilities = policy.action_probabilities(
        state,
        player,
        trace_session=trace_session,
    )
    return _sample_weighted(
        [(action, float(probabilities.get(action, 0.0))) for action in legal],
        rng,
    )


def play_evaluation_game(
    configuration: ConfiguredGame | Any,
    policy: DeepCFRPolicy,
    *,
    trained_seat: int,
    seed: int,
) -> dict[str, Any]:
    """Play one complete policy-versus-uniform game without recursive traversal."""

    configured = _coerce_configured_game(configuration)
    seed = _validate_seed(seed)
    if (
        isinstance(trained_seat, bool)
        or not isinstance(trained_seat, int)
        or trained_seat not in (0, 1)
    ):
        raise ValueError("trained_seat must be 0 or 1")

    state = configured.game.new_initial_state()
    sessions = [
        TraceSession(configured.game, player_id=player)
        for player in range(2)
    ]
    chance_rng = random.Random(seed)
    policy_rng = random.Random((seed & _SEED_MASK) ^ _POLICY_STREAM_XOR)
    uniform_rng = random.Random((seed & _SEED_MASK) ^ _UNIFORM_STREAM_XOR)
    chance_actions: list[int] = []
    player_action_count = 0

    while not state.is_terminal():
        if state.is_chance_node():
            action = _sample_weighted(state.chance_outcomes(), chance_rng)
            chance_actions.append(action)
            state.apply_action(action)
            continue

        player = int(state.current_player())
        legal = tuple(int(action) for action in state.legal_actions(player))
        if not legal:
            raise RuntimeError("player decision node has no legal actions")
        if len(legal) == 1:
            action = legal[0]
        elif player == trained_seat:
            action = _policy_action(
                policy,
                state,
                player,
                sessions[player],
                policy_rng,
            )
        else:
            action = int(uniform_rng.choice(legal))

        # Every player's own action belongs in perfect recall, including forced
        # nodes and the uniform opponent's decisions.
        sessions[player].record_action(state, action, forced=len(legal) == 1)
        state.apply_action(action)
        player_action_count += 1

    result = score_game(configured.rule_set, state)
    opponent = 1 - trained_seat
    trained_cash = int(result.final_cash_by_player[trained_seat])
    opponent_cash = int(result.final_cash_by_player[opponent])
    differential = trained_cash - opponent_cash
    trained_utility = float(result.utilities[trained_seat])
    outcome = (
        "win"
        if trained_utility > 0
        else "loss"
        if trained_utility < 0
        else "tie"
    )
    return {
        "seed": seed,
        "trained_seat": int(trained_seat),
        "trained_utility": trained_utility,
        "trained_final_cash": trained_cash,
        "opponent_final_cash": opponent_cash,
        "final_cash_differential": differential,
        "outcome": outcome,
        "chance_actions": chance_actions,
        "player_actions": player_action_count,
    }


def evaluate_policy(
    configuration: ConfiguredGame | Any,
    policy: DeepCFRPolicy,
    *,
    pairs: int,
    seed: int,
) -> dict[str, int | float | list[float]]:
    """Evaluate a policy in both seats for each shared-seed game pair."""

    configured = _coerce_configured_game(configuration)
    if isinstance(pairs, bool) or not isinstance(pairs, int) or pairs < 1:
        raise ValueError("pairs must be a positive integer")
    seed = _validate_seed(seed)

    games: list[dict[str, Any]] = []
    paired_utilities: list[float] = []
    for pair_index in range(pairs):
        pair_seed = seed + pair_index
        pair_games = [
            play_evaluation_game(
                configured,
                policy,
                trained_seat=trained_seat,
                seed=pair_seed,
            )
            for trained_seat in (0, 1)
        ]
        games.extend(pair_games)
        paired_utilities.append(
            statistics.fmean(
                float(game["trained_utility"]) for game in pair_games
            )
        )

    utilities = [float(game["trained_utility"]) for game in games]
    differentials = [float(game["final_cash_differential"]) for game in games]
    mean_utility = statistics.fmean(utilities)
    margin = 0.0
    if len(paired_utilities) > 1:
        margin = (
            _confidence_critical_value(len(paired_utilities))
            * statistics.stdev(paired_utilities)
            / math.sqrt(len(paired_utilities))
        )
    game_count = len(games)
    wins = sum(utility > 0.0 for utility in utilities)
    ties = sum(utility == 0.0 for utility in utilities)
    return {
        "pairs": int(pairs),
        "games": game_count,
        "trained_seat_mean_utility": float(mean_utility),
        "trained_seat_utility_ci95": [
            float(mean_utility - margin),
            float(mean_utility + margin),
        ],
        "win_rate": float(wins / game_count),
        "tie_rate": float(ties / game_count),
        "mean_final_cash_differential": float(statistics.fmean(differentials)),
    }


__all__ = ["evaluate_policy", "play_evaluation_game"]
