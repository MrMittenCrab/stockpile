"""Numerically guarded primitives for outcome-sampled Deep CFR.

The formulas mirror OpenSpiel's outcome-sampling MCCFR recursion with a zero
baseline.  Policies and targets use full, action-indexed vectors so a fixed
network head can be masked without remapping legal actions at every node.
Reach probabilities remain in float64 log space until an importance ratio is
materialized.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatVector: TypeAlias = NDArray[np.float64]
InformationSetKey: TypeAlias = tuple[int, str]
PolicyResult: TypeAlias = Sequence[float] | Mapping[int, float] | NDArray[np.float64]
PolicyCallable: TypeAlias = Callable[[Any, int, tuple[int, ...]], PolicyResult]


def _finite_scalar(value: float, name: str) -> np.float64:
    try:
        converted = np.float64(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite float") from error
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _finite_vector(values: ArrayLike, name: str) -> FloatVector:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a one-dimensional numeric vector") from error
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _legal_mask(mask: ArrayLike, size: int) -> NDArray[np.bool_]:
    raw = np.asarray(mask)
    if raw.ndim != 1 or raw.shape[0] != size:
        raise ValueError("legal_mask must be one-dimensional and match action count")
    if raw.dtype != np.bool_:
        try:
            numeric = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("legal_mask must contain booleans or zero/one values") from error
        if not np.all(np.isfinite(numeric)) or not np.all((numeric == 0) | (numeric == 1)):
            raise ValueError("legal_mask must contain booleans or zero/one values")
    result = raw.astype(np.bool_, copy=False)
    if not np.any(result):
        raise ValueError("legal_mask must contain at least one legal action")
    return result


def _policy_vector(
    policy: ArrayLike,
    legal_mask: NDArray[np.bool_],
    *,
    name: str,
) -> FloatVector:
    result = _finite_vector(policy, name)
    if result.shape != legal_mask.shape:
        raise ValueError(f"{name} and legal_mask must have the same shape")
    if np.any(result < 0):
        raise ValueError(f"{name} cannot contain negative probabilities")
    if np.any(result[~legal_mask] != 0):
        raise ValueError(f"{name} must assign zero probability to illegal actions")
    total = np.sum(result[legal_mask], dtype=np.float64)
    if not np.isfinite(total) or not np.isclose(total, 1.0, rtol=1e-7, atol=1e-10):
        raise ValueError(f"{name} must sum to one over legal actions")
    # Normalize close floating-point sums so all downstream formulas share the
    # exact same policy simplex.
    normalized = np.zeros_like(result)
    normalized[legal_mask] = result[legal_mask] / total
    return normalized


def regret_matching(regrets: ArrayLike, legal_mask: ArrayLike) -> FloatVector:
    """Convert a full regret/advantage vector to a masked policy.

    Positive legal regrets are normalized.  If there are none, probability is
    uniform over legal actions.  Scaling by the largest regret avoids overflow
    when several very large but finite regrets are present.
    """

    values = _finite_vector(regrets, "regrets")
    mask = _legal_mask(legal_mask, values.size)
    positive = np.zeros_like(values)
    positive[mask] = np.maximum(values[mask], 0.0)
    maximum = np.max(positive[mask])
    if maximum <= 0:
        result = mask.astype(np.float64)
        result /= np.sum(result, dtype=np.float64)
        return result
    scaled = positive / maximum
    result = scaled / np.sum(scaled, dtype=np.float64)
    result[~mask] = 0.0
    return result


def exploration_policy(
    policy: ArrayLike,
    legal_mask: ArrayLike,
    exploration: float = 0.6,
) -> FloatVector:
    """Mix a valid masked policy with uniform legal-action exploration."""

    values = _finite_vector(policy, "policy")
    mask = _legal_mask(legal_mask, values.size)
    normalized = _policy_vector(values, mask, name="policy")
    epsilon = _finite_scalar(exploration, "exploration")
    if epsilon < 0 or epsilon > 1:
        raise ValueError("exploration must be in [0, 1]")
    uniform = mask.astype(np.float64) / np.count_nonzero(mask)
    result = (1.0 - epsilon) * normalized + epsilon * uniform
    result[~mask] = 0.0
    return result


def _probability(value: float, name: str, *, positive: bool) -> np.float64:
    probability = _finite_scalar(value, name)
    lower_bound = probability > 0 if positive else probability >= 0
    if not lower_bound or probability > 1:
        interval = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} must be in {interval}")
    return probability


def _updated_log_reach(current: np.float64, probability: np.float64) -> np.float64:
    updated = np.float64(current + np.log(probability))
    if not np.isfinite(updated):
        raise FloatingPointError("log reach became nonfinite")
    return updated


def _safe_ratio(
    log_numerator: np.float64,
    numerator_is_zero: bool,
    log_denominator: np.float64,
    name: str,
) -> np.float64:
    if numerator_is_zero:
        return np.float64(0.0)
    difference = np.float64(log_numerator - log_denominator)
    if not np.isfinite(difference):
        raise FloatingPointError(f"{name} log-ratio is nonfinite")
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            ratio = np.float64(np.exp(difference))
    except FloatingPointError as error:
        raise FloatingPointError(f"{name} importance ratio overflowed") from error
    if not np.isfinite(ratio):
        raise FloatingPointError(f"{name} importance ratio is nonfinite")
    return ratio


@dataclass(frozen=True, slots=True)
class OutcomeSamplingReach:
    """Log-space reaches at one outcome-sampling traversal node.

    ``my`` is the update player's policy reach, ``opponent`` includes chance
    and every other player, and ``sample`` is the behavior-policy reach.  A
    zero target-policy reach is represented by a separate flag so every stored
    log value remains finite and accidental NaN/Inf input is always rejected.
    """

    log_my_reach: np.float64 = np.float64(0.0)
    log_opponent_reach: np.float64 = np.float64(0.0)
    log_sample_reach: np.float64 = np.float64(0.0)
    my_reach_is_zero: bool = False
    opponent_reach_is_zero: bool = False

    def __post_init__(self) -> None:
        for name in (
            "log_my_reach",
            "log_opponent_reach",
            "log_sample_reach",
        ):
            value = _finite_scalar(getattr(self, name), name)
            if value > 0:
                raise ValueError(f"{name} cannot be positive")
            object.__setattr__(self, name, value)
        if not isinstance(self.my_reach_is_zero, (bool, np.bool_)):
            raise TypeError("my_reach_is_zero must be boolean")
        if not isinstance(self.opponent_reach_is_zero, (bool, np.bool_)):
            raise TypeError("opponent_reach_is_zero must be boolean")
        object.__setattr__(self, "my_reach_is_zero", bool(self.my_reach_is_zero))
        object.__setattr__(
            self,
            "opponent_reach_is_zero",
            bool(self.opponent_reach_is_zero),
        )

    @classmethod
    def root(cls) -> "OutcomeSamplingReach":
        """Return unit policy, opponent, and sampling reaches."""

        return cls()

    def after_chance(self, probability: float) -> "OutcomeSamplingReach":
        """Advance through a sampled positive-probability chance outcome."""

        chance = _probability(probability, "chance probability", positive=True)
        return type(self)(
            log_my_reach=self.log_my_reach,
            log_opponent_reach=_updated_log_reach(
                self.log_opponent_reach,
                chance,
            ),
            log_sample_reach=_updated_log_reach(self.log_sample_reach, chance),
            my_reach_is_zero=self.my_reach_is_zero,
            opponent_reach_is_zero=self.opponent_reach_is_zero,
        )

    def after_action(
        self,
        *,
        actor_is_update_player: bool,
        policy_probability: float,
        sample_probability: float,
    ) -> "OutcomeSamplingReach":
        """Advance through one sampled player action.

        The behavior probability must be positive because the action was
        sampled.  Its target-policy probability may be zero when exploration
        selects a zero-regret action.
        """

        if not isinstance(actor_is_update_player, (bool, np.bool_)):
            raise TypeError("actor_is_update_player must be boolean")
        target = _probability(
            policy_probability,
            "policy probability",
            positive=False,
        )
        sampled = _probability(
            sample_probability,
            "sample probability",
            positive=True,
        )
        log_my = self.log_my_reach
        log_opponent = self.log_opponent_reach
        my_zero = self.my_reach_is_zero
        opponent_zero = self.opponent_reach_is_zero
        if actor_is_update_player:
            if target == 0:
                my_zero = True
            elif not my_zero:
                log_my = _updated_log_reach(log_my, target)
        else:
            if target == 0:
                opponent_zero = True
            elif not opponent_zero:
                log_opponent = _updated_log_reach(log_opponent, target)
        return type(self)(
            log_my_reach=log_my,
            log_opponent_reach=log_opponent,
            log_sample_reach=_updated_log_reach(self.log_sample_reach, sampled),
            my_reach_is_zero=my_zero,
            opponent_reach_is_zero=opponent_zero,
        )

    def opponent_over_sample(self) -> np.float64:
        """Return ``opponent_reach / sample_reach`` with overflow checks."""

        return _safe_ratio(
            self.log_opponent_reach,
            self.opponent_reach_is_zero,
            self.log_sample_reach,
            "opponent/sample",
        )

    def my_over_sample(self) -> np.float64:
        """Return ``my_reach / sample_reach`` with overflow checks."""

        return _safe_ratio(
            self.log_my_reach,
            self.my_reach_is_zero,
            self.log_sample_reach,
            "my/sample",
        )


def zero_baseline_child_values(
    sampled_action: int,
    child_value: float,
    sample_policy: ArrayLike,
    legal_mask: ArrayLike,
) -> FloatVector:
    """Return OpenSpiel's zero-baseline importance-corrected child values."""

    behavior = _finite_vector(sample_policy, "sample_policy")
    mask = _legal_mask(legal_mask, behavior.size)
    behavior = _policy_vector(behavior, mask, name="sample_policy")
    if isinstance(sampled_action, bool) or not isinstance(sampled_action, Integral):
        raise TypeError("sampled_action must be an integer action index")
    action = int(sampled_action)
    if action < 0 or action >= behavior.size or not mask[action]:
        raise ValueError("sampled_action must identify a legal action")
    if behavior[action] <= 0:
        raise ValueError("sampled action must have positive sampling probability")
    value = _finite_scalar(child_value, "child_value")
    corrected = np.float64(value / behavior[action])
    if not np.isfinite(corrected):
        raise FloatingPointError("importance-corrected child value is nonfinite")
    result = np.zeros_like(behavior)
    result[action] = corrected
    return result


def outcome_sampling_regret_target(
    child_values: ArrayLike,
    policy: ArrayLike,
    legal_mask: ArrayLike,
    reach: OutcomeSamplingReach,
) -> FloatVector:
    """Compute an importance-weighted instantaneous regret target.

    ``child_values`` should normally come from
    :func:`zero_baseline_child_values`.  Illegal target entries are always
    zero.
    """

    children = _finite_vector(child_values, "child_values")
    mask = _legal_mask(legal_mask, children.size)
    current_policy = _policy_vector(policy, mask, name="policy")
    if not isinstance(reach, OutcomeSamplingReach):
        raise TypeError("reach must be an OutcomeSamplingReach")
    value_estimate = outcome_sampling_value_estimate(
        children,
        current_policy,
        mask,
    )
    scale = reach.opponent_over_sample()
    result = np.zeros_like(children)
    result[mask] = (children[mask] - value_estimate) * scale
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("regret target is nonfinite")
    return result


def outcome_sampling_value_estimate(
    child_values: ArrayLike,
    policy: ArrayLike,
    legal_mask: ArrayLike,
) -> np.float64:
    """Return the policy-weighted value propagated by sampled recursion."""

    children = _finite_vector(child_values, "child_values")
    mask = _legal_mask(legal_mask, children.size)
    current_policy = _policy_vector(policy, mask, name="policy")
    value = np.float64(np.dot(current_policy[mask], children[mask]))
    if not np.isfinite(value):
        raise FloatingPointError("outcome-sampling value estimate is nonfinite")
    return value


def zero_baseline_regret_target(
    sampled_action: int,
    child_value: float,
    policy: ArrayLike,
    sample_policy: ArrayLike,
    legal_mask: ArrayLike,
    reach: OutcomeSamplingReach,
) -> FloatVector:
    """Convenience composition of zero-baseline child and regret helpers."""

    children = zero_baseline_child_values(
        sampled_action,
        child_value,
        sample_policy,
        legal_mask,
    )
    return outcome_sampling_regret_target(children, policy, legal_mask, reach)


def outcome_sampling_average_strategy_target(
    policy: ArrayLike,
    legal_mask: ArrayLike,
    reach: OutcomeSamplingReach,
    *,
    iteration_weight: float = 1.0,
) -> FloatVector:
    """Compute the reach- and iteration-weighted average-policy target."""

    values = _finite_vector(policy, "policy")
    mask = _legal_mask(legal_mask, values.size)
    current_policy = _policy_vector(values, mask, name="policy")
    if not isinstance(reach, OutcomeSamplingReach):
        raise TypeError("reach must be an OutcomeSamplingReach")
    weight = _finite_scalar(iteration_weight, "iteration_weight")
    if weight < 0:
        raise ValueError("iteration_weight cannot be negative")
    scale = np.float64(reach.my_over_sample() * weight)
    if not np.isfinite(scale):
        raise FloatingPointError("average-strategy importance weight is nonfinite")
    result = current_policy * scale
    result[~mask] = 0.0
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("average-strategy target is nonfinite")
    return result


def forced_action(legal_actions: Sequence[int]) -> int | None:
    """Return the sole legal action, or ``None`` when a choice is required."""

    actions = tuple(legal_actions)
    if not actions:
        raise ValueError("a non-terminal decision node must have a legal action")
    if any(isinstance(action, bool) or not isinstance(action, Integral) for action in actions):
        raise TypeError("legal actions must be integer action identifiers")
    normalized = tuple(int(action) for action in actions)
    if len(set(normalized)) != len(normalized):
        raise ValueError("legal actions cannot contain duplicates")
    return normalized[0] if len(normalized) == 1 else None


class NodeBudgetExceeded(RuntimeError):
    """Raised when a canonical validation traversal exceeds its node budget."""

    def __init__(self, budget: int, visited: int) -> None:
        self.budget = budget
        self.visited = visited
        super().__init__(
            f"canonical traversal exceeded node budget {budget} at node {visited}"
        )


@dataclass(frozen=True, slots=True)
class CanonicalTraversalResult:
    """Exact one-policy traversal values for a small sequential game tree."""

    expected_utility: float
    nodes_visited: int
    legal_actions: Mapping[InformationSetKey, tuple[int, ...]]
    counterfactual_action_values: Mapping[InformationSetKey, FloatVector]
    counterfactual_state_values: Mapping[InformationSetKey, float]
    regret_targets: Mapping[InformationSetKey, FloatVector]
    average_strategy_targets: Mapping[InformationSetKey, FloatVector]


def _resolved_canonical_policy(
    result: PolicyResult,
    legal_actions: tuple[int, ...],
) -> FloatVector:
    if isinstance(result, Mapping):
        unknown = set(result).difference(legal_actions)
        if unknown:
            raise ValueError(f"policy returned illegal action keys: {sorted(unknown)}")
        aligned = np.asarray(
            [result.get(action, 0.0) for action in legal_actions],
            dtype=np.float64,
        )
    else:
        vector = _finite_vector(result, "canonical policy")
        if vector.size == len(legal_actions):
            aligned = vector
        elif legal_actions and vector.size > max(legal_actions):
            aligned = vector[np.asarray(legal_actions, dtype=np.int64)]
        else:
            raise ValueError(
                "canonical policy must be legal-action aligned or full action indexed"
            )
    if not np.all(np.isfinite(aligned)) or np.any(aligned < 0):
        raise ValueError("canonical policy probabilities must be finite and nonnegative")
    total = np.sum(aligned, dtype=np.float64)
    if not np.isfinite(total) or not np.isclose(total, 1.0, rtol=1e-7, atol=1e-10):
        raise ValueError("canonical policy must sum to one")
    return aligned / total


def canonical_counterfactual_values(
    state: Any,
    update_player: int,
    policy: PolicyCallable,
    *,
    node_budget: int = 10_000,
) -> CanonicalTraversalResult:
    """Fully traverse a tiny sequential game for validation reference values.

    This deliberately simple kernel calls ``state.child(action)`` for every
    branch and is intended for games such as Kuhn poker, never Stockpile's
    production traversal.  The hard node budget prevents accidental expansion
    of a large game.  ``policy`` receives ``(state, player, legal_actions)`` and
    may return probabilities aligned with legal actions, a full action-indexed
    vector, or an action-to-probability mapping.
    """

    if isinstance(update_player, bool) or not isinstance(update_player, Integral):
        raise TypeError("update_player must be an integer")
    update_player = int(update_player)
    if update_player < 0:
        raise ValueError("update_player cannot be negative")
    if not callable(policy):
        raise TypeError("policy must be callable")
    if isinstance(node_budget, bool) or not isinstance(node_budget, int):
        raise TypeError("node_budget must be an integer")
    if node_budget < 1:
        raise ValueError("node_budget must be positive")

    visited = 0
    legal_by_key: dict[InformationSetKey, tuple[int, ...]] = {}
    policy_by_key: dict[InformationSetKey, FloatVector] = {}
    cf_actions: dict[InformationSetKey, FloatVector] = {}
    cf_states: dict[InformationSetKey, float] = {}
    regrets: dict[InformationSetKey, FloatVector] = {}
    averages: dict[InformationSetKey, FloatVector] = {}

    def traverse(node: Any, my_reach: float, opponent_reach: float) -> float:
        nonlocal visited
        visited += 1
        if visited > node_budget:
            raise NodeBudgetExceeded(node_budget, visited)

        if node.is_terminal():
            if hasattr(node, "player_return"):
                value = node.player_return(update_player)
            else:
                value = node.returns()[update_player]
            return float(_finite_scalar(value, "terminal utility"))

        if node.is_chance_node():
            outcomes = tuple(node.chance_outcomes())
            if not outcomes:
                raise ValueError("chance node has no outcomes")
            probabilities = np.asarray([item[1] for item in outcomes], dtype=np.float64)
            if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
                raise ValueError("chance probabilities must be finite and nonnegative")
            total = np.sum(probabilities, dtype=np.float64)
            if not np.isclose(total, 1.0, rtol=1e-7, atol=1e-10):
                raise ValueError("chance probabilities must sum to one")
            value = 0.0
            for (action, probability), normalized_probability in zip(
                outcomes,
                probabilities / total,
                strict=True,
            ):
                del probability
                if normalized_probability == 0:
                    continue
                child_value = traverse(
                    node.child(action),
                    my_reach,
                    opponent_reach * float(normalized_probability),
                )
                value += float(normalized_probability) * child_value
            return float(_finite_scalar(value, "chance-node value"))

        if hasattr(node, "is_simultaneous_node") and node.is_simultaneous_node():
            raise ValueError("canonical traversal requires a sequential game")
        player = int(node.current_player())
        if player < 0:
            raise ValueError("unsupported non-player node in canonical traversal")
        actions = tuple(int(action) for action in node.legal_actions())
        if not actions:
            raise ValueError("non-terminal player node has no legal actions")
        probabilities = _resolved_canonical_policy(
            policy(node, player, actions),
            actions,
        )
        information_state = str(node.information_state_string(player))
        key = (player, information_state)
        prior_actions = legal_by_key.get(key)
        if prior_actions is not None:
            if prior_actions != actions:
                raise ValueError("one information state exposed inconsistent legal actions")
            if not np.allclose(
                policy_by_key[key],
                probabilities,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError("policy differs across histories in one information state")
        else:
            legal_by_key[key] = actions
            policy_by_key[key] = probabilities.copy()

        action_values = np.empty(len(actions), dtype=np.float64)
        for index, (action, action_probability) in enumerate(
            zip(actions, probabilities, strict=True)
        ):
            if player == update_player:
                child_my_reach = my_reach * float(action_probability)
                child_opponent_reach = opponent_reach
            else:
                child_my_reach = my_reach
                child_opponent_reach = opponent_reach * float(action_probability)
            action_values[index] = traverse(
                node.child(action),
                child_my_reach,
                child_opponent_reach,
            )
        node_value = float(np.dot(probabilities, action_values))
        _finite_scalar(node_value, "decision-node value")

        if player == update_player:
            weighted_actions = action_values * opponent_reach
            weighted_state = node_value * opponent_reach
            weighted_regrets = (action_values - node_value) * opponent_reach
            weighted_average = probabilities * my_reach
            if key not in regrets:
                cf_actions[key] = np.zeros_like(action_values)
                cf_states[key] = 0.0
                regrets[key] = np.zeros_like(action_values)
                averages[key] = np.zeros_like(action_values)
            cf_actions[key] += weighted_actions
            cf_states[key] += weighted_state
            regrets[key] += weighted_regrets
            averages[key] += weighted_average
            if not all(
                np.all(np.isfinite(values))
                for values in (cf_actions[key], regrets[key], averages[key])
            ) or not math.isfinite(cf_states[key]):
                raise FloatingPointError("canonical counterfactual target is nonfinite")
        return node_value

    utility = traverse(state, 1.0, 1.0)
    return CanonicalTraversalResult(
        expected_utility=utility,
        nodes_visited=visited,
        legal_actions=dict(legal_by_key),
        counterfactual_action_values=cf_actions,
        counterfactual_state_values=cf_states,
        regret_targets=regrets,
        average_strategy_targets=averages,
    )


__all__ = [
    "CanonicalTraversalResult",
    "NodeBudgetExceeded",
    "OutcomeSamplingReach",
    "canonical_counterfactual_values",
    "exploration_policy",
    "forced_action",
    "outcome_sampling_average_strategy_target",
    "outcome_sampling_regret_target",
    "outcome_sampling_value_estimate",
    "regret_matching",
    "zero_baseline_child_values",
    "zero_baseline_regret_target",
]
