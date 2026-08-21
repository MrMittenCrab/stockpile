"""Focused tests for framework-independent Deep CFR primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math
import unittest

import numpy as np
import pyspiel

from stockpile.training.memory import ReservoirBuffer
from stockpile.training.sampling import (
    NodeBudgetExceeded,
    OutcomeSamplingReach,
    canonical_counterfactual_values,
    exploration_policy,
    forced_action,
    outcome_sampling_average_strategy_target,
    outcome_sampling_regret_target,
    outcome_sampling_value_estimate,
    regret_matching,
    zero_baseline_child_values,
    zero_baseline_regret_target,
)


@dataclass(frozen=True)
class _Sample:
    identifier: int
    payload: tuple[float, ...]


class ReservoirBufferTests(unittest.TestCase):
    def test_seeded_algorithm_r_is_deterministic_for_dataclass_samples(self):
        left = ReservoirBuffer[_Sample](capacity=4, seed=173)
        right = ReservoirBuffer[_Sample](capacity=4, seed=173)
        samples = [_Sample(index, (float(index),)) for index in range(30)]

        left_updates = left.extend(samples)
        right_updates = right.extend(samples)

        self.assertEqual(left_updates, right_updates)
        self.assertEqual(left.items, right.items)
        self.assertEqual(len(left), 4)
        self.assertEqual(left.seen_count, 30)
        self.assertTrue(any(not update.retained for update in left_updates[4:]))
        self.assertTrue(any(update.evicted is not None for update in left_updates[4:]))

    def test_sample_caps_request_and_checkpoint_restores_rng_exactly(self):
        original = ReservoirBuffer[int](capacity=5, seed=91)
        original.extend(range(20))
        checkpoint = original.state_dict()

        restored = ReservoirBuffer[int](capacity=5, seed=999)
        restored.load_state_dict(checkpoint)
        self.assertEqual(restored.values, original.values)
        self.assertEqual(restored.seen_count, original.seen_count)
        self.assertEqual(restored.sample(50), original.sample(50))

        for value in range(20, 40):
            self.assertEqual(restored.append(value), original.append(value))
        self.assertEqual(restored.items, original.items)
        self.assertEqual(restored.sample(3), original.sample(3))
        self.assertEqual(restored.sample(0), [])

        constructed = ReservoirBuffer.from_state_dict(original.state_dict())
        self.assertEqual(constructed.state_dict(), original.state_dict())

    def test_invalid_capacity_requests_and_checkpoint_shape_are_rejected(self):
        with self.assertRaises(ValueError):
            ReservoirBuffer[int](0)
        buffer = ReservoirBuffer[int](2, seed=1)
        with self.assertRaises(ValueError):
            buffer.sample(-1)
        buffer.append(1)
        state = buffer.state_dict()
        state["seen_count"] = 2
        with self.assertRaisesRegex(ValueError, "item count"):
            buffer.load_state_dict(state)
        with self.assertRaisesRegex(ValueError, "capacity mismatch"):
            ReservoirBuffer[int](3).load_state_dict(
                ReservoirBuffer[int](2).state_dict()
            )


class SamplingMathTests(unittest.TestCase):
    def test_regret_matching_masks_and_uses_uniform_fallback(self):
        mask = np.array([True, True, True, False])
        np.testing.assert_allclose(
            regret_matching([-2.0, 3.0, 1.0, 100.0], mask),
            [0.0, 0.75, 0.25, 0.0],
        )
        np.testing.assert_allclose(
            regret_matching([-2.0, 0.0, -1.0, 100.0], mask),
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0],
        )
        huge = np.finfo(np.float64).max
        probabilities = regret_matching([huge, huge, -1.0], [1, 1, 1])
        np.testing.assert_allclose(probabilities, [0.5, 0.5, 0.0])

    def test_exploration_policy_is_uniform_policy_mixture(self):
        probabilities = exploration_policy(
            [1.0, 0.0, 0.0],
            [True, True, False],
            exploration=0.6,
        )
        np.testing.assert_allclose(probabilities, [0.7, 0.3, 0.0])
        self.assertEqual(float(np.sum(probabilities)), 1.0)

    def test_log_reach_updates_and_zero_target_reach(self):
        reach = OutcomeSamplingReach.root().after_chance(0.25)
        reach = reach.after_action(
            actor_is_update_player=True,
            policy_probability=0.5,
            sample_probability=0.8,
        )
        reach = reach.after_action(
            actor_is_update_player=False,
            policy_probability=0.4,
            sample_probability=0.4,
        )
        self.assertAlmostEqual(float(reach.opponent_over_sample()), 1.25)
        self.assertAlmostEqual(float(reach.my_over_sample()), 6.25)
        self.assertTrue(
            all(
                math.isfinite(float(value))
                for value in (
                    reach.log_my_reach,
                    reach.log_opponent_reach,
                    reach.log_sample_reach,
                )
            )
        )

        zero_reach = OutcomeSamplingReach.root().after_action(
            actor_is_update_player=True,
            policy_probability=0.0,
            sample_probability=0.3,
        )
        self.assertEqual(float(zero_reach.my_over_sample()), 0.0)

    def test_zero_baseline_regret_and_average_targets_match_formulas(self):
        mask = np.array([True, True, True, False])
        policy = np.array([0.5, 0.25, 0.25, 0.0])
        sample_policy = np.array([0.5, 0.25, 0.25, 0.0])
        reach = OutcomeSamplingReach(
            log_my_reach=np.log(0.125),
            log_opponent_reach=np.log(0.5),
            log_sample_reach=np.log(0.25),
        )

        children = zero_baseline_child_values(
            1,
            2.0,
            sample_policy,
            mask,
        )
        np.testing.assert_allclose(children, [0.0, 8.0, 0.0, 0.0])
        self.assertEqual(
            float(outcome_sampling_value_estimate(children, policy, mask)),
            2.0,
        )
        regrets = outcome_sampling_regret_target(
            children,
            policy,
            mask,
            reach,
        )
        np.testing.assert_allclose(regrets, [-4.0, 12.0, -4.0, 0.0])
        np.testing.assert_allclose(
            zero_baseline_regret_target(
                1,
                2.0,
                policy,
                sample_policy,
                mask,
                reach,
            ),
            regrets,
        )
        np.testing.assert_allclose(
            outcome_sampling_average_strategy_target(
                policy,
                mask,
                reach,
                iteration_weight=3.0,
            ),
            [0.75, 0.375, 0.375, 0.0],
        )

    def test_zero_baseline_regret_estimator_is_unbiased_over_sampled_actions(self):
        mask = np.array([True, True, True])
        policy = np.array([0.5, 0.3, 0.2])
        behavior = np.array([0.4, 0.4, 0.2])
        action_utilities = np.array([-1.5, 2.0, 0.25])

        expected_estimate = np.zeros(3, dtype=np.float64)
        for action, sampling_probability in enumerate(behavior):
            expected_estimate += sampling_probability * zero_baseline_regret_target(
                action,
                action_utilities[action],
                policy,
                behavior,
                mask,
                OutcomeSamplingReach.root(),
            )
        exact_value = float(np.dot(policy, action_utilities))
        np.testing.assert_allclose(
            expected_estimate,
            action_utilities - exact_value,
            rtol=1e-14,
            atol=1e-14,
        )

    def test_nonfinite_inputs_and_importance_overflow_are_rejected(self):
        with self.assertRaises(ValueError):
            regret_matching([0.0, np.inf], [True, True])
        with self.assertRaises(ValueError):
            OutcomeSamplingReach(log_my_reach=np.nan)
        with self.assertRaises(ValueError):
            zero_baseline_child_values(
                0,
                np.inf,
                [1.0, 0.0],
                [True, False],
            )
        overflowing = OutcomeSamplingReach(
            log_opponent_reach=0.0,
            log_sample_reach=-1_000.0,
        )
        with self.assertRaises(FloatingPointError):
            overflowing.opponent_over_sample()

    def test_forced_action_helper(self):
        self.assertEqual(forced_action([7]), 7)
        self.assertIsNone(forced_action([7, 8]))
        with self.assertRaises(ValueError):
            forced_action([])
        with self.assertRaises(ValueError):
            forced_action([7, 7])


class CanonicalTraversalTests(unittest.TestCase):
    @staticmethod
    def _uniform_policy(_state, _player, legal_actions):
        probability = 1.0 / len(legal_actions)
        return {action: probability for action in legal_actions}

    def test_kuhn_reference_traversal_returns_exact_consistent_targets(self):
        state = pyspiel.load_game("kuhn_poker").new_initial_state()
        result = canonical_counterfactual_values(
            state,
            update_player=0,
            policy=self._uniform_policy,
            node_budget=1_000,
        )

        self.assertAlmostEqual(result.expected_utility, 0.125)
        self.assertGreater(result.nodes_visited, 1)
        self.assertTrue(result.regret_targets)
        for key, regrets in result.regret_targets.items():
            np.testing.assert_allclose(
                regrets,
                result.counterfactual_action_values[key]
                - result.counterfactual_state_values[key],
            )
            self.assertEqual(regrets.shape, (len(result.legal_actions[key]),))
            self.assertTrue(np.all(result.average_strategy_targets[key] >= 0))

    def test_node_budget_guards_canonical_expansion(self):
        state = pyspiel.load_game("kuhn_poker").new_initial_state()
        with self.assertRaises(NodeBudgetExceeded) as raised:
            canonical_counterfactual_values(
                state,
                update_player=0,
                policy=self._uniform_policy,
                node_budget=2,
            )
        self.assertEqual(raised.exception.budget, 2)
        self.assertEqual(raised.exception.visited, 3)


if __name__ == "__main__":
    unittest.main()
