"""Focused contracts for paired Stockpile Lite policy evaluation."""

from __future__ import annotations

import json
import math
import unittest
from unittest.mock import patch

import stockpile
from stockpile.training import evaluation
from stockpile.training.encoding import TraceSession


class _UniformTrackingPolicy:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def action_probabilities(
        self,
        state,
        player_id=None,
        *,
        trace_session=None,
    ):
        player = int(state.current_player()) if player_id is None else int(player_id)
        assert trace_session is not None
        self.calls.append((player, trace_session.trace.length))
        legal = tuple(int(action) for action in state.legal_actions(player))
        probability = 1.0 / len(legal)
        return {action: probability for action in legal}


class _FixedRandom:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


class EvaluationGameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.configuration = stockpile.resolve_configuration(
            "lite",
            round_count=1,
        )

    def test_complete_games_pair_exact_chance_and_record_both_players(self):
        policy = _UniformTrackingPolicy()
        recorded: list[tuple[int, int, int]] = []

        class TrackingTraceSession(TraceSession):
            def record_action(self, state, action_id, forced=None):
                recorded.append(
                    (self.player_id, int(state.current_player()), len(state.history()))
                )
                return super().record_action(state, action_id, forced)

        with patch.object(evaluation, "TraceSession", TrackingTraceSession):
            seat_zero = evaluation.play_evaluation_game(
                self.configuration,
                policy,
                trained_seat=0,
                seed=2718,
            )

        seat_one = evaluation.play_evaluation_game(
            self.configuration.configured_game,
            policy,
            trained_seat=1,
            seed=2718,
        )

        self.assertEqual(seat_zero["chance_actions"], seat_one["chance_actions"])
        self.assertGreater(seat_zero["player_actions"], 0)
        self.assertEqual(seat_zero["player_actions"], len(recorded))
        self.assertEqual({owner for owner, _actor, _step in recorded}, {0, 1})
        self.assertTrue(all(owner == actor for owner, actor, _step in recorded))
        self.assertTrue(policy.calls)
        self.assertEqual({player for player, _length in policy.calls}, {0, 1})
        for player in (0, 1):
            lengths = [length for owner, length in policy.calls if owner == player]
            self.assertTrue(
                all(left < right for left, right in zip(lengths, lengths[1:]))
            )
        json.dumps(seat_zero, allow_nan=False)
        json.dumps(seat_one, allow_nan=False)

    def test_chance_sampler_uses_supplied_nonuniform_probabilities(self):
        outcomes = [(7, 0.2), (9, 0.8)]
        self.assertEqual(evaluation._sample_weighted(outcomes, _FixedRandom(0.0)), 7)
        self.assertEqual(
            evaluation._sample_weighted(outcomes, _FixedRandom(0.199999)),
            7,
        )
        self.assertEqual(evaluation._sample_weighted(outcomes, _FixedRandom(0.2)), 9)
        self.assertEqual(
            evaluation._sample_weighted(outcomes, _FixedRandom(0.999999)),
            9,
        )
        for invalid in ([], [(1, 0.0)], [(1, -0.1)], [(1, math.nan)]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evaluation._sample_weighted(invalid, _FixedRandom(0.5))

    def test_rejects_noncanonical_games_seats_and_seeds(self):
        policy = _UniformTrackingPolicy()
        with self.assertRaisesRegex(ValueError, "canonical two-player Lite"):
            evaluation.play_evaluation_game(
                stockpile.resolve_configuration("classic", round_count=1),
                policy,
                trained_seat=0,
                seed=1,
            )
        with self.assertRaisesRegex(ValueError, "sell_order"):
            evaluation.play_evaluation_game(
                stockpile.resolve_configuration(
                    "lite",
                    round_count=1,
                    sell_order=True,
                ),
                policy,
                trained_seat=0,
                seed=1,
            )
        for invalid_seat in (2, True, 0.5):
            with self.subTest(trained_seat=invalid_seat), self.assertRaisesRegex(
                ValueError,
                "trained_seat",
            ):
                evaluation.play_evaluation_game(
                    self.configuration,
                    policy,
                    trained_seat=invalid_seat,
                    seed=1,
                )
        with self.assertRaisesRegex(TypeError, "seed"):
            evaluation.play_evaluation_game(
                self.configuration,
                policy,
                trained_seat=0,
                seed=True,
            )


class PairedMetricTests(unittest.TestCase):
    def test_metrics_use_shared_pair_seeds_and_are_json_serializable(self):
        configuration = stockpile.resolve_configuration("lite", round_count=1)
        policy = _UniformTrackingPolicy()
        canned = {
            (100, 0): (1.0, 10),
            (100, 1): (-1.0, -4),
            (101, 0): (0.0, 0),
            (101, 1): (1.0, 2),
        }
        calls: list[tuple[int, int]] = []

        def play(_configuration, _policy, *, trained_seat, seed):
            calls.append((seed, trained_seat))
            utility, differential = canned[(seed, trained_seat)]
            return {
                "trained_utility": utility,
                "final_cash_differential": differential,
            }

        with patch.object(evaluation, "play_evaluation_game", side_effect=play):
            metrics = evaluation.evaluate_policy(
                configuration,
                policy,
                pairs=2,
                seed=100,
            )

        self.assertEqual(calls, [(100, 0), (100, 1), (101, 0), (101, 1)])
        self.assertEqual(metrics["pairs"], 2)
        self.assertEqual(metrics["games"], 4)
        self.assertAlmostEqual(metrics["trained_seat_mean_utility"], 0.25)
        self.assertEqual(len(metrics["trained_seat_utility_ci95"]), 2)
        self.assertAlmostEqual(
            metrics["trained_seat_utility_ci95"][0],
            -2.9265,
        )
        self.assertAlmostEqual(
            metrics["trained_seat_utility_ci95"][1],
            3.4265,
        )
        self.assertAlmostEqual(metrics["win_rate"], 0.5)
        self.assertAlmostEqual(metrics["tie_rate"], 0.25)
        self.assertAlmostEqual(metrics["mean_final_cash_differential"], 2.0)
        json.dumps(metrics, allow_nan=False)

    def test_one_pair_has_a_degenerate_finite_interval_and_validates_count(self):
        configuration = stockpile.resolve_configuration("lite", round_count=1)
        policy = _UniformTrackingPolicy()

        def play(_configuration, _policy, *, trained_seat, seed):
            del trained_seat, seed
            return {
                "trained_utility": 0.25,
                "final_cash_differential": -1,
            }

        with patch.object(evaluation, "play_evaluation_game", side_effect=play):
            metrics = evaluation.evaluate_policy(
                configuration,
                policy,
                pairs=1,
                seed=7,
            )
        self.assertEqual(metrics["trained_seat_utility_ci95"], [0.25, 0.25])
        self.assertTrue(
            all(
                math.isfinite(float(value))
                for key, value in metrics.items()
                if key != "trained_seat_utility_ci95"
            )
        )

        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evaluation.evaluate_policy(
                    configuration,
                    policy,
                    pairs=invalid,
                    seed=7,
                )


if __name__ == "__main__":
    unittest.main()
