"""Learning-curve schedule, scoring, persistence, and plotting contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stockpile.training import learning_curve
from stockpile.training.config import DeepCFRConfig


class EvaluationScheduleTests(unittest.TestCase):
    def test_default_one_round_schedule_matches_traversal_grid(self) -> None:
        config = DeepCFRConfig()
        self.assertEqual(config.iterations_per_stage, 100)
        self.assertEqual(config.traversals_per_player, 20)
        self.assertEqual(config.learning_curve_pairs, 500)
        self.assertEqual(config.learning_curve_bootstrap_resamples, 10_000)

        iterations = learning_curve.evaluation_checkpoint_iterations(
            config.iterations_per_stage,
            checkpoint_count=config.learning_curve_checkpoint_count,
        )
        self.assertEqual(iterations, tuple(range(10, 101, 10)))
        traversals = [
            learning_curve.stage_traversals(iteration, config.traversals_per_player)
            for iteration in iterations
        ]
        self.assertEqual(
            traversals,
            [400, 800, 1_200, 1_600, 2_000, 2_400, 2_800, 3_200, 3_600, 4_000],
        )

    def test_schedule_rounds_deduplicates_and_always_includes_final(self) -> None:
        self.assertEqual(
            learning_curve.evaluation_checkpoint_iterations(2),
            (1, 2),
        )
        self.assertEqual(
            learning_curve.evaluation_checkpoint_iterations(5),
            (1, 2, 3, 4, 5),
        )


class ScoringAndBootstrapTests(unittest.TestCase):
    def test_outcome_scores_and_pair_bootstrap_are_deterministic(self) -> None:
        self.assertEqual(learning_curve.outcome_score("win"), 1.0)
        self.assertEqual(learning_curve.outcome_score("tie"), 0.5)
        self.assertEqual(learning_curve.outcome_score("loss"), 0.0)

        lower, upper = learning_curve.bootstrap_mean_interval(
            (0.0, 0.5, 1.0, 1.0),
            resamples=1_000,
            seed=123,
        )
        again = learning_curve.bootstrap_mean_interval(
            (0.0, 0.5, 1.0, 1.0),
            resamples=1_000,
            seed=123,
        )
        self.assertEqual((lower, upper), again)
        self.assertLessEqual(lower, 0.625)
        self.assertGreaterEqual(upper, 0.625)

    def test_checkpoint_evaluation_uses_fixed_pair_seeds_and_pair_scores(self) -> None:
        canned = {
            (17, 0): ("win", 1.0, 8),
            (17, 1): ("loss", -1.0, -2),
            (18, 0): ("tie", 0.0, 0),
            (18, 1): ("win", 1.0, 4),
        }
        calls: list[tuple[int, int]] = []

        def play(_configuration, _policy, *, trained_seat, seed):
            calls.append((seed, trained_seat))
            outcome, utility, differential = canned[(seed, trained_seat)]
            return {
                "outcome": outcome,
                "trained_utility": utility,
                "final_cash_differential": differential,
            }

        with patch.object(learning_curve, "play_evaluation_game", side_effect=play):
            record = learning_curve.evaluate_learning_curve_checkpoint(
                object(),
                object(),
                pairs=2,
                evaluation_seed=17,
                bootstrap_resamples=200,
                bootstrap_rng_seed=99,
                round_horizon=1,
                stage_index=0,
                stage_iteration=10,
                global_iteration=10,
                stage_traversal_count=400,
                cumulative_traversal_count=400,
            )

        self.assertEqual(calls, [(17, 0), (17, 1), (18, 0), (18, 1)])
        self.assertEqual(record["evaluation_pairs"], 2)
        self.assertEqual(record["evaluation_games"], 4)
        self.assertEqual(record["wins"], 2)
        self.assertEqual(record["losses"], 1)
        self.assertEqual(record["ties"], 1)
        # pair scores: mean(1,0)=0.5 and mean(0.5,1)=0.75 → mean 0.625
        self.assertAlmostEqual(record["score"], 0.625)
        self.assertAlmostEqual(record["win_rate"], 0.5)
        self.assertLessEqual(record["score_ci95_lower"], record["score"])
        self.assertGreaterEqual(record["score_ci95_upper"], record["score"])
        self.assertLessEqual(record["win_rate_ci95_lower"], record["win_rate"])
        self.assertGreaterEqual(record["win_rate_ci95_upper"], record["win_rate"])
        self.assertAlmostEqual(record["mean_utility"], 0.25)
        self.assertAlmostEqual(record["mean_final_cash_differential"], 2.5)
        json.dumps(record, allow_nan=False)


class PersistenceAndPlotTests(unittest.TestCase):
    def test_store_appends_skips_duplicates_and_rewrites_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = learning_curve.LearningCurveStore(
                root,
                run_seed=7,
                evaluation_pairs=2,
                bootstrap_resamples=32,
            )
            first = {
                "round_horizon": 1,
                "stage_index": 0,
                "stage_iteration": 1,
                "global_iteration": 1,
                "stage_traversals": 40,
                "cumulative_traversals": 40,
                "evaluation_pairs": 2,
                "evaluation_games": 4,
                "wins": 2,
                "losses": 1,
                "ties": 1,
                "win_rate": 0.5,
                "win_rate_ci95_lower": 0.0,
                "win_rate_ci95_upper": 1.0,
                "score": 0.625,
                "score_ci95_lower": 0.25,
                "score_ci95_upper": 0.9,
                "mean_utility": 0.1,
                "mean_final_cash_differential": 1.5,
            }
            self.assertTrue(store.append(first))
            self.assertFalse(store.append(first))
            second = dict(first)
            second.update(
                {
                    "stage_iteration": 2,
                    "global_iteration": 2,
                    "stage_traversals": 80,
                    "cumulative_traversals": 80,
                    "score": 0.7,
                }
            )
            self.assertTrue(store.append(second))

            reloaded = learning_curve.LearningCurveStore(
                root,
                run_seed=7,
                evaluation_pairs=2,
                bootstrap_resamples=32,
            )
            self.assertEqual(len(reloaded.checkpoints), 2)
            self.assertTrue(reloaded.contains(0, 1))
            self.assertTrue(reloaded.contains(0, 2))
            self.assertTrue(store.csv_path.is_file())
            rows = store.csv_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 3)
            self.assertTrue(store.evaluation_history_path.is_file())
            history_rows = store.evaluation_history_path.read_text(
                encoding="utf-8"
            ).strip().splitlines()
            self.assertEqual(
                history_rows[0],
                "traversals,games,wins,losses,ties,win_rate,mean_utility,ci_low,ci_high",
            )
            self.assertEqual(len(history_rows), 3)

            plot_path = learning_curve.plot_learning_curve(
                store.json_path,
                root / "analysis" / "learning_curve.png",
            )
            self.assertTrue(plot_path.is_file())


class UntilWinRateConfigTests(unittest.TestCase):
    def test_until_win_rate_defaults_and_validation(self) -> None:
        config = DeepCFRConfig(
            until_win_rate=0.7,
            max_traversals=1_000_000,
        )
        self.assertEqual(config.eval_every_traversals, 10_000)
        self.assertEqual(config.eval_games, 2_000)
        self.assertEqual(config.eval_every_iterations, 250)
        self.assertEqual(config.learning_curve_evaluation_pairs, 1_000)

        with self.assertRaisesRegex(ValueError, "max_traversals"):
            DeepCFRConfig(until_win_rate=0.7)
        with self.assertRaisesRegex(ValueError, "require --until-win-rate"):
            DeepCFRConfig(eval_every_traversals=10_000)
        with self.assertRaisesRegex(ValueError, "divisible"):
            DeepCFRConfig(
                until_win_rate=0.7,
                eval_every_traversals=10001,
                max_traversals=1_000_000,
            )

    def test_checkpoint_evaluation_seeds_differ_by_iteration(self) -> None:
        first = learning_curve.checkpoint_evaluation_seed(
            42, stage_index=0, stage_iteration=250
        )
        second = learning_curve.checkpoint_evaluation_seed(
            42, stage_index=0, stage_iteration=500
        )
        self.assertNotEqual(first, second)
        self.assertEqual(
            learning_curve.checkpoint_evaluation_seed(
                42, stage_index=0, stage_iteration=250
            ),
            first,
        )

    def test_consecutive_win_rate_streak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = learning_curve.LearningCurveStore(
                temporary,
                run_seed=1,
                evaluation_pairs=1,
                bootstrap_resamples=8,
            )
            base = {
                "round_horizon": 1,
                "stage_index": 0,
                "global_iteration": 1,
                "stage_traversals": 40,
                "evaluation_pairs": 1,
                "evaluation_games": 2,
                "wins": 2,
                "losses": 0,
                "ties": 0,
                "win_rate": 0.6,
                "win_rate_ci95_lower": 0.5,
                "win_rate_ci95_upper": 0.7,
                "score": 0.6,
                "score_ci95_lower": 0.5,
                "score_ci95_upper": 0.7,
                "mean_utility": 0.1,
                "mean_final_cash_differential": 1.0,
            }
            for index, rate in enumerate((0.6, 0.71, 0.72), start=1):
                record = dict(base)
                record.update(
                    {
                        "stage_iteration": index,
                        "global_iteration": index,
                        "cumulative_traversals": index * 40,
                        "win_rate": rate,
                        "score": rate,
                    }
                )
                store.append(record)
            self.assertEqual(store.consecutive_win_rate_streak(0.7), 2)
            self.assertEqual(store.consecutive_win_rate_streak(0.75), 0)


if __name__ == "__main__":
    unittest.main()
