"""End-to-end contracts for the optional sampled Deep CFR trainer."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from io import StringIO
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import stockpile
from stockpile.training.config import (
    DEFAULT_CURRICULUM,
    CurriculumConfig,
    DeepCFRConfig,
    parse_curriculum,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class CurriculumTests(unittest.TestCase):
    def test_deep_cfr_defaults_bound_strict_history_memory(self):
        config = DeepCFRConfig()

        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.memory_capacity, 2_000)
        self.assertEqual(config.output_dir, Path("artifacts/deep_cfr/default"))

    def test_six_round_default_skips_five_but_manual_schedule_can_include_it(self):
        self.assertEqual(DEFAULT_CURRICULUM, (1, 2, 3, 4, 6))
        self.assertEqual(CurriculumConfig.for_target(6).rounds, DEFAULT_CURRICULUM)
        self.assertEqual(
            CurriculumConfig.for_target(6, "1,2,3,4,5,6").rounds,
            (1, 2, 3, 4, 5, 6),
        )

    def test_curriculum_is_ordered_bounded_and_ends_at_target(self):
        for invalid in ("", "2,3", "1,1", "1,7", "1,three"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_curriculum(invalid)
        with self.assertRaisesRegex(ValueError, "end"):
            CurriculumConfig.for_target(4, "1,2,3")

    def test_importing_training_package_does_not_import_torch(self):
        repository = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import stockpile.training; "
                    "assert 'torch' not in sys.modules"
                ),
            ],
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch training dependency")
class TrainerSmokeTests(unittest.TestCase):
    def _config(self, output_dir: Path) -> DeepCFRConfig:
        return DeepCFRConfig(
            curriculum=CurriculumConfig((1,)),
            iterations_per_stage=1,
            traversals_per_player=1,
            advantage_train_steps=1,
            strategy_train_steps=1,
            batch_size=8,
            memory_capacity=64,
            checkpoint_every=1,
            evaluation_pairs=1,
            learning_curve_pairs=1,
            learning_curve_bootstrap_resamples=16,
            learning_curve_checkpoint_count=10,
            seed=91,
            device="cpu",
            output_dir=output_dir,
        )

    def test_one_round_train_export_and_exact_checkpoint_restore(self):
        import torch

        from stockpile import complexity_cache
        from stockpile.training.policy import DeepCFRPolicy
        from stockpile.training.trainer import DeepCFRTrainer

        configuration = stockpile.resolve_configuration("lite", round_count=1)
        with TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            trainer = DeepCFRTrainer(
                config,
                base_configuration=configuration,
                output=StringIO(),
            )
            result = trainer.train()

            self.assertEqual(result.completed_rounds, (1,))
            self.assertTrue(result.final_checkpoint.is_file())
            self.assertTrue(result.final_policy.is_file())
            self.assertTrue((Path(temporary) / "metrics.jsonl").is_file())
            self.assertTrue((Path(temporary) / "learning_curve.json").is_file())
            self.assertTrue((Path(temporary) / "learning_curve.csv").is_file())
            self.assertEqual(trainer.stage_iteration, 1)
            self.assertEqual(trainer.global_iteration, 1)
            self.assertGreater(len(trainer.advantage_memories[0]), 0)
            self.assertGreater(len(trainer.advantage_memories[1]), 0)
            self.assertGreater(len(trainer.strategy_memory), 0)
            for memory in trainer.advantage_memories:
                for sample in memory:
                    self.assertEqual(sample.target.dtype.name, "float64")

            policy = DeepCFRPolicy.load(result.final_policy)
            self.assertEqual(policy.metadata["rounds"], 1)
            self.assertFalse(policy.metadata["sell_order"])
            self.assertFalse(policy.metadata["equilibrium_claim"])

            capped_rules = replace(
                configuration.rule_set,
                standard_price_ceiling=10,
            )
            capped_game = replace(
                configuration.configured_game,
                rule_set=capped_rules,
            )
            capped_fingerprint = complexity_cache.semantic_fingerprint(capped_game)

            legacy_policy_path = Path(temporary) / "legacy_capped_policy.pt"
            legacy_policy_payload = torch.load(
                result.final_policy,
                map_location="cpu",
                weights_only=False,
            )
            legacy_policy_payload["metadata"][
                "semantic_fingerprint"
            ] = capped_fingerprint
            legacy_policy_payload["metadata"].pop("price_semantics", None)
            torch.save(legacy_policy_payload, legacy_policy_path)
            self.assertEqual(
                DeepCFRPolicy.load(legacy_policy_path).metadata[
                    "semantic_fingerprint"
                ],
                capped_fingerprint,
            )

            legacy_full_path = Path(temporary) / "legacy_capped_full.pt"
            legacy_full_payload = torch.load(
                result.final_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            legacy_full_payload["schema_version"] = 1
            legacy_full_payload["metadata"][
                "semantic_fingerprint"
            ] = capped_fingerprint
            legacy_full_payload["metadata"].pop("price_semantics", None)
            torch.save(legacy_full_payload, legacy_full_path)
            legacy_full_loader = DeepCFRTrainer(
                config,
                base_configuration=configuration,
                output=StringIO(),
            )
            with self.assertRaisesRegex(ValueError, "game semantics do not match"):
                legacy_full_loader.load_checkpoint(legacy_full_path)

            original_next_random = trainer.rng.random()
            original_sample_ids = [
                sample.information.perfect_recall_id
                for sample in trainer.advantage_memories[0].sample(4)
            ]
            restored = DeepCFRTrainer(
                config,
                base_configuration=configuration,
                output=StringIO(),
            )
            restored.load_checkpoint(result.final_checkpoint)
            self.assertEqual(restored.stage_iteration, trainer.stage_iteration)
            self.assertEqual(restored.global_iteration, trainer.global_iteration)
            self.assertEqual(restored.rng.random(), original_next_random)
            restored_sample_ids = [
                sample.information.perfect_recall_id
                for sample in restored.advantage_memories[0].sample(4)
            ]
            self.assertEqual(restored_sample_ids, original_sample_ids)
            for original, loaded in zip(
                trainer.strategy_network.parameters(),
                restored.strategy_network.parameters(),
                strict=True,
            ):
                self.assertTrue(torch.equal(original, loaded))

            metrics_path = Path(temporary) / "metrics.jsonl"
            checkpoint_metrics = metrics_path.read_bytes()
            post_checkpoint_metrics = (
                checkpoint_metrics
                + b'{"kind":"post_checkpoint_uncommitted_metric"}\n'
            )
            metrics_path.write_bytes(post_checkpoint_metrics)
            config_path = Path(temporary) / "config.json"
            config_document = json.loads(config_path.read_text(encoding="utf-8"))
            config_document["future_additive_metadata"] = {
                "preserve": "exactly"
            }
            post_checkpoint_config = (
                json.dumps(config_document, indent=1, sort_keys=False) + "\n"
            ).encode("utf-8")
            config_path.write_bytes(post_checkpoint_config)
            resumed = DeepCFRTrainer(
                config,
                base_configuration=configuration,
                output=StringIO(),
            )
            with patch(
                "stockpile.training.evaluation.evaluate_policy"
            ) as evaluate_policy:
                resumed_result = resumed.train(resume=result.final_checkpoint)
            evaluate_policy.assert_not_called()
            self.assertEqual(resumed_result.completed_rounds, (1,))
            self.assertEqual(resumed_result.metrics, result.metrics)
            self.assertEqual(metrics_path.read_bytes(), checkpoint_metrics)
            self.assertEqual(config_path.read_bytes(), post_checkpoint_config)

            for source_name, preserved in (("metrics.jsonl", post_checkpoint_metrics),):
                with self.subTest(recovered=source_name):
                    digest = hashlib.sha256(preserved).hexdigest()
                    recovery_dir = (
                        Path(temporary)
                        / "recovery"
                        / "resume_reconciliation"
                        / source_name
                        / f"sha256-{digest}"
                    )
                    self.assertEqual(
                        (recovery_dir / source_name).read_bytes(),
                        preserved,
                    )
                    provenance = json.loads(
                        (recovery_dir / "provenance.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(
                        provenance,
                        {
                            "kind": "stockpile_deep_cfr_resume_recovery",
                            "preserved_byte_count": len(preserved),
                            "preserved_sha256": digest,
                            "reason": "resume_reconciliation",
                            "schema_version": 1,
                            "source_path": source_name,
                        },
                    )
            self.assertFalse(
                (
                    Path(temporary)
                    / "recovery"
                    / "resume_reconciliation"
                    / "config.json"
                ).exists()
            )

            incompatible_output = Path(temporary) / "incompatible"
            incompatible = DeepCFRTrainer(
                replace(
                    config,
                    output_dir=incompatible_output,
                    learning_rate=config.learning_rate * 2,
                ),
                base_configuration=configuration,
                output=StringIO(),
            )
            with self.assertRaisesRegex(ValueError, "configuration does not match"):
                incompatible.train(resume=result.final_checkpoint)
            self.assertFalse((incompatible_output / "config.json").exists())
            self.assertFalse((incompatible_output / "metrics.jsonl").exists())

    def test_fresh_run_refuses_nonempty_output_unless_overwrite_is_enabled(self):
        from stockpile.training.trainer import DeepCFRTrainer

        configuration = stockpile.resolve_configuration("lite", round_count=1)
        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            marker = output_dir / "existing.txt"
            marker.write_text("preserve until authorized\n", encoding="utf-8")
            config = self._config(output_dir)
            refused = DeepCFRTrainer(
                config,
                base_configuration=configuration,
                output=StringIO(),
            )

            with self.assertRaisesRegex(ValueError, "output directory is not empty"):
                refused.train()
            self.assertEqual(
                marker.read_text(encoding="utf-8"),
                "preserve until authorized\n",
            )
            self.assertFalse((output_dir / "config.json").exists())
            self.assertFalse((output_dir / "metrics.jsonl").exists())

            allowed = DeepCFRTrainer(
                config,
                base_configuration=configuration,
                output=StringIO(),
            )
            with (
                patch.object(
                    allowed,
                    "_reset_stage",
                    side_effect=RuntimeError("continued past output guard"),
                ),
                self.assertRaisesRegex(RuntimeError, "continued past output guard"),
            ):
                allowed.train(overwrite=True)
            self.assertTrue((output_dir / "config.json").is_file())
            self.assertTrue((output_dir / "metrics.jsonl").is_file())

    def test_policy_loader_rejects_encoder_and_action_schema_mismatches(self):
        import torch

        from stockpile.training.encoding import ENCODING_SCHEMA_VERSION
        from stockpile.training.policy import (
            ACTION_SCHEMA_VERSION,
            POLICY_SCHEMA_VERSION,
            DeepCFRPolicy,
        )

        valid_metadata = {
            "encoder_schema_version": ENCODING_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
        }
        cases = (
            (
                "encoder_schema_version",
                "future_encoder",
                "encoder schema is incompatible",
            ),
            (
                "action_schema_version",
                "future_actions",
                "action schema is incompatible",
            ),
        )
        with TemporaryDirectory() as temporary:
            for field, incompatible_value, message in cases:
                with self.subTest(field=field):
                    metadata = dict(valid_metadata)
                    metadata[field] = incompatible_value
                    path = Path(temporary) / f"{field}.pt"
                    torch.save(
                        {
                            "kind": "stockpile_deep_cfr_policy",
                            "schema_version": POLICY_SCHEMA_VERSION,
                            "metadata": metadata,
                        },
                        path,
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        DeepCFRPolicy.load(path)

    def test_trainer_rejects_ordered_selling_and_non_lite_games(self):
        from stockpile.training.trainer import DeepCFRTrainer

        with TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            with self.assertRaisesRegex(ValueError, "enabled overrides.*sell_order"):
                DeepCFRTrainer(
                    config,
                    base_configuration=stockpile.resolve_configuration(
                        "lite",
                        round_count=1,
                        sell_order=True,
                    ),
                    output=StringIO(),
                )
            with self.assertRaisesRegex(ValueError, "enabled overrides.*impact"):
                DeepCFRTrainer(
                    config,
                    base_configuration=stockpile.resolve_configuration(
                        "lite",
                        round_count=1,
                        impact=True,
                    ),
                    output=StringIO(),
                )
            with self.assertRaisesRegex(ValueError, "only Stockpile Lite"):
                DeepCFRTrainer(
                    config,
                    base_configuration=stockpile.resolve_configuration(
                        "classic",
                        round_count=1,
                    ),
                    output=StringIO(),
                )

    def test_until_win_rate_stops_after_two_consecutive_hits(self):
        from stockpile.training.trainer import DeepCFRTrainer

        configuration = stockpile.resolve_configuration("lite", round_count=1)
        with TemporaryDirectory() as temporary:
            config = DeepCFRConfig(
                curriculum=CurriculumConfig((1,)),
                iterations_per_stage=1,
                traversals_per_player=1,
                advantage_train_steps=1,
                strategy_train_steps=1,
                batch_size=8,
                memory_capacity=64,
                checkpoint_every=1,
                evaluation_pairs=1,
                learning_curve_pairs=1,
                learning_curve_bootstrap_resamples=16,
                until_win_rate=0.7,
                eval_every_traversals=2,
                eval_games=2,
                max_traversals=20,
                seed=17,
                device="cpu",
                output_dir=Path(temporary),
            )
            trainer = DeepCFRTrainer(
                config,
                base_configuration=configuration,
                output=StringIO(),
            )
            rates = [0.4, 0.75, 0.8]

            def fake_evaluate(*_args, **kwargs):
                rate = rates.pop(0) if rates else 0.8
                games = int(kwargs["pairs"]) * 2
                wins = int(round(rate * games))
                return {
                    "round_horizon": 1,
                    "stage_index": int(kwargs["stage_index"]),
                    "stage_iteration": int(kwargs["stage_iteration"]),
                    "global_iteration": int(kwargs["global_iteration"]),
                    "stage_traversals": int(kwargs["stage_traversal_count"]),
                    "cumulative_traversals": int(kwargs["cumulative_traversal_count"]),
                    "evaluation_pairs": int(kwargs["pairs"]),
                    "evaluation_games": games,
                    "wins": wins,
                    "losses": games - wins,
                    "ties": 0,
                    "win_rate": float(wins / games),
                    "win_rate_ci95_lower": max(0.0, rate - 0.05),
                    "win_rate_ci95_upper": min(1.0, rate + 0.05),
                    "score": float(rate),
                    "score_ci95_lower": max(0.0, rate - 0.05),
                    "score_ci95_upper": min(1.0, rate + 0.05),
                    "mean_utility": float(rate - 0.5),
                    "mean_final_cash_differential": float(rate),
                }

            with patch(
                "stockpile.training.trainer.evaluate_learning_curve_checkpoint",
                side_effect=fake_evaluate,
            ):
                result = trainer.train()

            self.assertTrue(result.target_reached)
            self.assertGreaterEqual(result.final_win_rate, 0.7)
            self.assertEqual(result.cumulative_traversals, 6)
            history = Path(temporary) / "evaluation_history.csv"
            self.assertTrue(history.is_file())
            rows = history.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 4)
            archived = sorted((Path(temporary) / "checkpoints").iterdir())
            self.assertEqual(len(archived), 3)
            self.assertTrue((archived[-1] / "full.pt").is_file())
            self.assertTrue((archived[-1] / "policy.pt").is_file())

    def test_until_win_rate_stops_at_max_traversals_without_target(self):
        from stockpile.training.trainer import DeepCFRTrainer

        configuration = stockpile.resolve_configuration("lite", round_count=1)
        with TemporaryDirectory() as temporary:
            config = DeepCFRConfig(
                curriculum=CurriculumConfig((1,)),
                iterations_per_stage=1,
                traversals_per_player=1,
                advantage_train_steps=1,
                strategy_train_steps=1,
                batch_size=8,
                memory_capacity=64,
                checkpoint_every=1,
                evaluation_pairs=1,
                learning_curve_bootstrap_resamples=8,
                until_win_rate=0.95,
                eval_every_traversals=2,
                eval_games=2,
                max_traversals=6,
                seed=3,
                device="cpu",
                output_dir=Path(temporary),
            )
            trainer = DeepCFRTrainer(
                config,
                base_configuration=configuration,
                output=StringIO(),
            )

            def fake_evaluate(*_args, **kwargs):
                games = int(kwargs["pairs"]) * 2
                return {
                    "round_horizon": 1,
                    "stage_index": int(kwargs["stage_index"]),
                    "stage_iteration": int(kwargs["stage_iteration"]),
                    "global_iteration": int(kwargs["global_iteration"]),
                    "stage_traversals": int(kwargs["stage_traversal_count"]),
                    "cumulative_traversals": int(kwargs["cumulative_traversal_count"]),
                    "evaluation_pairs": int(kwargs["pairs"]),
                    "evaluation_games": games,
                    "wins": 1,
                    "losses": games - 1,
                    "ties": 0,
                    "win_rate": 1.0 / games,
                    "win_rate_ci95_lower": 0.0,
                    "win_rate_ci95_upper": 0.5,
                    "score": 0.25,
                    "score_ci95_lower": 0.0,
                    "score_ci95_upper": 0.5,
                    "mean_utility": -0.25,
                    "mean_final_cash_differential": -1.0,
                }

            stream = StringIO()
            trainer.output = stream
            with patch(
                "stockpile.training.trainer.evaluate_learning_curve_checkpoint",
                side_effect=fake_evaluate,
            ):
                result = trainer.train()

            self.assertFalse(result.target_reached)
            self.assertEqual(result.cumulative_traversals, 6)
            self.assertIsNotNone(result.final_win_rate)
            self.assertIn("MAX TRAVERSALS REACHED", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
