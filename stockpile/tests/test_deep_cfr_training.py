"""End-to-end contracts for the optional sampled Deep CFR trainer."""

from __future__ import annotations

from dataclasses import replace
from io import StringIO
import importlib.util
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
            seed=91,
            device="cpu",
            output_dir=output_dir,
        )

    def test_one_round_train_export_and_exact_checkpoint_restore(self):
        import torch

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
            metrics_before_resume = metrics_path.read_text(encoding="utf-8")
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
            self.assertEqual(
                metrics_path.read_text(encoding="utf-8"),
                metrics_before_resume,
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
            with self.assertRaisesRegex(ValueError, "only Stockpile Lite"):
                DeepCFRTrainer(
                    config,
                    base_configuration=stockpile.resolve_configuration(
                        "classic",
                        round_count=1,
                    ),
                    output=StringIO(),
                )


if __name__ == "__main__":
    unittest.main()
