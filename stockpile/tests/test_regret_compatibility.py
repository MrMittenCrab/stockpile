"""Regression contracts for sampled-regret arithmetic and legacy artifacts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import stockpile
from stockpile.training.config import (
    CurriculumConfig,
    DeepCFRConfig,
    NetworkConfig,
)
from stockpile.training.encoding import (
    ACTION_COUNT,
    ENCODING_SCHEMA_VERSION,
)
from stockpile.training.regret import (
    IterationRegretRecord,
    RegretFormatError,
    RegretIterationCapture,
    RegretSidecarArchive,
    RegretTraversalCapture,
    TraversalRegretRecord,
    analyze_regret,
    analyze_run,
    load_regret_sidecar,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _identifier(value: int) -> str:
    return f"{value:064x}"


def _traversal(
    player: int,
    ordinal: int,
    observations: tuple[tuple[str, tuple[float, ...]], ...] = (),
) -> TraversalRegretRecord:
    capture = RegretTraversalCapture(player, ordinal)
    for identifier, legal_values in observations:
        legal_mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
        legal_mask[: len(legal_values)] = True
        target = np.zeros(ACTION_COUNT, dtype=np.float64)
        target[: len(legal_values)] = np.asarray(
            legal_values,
            dtype=np.float64,
        )
        capture.add_target(
            perfect_recall_id=identifier,
            legal_mask=legal_mask,
            target=target,
        )
    return capture.finish()


def _iteration(
    stage_iteration: int,
    player_0: tuple[tuple[tuple[str, tuple[float, ...]], ...], ...],
    player_1: tuple[tuple[tuple[str, tuple[float, ...]], ...], ...],
    *,
    stage_index: int = 0,
    round_count: int = 1,
    global_iteration: int | None = None,
) -> IterationRegretRecord:
    capture = RegretIterationCapture(
        stage_index=stage_index,
        round_count=round_count,
        stage_iteration=stage_iteration,
        global_iteration=(
            stage_iteration if global_iteration is None else global_iteration
        ),
    )
    for player, traversals in enumerate((player_0, player_1)):
        for ordinal, observations in enumerate(traversals):
            capture.add_traversal(
                _traversal(player, ordinal, observations)
            )
    return capture.finish()


def _empty_stratum(count: int = 2):
    return tuple(() for _ in range(count))


class SignedRegretArithmeticTests(unittest.TestCase):
    def test_duplicate_signed_targets_cancel_without_absolute_values(self):
        identifier = _identifier(1)
        capture = RegretTraversalCapture(0, 0)
        legal_mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
        legal_mask[:2] = True
        for values in ((7.5, -3.25), (-7.5, 3.25)):
            target = np.zeros(ACTION_COUNT, dtype=np.float64)
            target[:2] = values
            capture.add_target(
                perfect_recall_id=identifier,
                legal_mask=legal_mask,
                target=target,
            )

        observation = capture.finish().observations[0]

        self.assertEqual(observation.target_values, (0.0, 0.0))

    def test_traversal_mean_counts_an_unvisited_traversal_as_zero(self):
        identifier = _identifier(2)
        record = _iteration(
            1,
            (
                ((identifier, (4.0, -4.0)),),
                (),
            ),
            _empty_stratum(),
        )
        with TemporaryDirectory() as temporary:
            RegretSidecarArchive(temporary).commit(record)
            report = analyze_regret(temporary, replicates=8, seed=3)

        point = report["stages"][0]["series"][0]["point"]
        self.assertEqual(point["player_0"], 2.0)
        self.assertEqual(point["player_1"], 0.0)
        self.assertEqual(point["maximum"], 2.0)

    def test_positive_part_is_applied_after_signed_prefix_accumulation(self):
        identifier = _identifier(3)
        records = (
            _iteration(
                1,
                (
                    ((identifier, (4.0, -4.0)),),
                    (),
                ),
                _empty_stratum(),
            ),
            _iteration(
                2,
                (
                    ((identifier, (-4.0, 4.0)),),
                    (),
                ),
                _empty_stratum(),
            ),
        )
        with TemporaryDirectory() as temporary:
            archive = RegretSidecarArchive(temporary)
            for record in records:
                archive.commit(record)
            report = analyze_regret(temporary, replicates=8, seed=3)

        series = report["stages"][0]["series"]
        self.assertEqual(
            [entry["point"]["player_0"] for entry in series],
            [2.0, 0.0],
        )

    def test_every_available_iteration_is_reported_and_stages_reset(self):
        identifier = _identifier(4)
        records = (
            _iteration(
                1,
                (((identifier, (2.0, -2.0)),),),
                ((),),
            ),
            _iteration(
                2,
                (((identifier, (2.0, -2.0)),),),
                ((),),
            ),
            _iteration(
                1,
                (((identifier, (6.0, -6.0)),),),
                ((),),
                stage_index=1,
                round_count=2,
                global_iteration=3,
            ),
        )
        with TemporaryDirectory() as temporary:
            archive = RegretSidecarArchive(temporary)
            for record in records:
                archive.commit(record)
            report = analyze_regret(temporary, replicates=4, seed=11)

        self.assertEqual(
            [stage["stage_index"] for stage in report["stages"]],
            [0, 1],
        )
        self.assertEqual(
            [len(stage["series"]) for stage in report["stages"]],
            [2, 1],
        )
        self.assertEqual(
            [
                entry["stage_iteration"]
                for entry in report["stages"][0]["series"]
            ],
            [1, 2],
        )
        self.assertEqual(
            report["stages"][1]["series"][0]["point"]["player_0"],
            6.0,
        )

    def test_fixed_seed_ci_matches_complete_traversal_resampling(self):
        identifier = _identifier(5)
        record = _iteration(
            1,
            (
                ((identifier, (4.0, -4.0)),),
                (),
            ),
            _empty_stratum(),
        )
        replicates = 17
        confidence = 0.80
        seed = 29
        with TemporaryDirectory() as temporary:
            RegretSidecarArchive(temporary).commit(record)
            first = analyze_regret(
                temporary,
                replicates=replicates,
                confidence=confidence,
                seed=seed,
            )
            second = analyze_regret(
                temporary,
                replicates=replicates,
                confidence=confidence,
                seed=seed,
            )

        self.assertEqual(first, second)
        rng = np.random.Generator(np.random.PCG64(seed))
        counts = rng.multinomial(
            2,
            np.asarray((0.5, 0.5), dtype=np.float64),
            size=replicates,
        )
        bootstrap_statistics = counts[:, 0].astype(np.float64) * 2.0
        expected = np.quantile(bootstrap_statistics, (0.10, 0.90))
        actual = first["stages"][0]["series"][0]["confidence_interval"]
        np.testing.assert_allclose(actual["player_0"], expected)
        np.testing.assert_allclose(actual["maximum"], expected)

    def test_confidence_level_must_be_finite_and_strictly_between_bounds(self):
        for confidence in (-1.0, 0.0, 1.0, 2.0, np.nan, np.inf, -np.inf):
            with self.subTest(confidence=confidence), self.assertRaises(
                RegretFormatError
            ):
                analyze_regret((), confidence=confidence)


class RegretPersistenceTests(unittest.TestCase):
    def test_sidecar_round_trip_preserves_exact_signed_float64_values(self):
        positive = np.nextafter(np.float64(1.0), np.float64(2.0))
        negative = -np.nextafter(np.float64(3.0), np.float64(4.0))
        record = _iteration(
            1,
            (((_identifier(10), (positive, negative)),),),
            ((),),
        )
        with TemporaryDirectory() as temporary:
            archive = RegretSidecarArchive(temporary)
            path = archive.commit(record)
            original_bytes = path.read_bytes()
            loaded = load_regret_sidecar(path)
            archive.commit(record)

            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(loaded, record)
            original_target = record.traversals[0].observations[0].dense_target()
            loaded_target = loaded.traversals[0].observations[0].dense_target()
            np.testing.assert_array_equal(
                original_target.view(np.uint64),
                loaded_target.view(np.uint64),
            )
            with np.load(path, allow_pickle=False) as arrays:
                self.assertTrue(
                    all(not arrays[name].dtype.hasobject for name in arrays.files)
                )

    def test_checkpoint_restore_archives_tail_and_restores_exact_snapshot(self):
        first = _iteration(1, ((),), ((),))
        second = _iteration(2, ((),), ((),))
        with TemporaryDirectory() as temporary:
            archive = RegretSidecarArchive(temporary)
            first_path = archive.commit(first)
            state = archive.checkpoint_state(embed_records=True)
            second_path = archive.commit(second)
            second_bytes = second_path.read_bytes()
            corrupt_bytes = b"not a regret sidecar"
            first_path.write_bytes(corrupt_bytes)

            result = archive.restore_checkpoint_state(state)

            self.assertEqual(result.kept, ())
            self.assertEqual(len(result.restored), 1)
            self.assertEqual(load_regret_sidecar(first_path), first)
            self.assertFalse(second_path.exists())
            archived_bytes = {path.read_bytes() for path in result.archived}
            self.assertEqual(archived_bytes, {corrupt_bytes, second_bytes})


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch training dependency")
class TrainerArtifactCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _config(output_dir: Path) -> DeepCFRConfig:
        return DeepCFRConfig(
            curriculum=CurriculumConfig((1,)),
            network=NetworkConfig(
                observation_hidden=4,
                history_hidden=4,
                event_hidden=4,
                fusion_hidden=4,
            ),
            iterations_per_stage=1,
            traversals_per_player=1,
            advantage_train_steps=1,
            strategy_train_steps=1,
            batch_size=1,
            memory_capacity=8,
            checkpoint_every=1,
            evaluation_pairs=1,
            learning_curve_pairs=1,
            learning_curve_bootstrap_resamples=8,
            seed=313,
            device="cpu",
            output_dir=output_dir,
        )

    @staticmethod
    def _initialized_trainer(output_dir: Path):
        from stockpile.training.trainer import DeepCFRTrainer

        config = TrainerArtifactCompatibilityTests._config(output_dir)
        configuration = stockpile.resolve_configuration(
            "lite",
            round_count=1,
        )
        trainer = DeepCFRTrainer(
            config,
            base_configuration=configuration,
            output=StringIO(),
        )
        trainer._reset_stage(0, transfer_weights=False)
        return trainer, config, configuration

    def test_v2_trainer_checkpoint_rehydrates_sidecar_and_archives_tail(self):
        import torch

        from stockpile.training.trainer import CHECKPOINT_SCHEMA_VERSION, DeepCFRTrainer

        first = _iteration(1, ((),), ((),))
        second = _iteration(2, ((),), ((),))
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            trainer, config, configuration = self._initialized_trainer(run_dir)
            first_path = trainer.regret_archive.commit(first)
            first_bytes = first_path.read_bytes()
            trainer.stage_iteration = 1
            trainer.global_iteration = 1
            checkpoint, _policy = trainer.save_checkpoint()
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

            self.assertEqual(payload["schema_version"], CHECKPOINT_SCHEMA_VERSION)
            self.assertEqual(
                payload["sampled_regret_telemetry"]["records"][0]["payload"],
                first_bytes,
            )

            second_path = trainer.regret_archive.commit(second)
            second_bytes = second_path.read_bytes()
            first_path.unlink()
            restored = DeepCFRTrainer(
                config,
                base_configuration=configuration,
                output=StringIO(),
            )
            restored.load_checkpoint(checkpoint)

            self.assertEqual(first_path.read_bytes(), first_bytes)
            self.assertFalse(second_path.exists())
            archived = tuple(
                run_dir.glob(
                    "analysis/sampled_regret_archive/**/*.bak"
                )
            )
            self.assertIn(second_bytes, {path.read_bytes() for path in archived})
            self.assertEqual(restored.stage_iteration, 1)
            self.assertEqual(restored.global_iteration, 1)

    def test_schema_v1_full_checkpoint_without_regret_history_still_loads(self):
        import torch

        from stockpile.training.trainer import (
            LEGACY_CHECKPOINT_SCHEMA_VERSION,
            DeepCFRTrainer,
        )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            trainer, _config, configuration = self._initialized_trainer(source)
            full_path, _policy_path = trainer.save_checkpoint()
            payload = torch.load(full_path, map_location="cpu", weights_only=False)
            payload["schema_version"] = LEGACY_CHECKPOINT_SCHEMA_VERSION
            payload.pop("sampled_regret_telemetry")
            payload.pop("stage_evaluated", None)
            payload["metadata"]["checkpoint_schema"] = (
                LEGACY_CHECKPOINT_SCHEMA_VERSION
            )
            payload["metadata"].pop(
                "sampled_regret_sidecar_schema_version",
                None,
            )
            legacy_path = root / "legacy" / "round_01" / "full.pt"
            legacy_path.parent.mkdir(parents=True)
            torch.save(payload, legacy_path)
            source_digest = hashlib.sha256(legacy_path.read_bytes()).hexdigest()

            restored = DeepCFRTrainer(
                self._config(destination),
                base_configuration=configuration,
                output=StringIO(),
            )
            restored.load_checkpoint(legacy_path)

            self.assertEqual(restored.stage_index, 0)
            self.assertEqual(restored.stage_iteration, 0)
            self.assertFalse(restored.stage_evaluated)
            self.assertEqual(RegretSidecarArchive(destination).sidecars(), ())
            self.assertEqual(
                hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
                source_digest,
            )
            report = analyze_run(legacy_path, bootstrap_replicates=4)
            self.assertFalse(report["availability"]["available"])
            self.assertEqual(report["stages"], [])

    def test_legacy_policy_without_regret_metadata_remains_loadable(self):
        import torch

        from stockpile.training.models import DeepCFRNetwork
        from stockpile.training.policy import (
            ACTION_SCHEMA_VERSION,
            POLICY_SCHEMA_VERSION,
            DeepCFRPolicy,
        )

        with TemporaryDirectory() as temporary:
            network_config = self._config(Path(temporary)).network
            original = DeepCFRNetwork(network_config)
            legacy_path = Path(temporary) / "policy.pt"
            torch.save(
                {
                    "kind": "stockpile_deep_cfr_policy",
                    "schema_version": POLICY_SCHEMA_VERSION,
                    "network_config": asdict(network_config),
                    "strategy_network": original.state_dict(),
                    "metadata": {
                        "encoder_schema_version": ENCODING_SCHEMA_VERSION,
                        "action_schema_version": ACTION_SCHEMA_VERSION,
                    },
                },
                legacy_path,
            )
            source_digest = hashlib.sha256(legacy_path.read_bytes()).hexdigest()

            loaded = DeepCFRPolicy.load(legacy_path)

            for name, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.network.state_dict()[name]))
            self.assertEqual(
                hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
                source_digest,
            )


if __name__ == "__main__":
    unittest.main()
