"""Focused contracts for signed-regret sidecars and offline bootstrap analysis."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from stockpile.training.encoding import ACTION_COUNT
from stockpile.training.regret import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE,
    REGRET_SIDECAR_SCHEMA_VERSION,
    RegretArchiveError,
    RegretFormatError,
    RegretIterationCapture,
    RegretSidecarArchive,
    RegretTraversalCapture,
    analysis_report_path,
    analyze_regret,
    analyze_run,
    load_regret_sidecar,
    regret_sidecar_path,
    write_analysis_report,
)


def _identifier(value: int) -> str:
    return f"{value:064x}"


def _mask(*actions: int) -> np.ndarray:
    result = np.zeros(ACTION_COUNT, dtype=np.bool_)
    result[np.asarray(actions, dtype=np.intp)] = True
    return result


def _target(values: dict[int, float]) -> np.ndarray:
    result = np.zeros(ACTION_COUNT, dtype=np.float64)
    for action, value in values.items():
        result[action] = value
    return result


def _traversal(
    player: int,
    ordinal: int,
    observations: list[tuple[str, dict[int, float]]],
):
    capture = RegretTraversalCapture(player, ordinal)
    for identifier, values in observations:
        actions = tuple(sorted(values))
        capture.add_target(
            perfect_recall_id=identifier,
            legal_mask=_mask(*actions),
            target=_target(values),
        )
    return capture.finish()


def _iteration(
    stage_iteration: int,
    player_0: list[list[tuple[str, dict[int, float]]]],
    player_1: list[list[tuple[str, dict[int, float]]]],
    *,
    stage_index: int = 0,
    round_count: int = 1,
    global_iteration: int | None = None,
):
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


class CaptureTests(unittest.TestCase):
    def test_capture_copies_sparse_signed_targets_and_compacts_duplicate_ids(self):
        identifier = _identifier(1)
        legal_mask = _mask(0, 2)
        first = _target({0: 4.0, 2: -3.0})
        second = _target({0: -7.0, 2: 5.0})
        capture = RegretTraversalCapture(0, 0)
        capture.add_target(
            perfect_recall_id=identifier,
            legal_mask=legal_mask,
            target=first,
        )
        capture.add_target(
            perfect_recall_id=identifier,
            legal_mask=legal_mask,
            target=second,
        )
        first[:] = 999.0
        second[:] = 999.0

        record = capture.finish()

        self.assertIs(record, capture.finish())
        self.assertEqual(len(record.observations), 1)
        observation = record.observations[0]
        self.assertEqual(observation.perfect_recall_id, identifier)
        self.assertEqual(observation.action_ids, (0, 2))
        self.assertEqual(observation.target_values, (-3.0, 2.0))
        np.testing.assert_array_equal(
            observation.legal_mask,
            legal_mask,
        )
        np.testing.assert_allclose(
            observation.dense_target(),
            _target({0: -3.0, 2: 2.0}),
        )
        with self.assertRaisesRegex(RegretArchiveError, "after traversal finish"):
            capture.add_target(
                perfect_recall_id=_identifier(2),
                legal_mask=_mask(0),
                target=_target({0: 1.0}),
            )

    def test_capture_rejects_mask_mismatch_illegal_values_and_nonfinite_values(self):
        identifier = _identifier(3)
        capture = RegretTraversalCapture(0, 0)
        capture.add_target(
            perfect_recall_id=identifier,
            legal_mask=_mask(0),
            target=_target({0: 1.0}),
        )
        capture.add_target(
            perfect_recall_id=identifier,
            legal_mask=_mask(1),
            target=_target({1: 1.0}),
        )
        with self.assertRaisesRegex(RegretFormatError, "inconsistent legal masks"):
            capture.finish()

        invalid = _target({0: 1.0, 1: 2.0})
        with self.assertRaisesRegex(RegretFormatError, "zero on illegal"):
            RegretTraversalCapture(0, 0).add_target(
                perfect_recall_id=_identifier(4),
                legal_mask=_mask(0),
                target=invalid,
            )
        nonfinite = _target({0: np.inf})
        with self.assertRaisesRegex(RegretFormatError, "finite"):
            RegretTraversalCapture(0, 0).add_target(
                perfect_recall_id=_identifier(5),
                legal_mask=_mask(0),
                target=nonfinite,
            )


class SidecarTests(unittest.TestCase):
    def test_atomic_npz_round_trip_is_sparse_no_pickle_and_integrity_checked(self):
        iteration = _iteration(
            1,
            [
                [(_identifier(10), {0: 4.0, 1: -1.0})],
                [(_identifier(10), {0: -2.0, 1: 3.0})],
            ],
            [
                [(_identifier(11), {0: 2.0, 1: -1.0})],
                [],
            ],
        )
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            archive = RegretSidecarArchive(run_dir)
            path = archive.commit(iteration)

            self.assertEqual(
                path,
                regret_sidecar_path(
                    run_dir,
                    stage_index=0,
                    round_count=1,
                    stage_iteration=1,
                ),
            )
            self.assertEqual(archive.commit(iteration), path)
            loaded = load_regret_sidecar(path)
            self.assertEqual(loaded.stage_iteration, 1)
            self.assertEqual(len(loaded.traversals), 4)
            self.assertEqual(
                loaded.traversals[0].observations[0].target_values,
                (4.0, -1.0),
            )

            with np.load(path, allow_pickle=False) as values:
                self.assertTrue(values.files)
                self.assertTrue(
                    all(not values[name].dtype.hasobject for name in values.files)
                )
                arrays = {
                    name: np.array(values[name], copy=True) for name in values.files
                }
            arrays["signed_targets"][0] += 1.0
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(RegretFormatError, "integrity digest"):
                load_regret_sidecar(path)

    def test_conflicting_iteration_cannot_overwrite_existing_sidecar(self):
        original = _iteration(1, [[]], [[]])
        conflicting = _iteration(
            1,
            [[(_identifier(20), {0: 1.0})]],
            [[]],
        )
        with TemporaryDirectory() as temporary:
            archive = RegretSidecarArchive(temporary)
            archive.commit(original)
            with self.assertRaisesRegex(RegretArchiveError, "conflicting"):
                archive.commit(conflicting)

    def test_checkpoint_state_validates_purely_and_rehydrates_a_fork(self):
        first = _iteration(1, [[]], [[]])
        second = _iteration(2, [[]], [[]])
        third = _iteration(3, [[]], [[]])
        with TemporaryDirectory() as source_name, TemporaryDirectory() as fork_name:
            source = RegretSidecarArchive(source_name)
            source.commit(first)
            source.commit(second)
            state = source.checkpoint_state(embed_records=True)
            source_bytes = {
                path.relative_to(source.run_dir): path.read_bytes()
                for path in source.sidecars()
            }

            RegretSidecarArchive.validate_checkpoint_state(state)
            corrupt = copy.deepcopy(state)
            corrupt["records"][0]["payload"] = (
                corrupt["records"][0]["payload"][:-1] + b"x"
            )
            with self.assertRaisesRegex(RegretFormatError, "hash mismatch"):
                RegretSidecarArchive.validate_checkpoint_state(corrupt)

            fork = RegretSidecarArchive(fork_name)
            fork.commit(third)
            restored = fork.restore_checkpoint_state(state)

            self.assertEqual(len(restored.restored), 2)
            self.assertEqual(len(restored.archived), 1)
            self.assertTrue(all(path.is_file() for path in restored.archived))
            self.assertEqual(len(fork.sidecars()), 2)
            for relative, expected in source_bytes.items():
                self.assertEqual((fork.run_dir / relative).read_bytes(), expected)

            manifest_only = source.checkpoint_state(embed_records=False)
            empty = RegretSidecarArchive(Path(fork_name) / "empty")
            with self.assertRaisesRegex(RegretArchiveError, "no embedded copy"):
                empty.restore_checkpoint_state(manifest_only)


class AnalysisTests(unittest.TestCase):
    @staticmethod
    def _two_iteration_records():
        information_a = _identifier(100)
        information_b = _identifier(101)
        information_c = _identifier(102)
        first = _iteration(
            1,
            [
                [(information_a, {0: 4.0, 1: -1.0})],
                [(information_a, {0: -2.0, 1: 3.0})],
            ],
            [
                [(information_c, {0: 2.0, 1: -1.0})],
                [(information_c, {0: 2.0, 1: -1.0})],
            ],
        )
        second = _iteration(
            2,
            [
                [
                    (information_a, {0: -6.0, 1: 1.0}),
                    (information_b, {0: 8.0, 1: -1.0}),
                ],
                [(information_a, {0: 2.0, 1: -5.0})],
            ],
            [
                [(information_c, {0: -3.0, 1: 3.0})],
                [(information_c, {0: -3.0, 1: 3.0})],
            ],
        )
        return first, second

    def test_signed_point_averages_traversals_and_emits_every_outer_prefix(self):
        with TemporaryDirectory() as temporary:
            archive = RegretSidecarArchive(temporary)
            for iteration in self._two_iteration_records():
                archive.commit(iteration)

            report = analyze_run(
                temporary,
                confidence=0.90,
                bootstrap_replicates=512,
                seed=0,
            )
            repeated = analyze_run(
                temporary,
                confidence=0.90,
                bootstrap_replicates=512,
                seed=0,
            )

            self.assertEqual(report, repeated)
            self.assertTrue(report["availability"]["available"])
            self.assertEqual(report["availability"]["source"], "iteration_sidecars")
            stage = report["stages"][0]
            self.assertTrue(stage["complete_prefix"])
            self.assertEqual(len(stage["series"]), 2)
            self.assertEqual(
                stage["series"][0]["point"],
                {"player_0": 1.0, "player_1": 2.0, "maximum": 2.0},
            )
            self.assertEqual(
                stage["series"][1]["point"],
                {"player_0": 2.0, "player_1": 1.0, "maximum": 2.0},
            )
            self.assertEqual(stage["point"], stage["series"][-1]["point"])
            self.assertEqual(
                stage["confidence_interval"],
                stage["series"][-1]["confidence_interval"],
            )
            for entry in stage["series"]:
                for interval in entry["confidence_interval"].values():
                    self.assertEqual(len(interval), 2)
                    self.assertTrue(all(np.isfinite(interval)))

    def test_bootstrap_resamples_whole_traversals_inside_fixed_stratum(self):
        # Both IDs live in the same traversal, so their bootstrap weights must
        # move together. With N=2, p0 is 10 * count(traversal 0).
        first = _iteration(
            1,
            [
                [
                    (_identifier(200), {0: 10.0}),
                    (_identifier(201), {0: 10.0}),
                ],
                [],
            ],
            [[], []],
        )
        with TemporaryDirectory() as temporary:
            RegretSidecarArchive(temporary).commit(first)
            report = analyze_regret(
                temporary,
                replicates=1_024,
                confidence=0.50,
                seed=7,
            )

        stage = report["stages"][0]
        self.assertEqual(stage["point"]["player_0"], 10.0)
        lower, upper = stage["confidence_interval"]["player_0"]
        # Clustered support is {0, 10, 20}; central empirical quantiles cannot
        # produce the 5/15 pair characteristic of four independent rows.
        self.assertIn(lower, (0.0, 10.0))
        self.assertIn(upper, (10.0, 20.0))

    def test_suffix_only_prefix_is_unavailable_and_statistics_are_json_null(self):
        incomplete = _iteration(
            2,
            [[(_identifier(300), {0: 1.0})]],
            [[]],
        )
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            RegretSidecarArchive(run_dir).commit(incomplete)
            report = analyze_run(
                run_dir,
                bootstrap_replicates=32,
            )

            self.assertFalse(report["availability"]["available"])
            stage = report["stages"][0]
            self.assertFalse(stage["complete_prefix"])
            self.assertIsNone(stage["point"]["maximum"])
            self.assertIsNone(stage["series"][0]["confidence_interval"]["maximum"])
            destination = write_analysis_report(run_dir, report)

            self.assertEqual(destination, analysis_report_path(run_dir))
            decoded = json.loads(destination.read_text(encoding="utf-8"))
            self.assertIsNone(decoded["stages"][0]["point"]["maximum"])
            self.assertNotIn("NaN", destination.read_text(encoding="utf-8"))

    def test_legacy_run_is_na_without_loading_torch_and_defaults_are_exact(self):
        with TemporaryDirectory() as temporary:
            with patch.dict(sys.modules, {"torch": None}):
                report = analyze_run(temporary)

        self.assertFalse(report["availability"]["available"])
        self.assertEqual(report["stages"], [])
        self.assertEqual(
            report["bootstrap"],
            {
                "method": "stratified_complete_traversal_percentile",
                "replicates": DEFAULT_BOOTSTRAP_REPLICATES,
                "confidence": DEFAULT_CONFIDENCE,
                "seed": DEFAULT_BOOTSTRAP_SEED,
                "bit_generator": "PCG64",
            },
        )

    def test_marked_v2_run_falls_back_to_self_contained_checkpoint(self):
        with TemporaryDirectory() as source_name, TemporaryDirectory() as run_name:
            source = RegretSidecarArchive(source_name)
            source.commit(self._two_iteration_records()[0])
            state = source.checkpoint_state(embed_records=True)
            run_dir = Path(run_name)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "sampled_regret_telemetry": {
                            "record_schema_version": REGRET_SIDECAR_SCHEMA_VERSION
                        }
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = run_dir / "round_01" / "full.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"placeholder")
            fake_torch = SimpleNamespace(
                load=lambda *_args, **_kwargs: {
                    "kind": "stockpile_deep_cfr_training",
                    "schema_version": 2,
                    "sampled_regret_telemetry": state,
                }
            )

            with patch.dict(sys.modules, {"torch": fake_torch}):
                report = analyze_run(run_dir, bootstrap_replicates=64)

        self.assertTrue(report["availability"]["available"])
        self.assertEqual(report["availability"]["source"], "embedded_checkpoint")
        self.assertEqual(len(report["stages"]), 1)

    def test_explicit_v2_full_is_self_contained_without_run_metadata(self):
        with TemporaryDirectory() as source_name, TemporaryDirectory() as run_name:
            source = RegretSidecarArchive(source_name)
            source.commit(self._two_iteration_records()[0])
            state = source.checkpoint_state(embed_records=True)
            checkpoint = Path(run_name) / "copied" / "full.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"placeholder")
            fake_torch = SimpleNamespace(
                load=lambda *_args, **_kwargs: {
                    "kind": "stockpile_deep_cfr_training",
                    "schema_version": 2,
                    "sampled_regret_telemetry": state,
                }
            )

            with patch.dict(sys.modules, {"torch": fake_torch}):
                report = analyze_run(checkpoint, bootstrap_replicates=64)

        self.assertTrue(report["availability"]["available"])
        self.assertTrue(report["availability"]["telemetry_declared"])
        self.assertEqual(report["availability"]["source"], "embedded_checkpoint")

    def test_gap_sidecars_merge_embedded_checkpoint_before_declaring_na(self):
        first, second = self._two_iteration_records()
        with TemporaryDirectory() as source_name, TemporaryDirectory() as run_name:
            source = RegretSidecarArchive(source_name)
            source.commit(first)
            state = source.checkpoint_state(embed_records=True)
            run_dir = Path(run_name)
            RegretSidecarArchive(run_dir).commit(second)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "sampled_regret_telemetry": {
                            "record_schema_version": REGRET_SIDECAR_SCHEMA_VERSION
                        }
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = run_dir / "round_01" / "full.pt"
            checkpoint.write_bytes(b"placeholder")
            fake_torch = SimpleNamespace(
                load=lambda *_args, **_kwargs: {
                    "kind": "stockpile_deep_cfr_training",
                    "schema_version": 2,
                    "sampled_regret_telemetry": state,
                }
            )

            with patch.dict(sys.modules, {"torch": fake_torch}):
                report = analyze_run(run_dir, bootstrap_replicates=64)

        self.assertTrue(report["availability"]["available"])
        self.assertEqual(
            report["availability"]["source"],
            "iteration_sidecars_plus_embedded_checkpoint",
        )
        self.assertEqual(len(report["stages"][0]["series"]), 2)
        self.assertEqual(report["stages"][0]["point"]["maximum"], 2.0)

    def test_explicit_policy_is_na_even_when_sibling_sidecars_exist(self):
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            RegretSidecarArchive(run_dir).commit(self._two_iteration_records()[0])
            policy = run_dir / "round_01" / "policy.pt"
            policy.write_bytes(b"not inspected")

            report = analyze_run(policy)

        self.assertFalse(report["availability"]["available"])
        self.assertEqual(report["availability"]["reason"], "policy_has_no_sampled_regret")

    def test_contiguous_sidecars_never_load_sibling_full_checkpoint(self):
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            RegretSidecarArchive(run_dir).commit(self._two_iteration_records()[0])
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "sampled_regret_telemetry": {
                            "record_schema_version": REGRET_SIDECAR_SCHEMA_VERSION
                        }
                    }
                ),
                encoding="utf-8",
            )
            full = run_dir / "round_01" / "full.pt"
            full.write_bytes(b"must not load")
            fake_torch = SimpleNamespace(
                load=lambda *_args, **_kwargs: self.fail(
                    "contiguous sidecars must not load full.pt"
                )
            )

            with patch.dict(sys.modules, {"torch": fake_torch}):
                report = analyze_run(run_dir, bootstrap_replicates=32)

        self.assertTrue(report["availability"]["available"])
        self.assertEqual(report["availability"]["source"], "iteration_sidecars")

    def test_confidence_and_bootstrap_count_are_strictly_validated(self):
        for confidence in (0.0, 1.0, np.nan, np.inf):
            with self.subTest(confidence=confidence):
                with self.assertRaises(RegretFormatError):
                    analyze_regret([], confidence=confidence)
        with self.assertRaises(RegretFormatError):
            analyze_regret([], replicates=0)


if __name__ == "__main__":
    unittest.main()
