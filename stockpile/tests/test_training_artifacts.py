"""Filesystem contracts for versioned Deep CFR artifact runs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from stockpile.training.artifacts import (
    DEFAULT_ARTIFACT_ROOT,
    PROJECT_ROOT,
    RESUME_PROVENANCE_NAME,
    RESUME_PROVENANCE_SCHEMA_VERSION,
    RUN_MANIFEST_NAME,
    RUN_SCHEMA_VERSION,
    discover_legacy_runs,
    discover_runs,
    find_run_for_path,
    parse_run_name,
    plan_resume_destination,
    provenance_dict,
    read_run,
    reserve_run,
    resolve_fresh_output,
    resolve_run,
    run_name,
    update_run_manifest,
)


def _saved_config(path: Path, mode: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "base_game": {"mode": mode},
                "training": {"output_dir": "an-old-location"},
            }
        ),
        encoding="utf-8",
    )


def _checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint fixture")
    return path


def _current_saved_config(path: Path, mode: str) -> None:
    _saved_config(path, mode)
    document = json.loads((path / "config.json").read_text(encoding="utf-8"))
    document["sampled_regret_telemetry"] = {"record_schema_version": 1}
    (path / "config.json").write_text(
        json.dumps(document, sort_keys=True),
        encoding="utf-8",
    )


def _file_bytes(path: Path) -> dict[str, bytes]:
    return {
        str(file.relative_to(path)): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file()
    }


class RunReservationTests(unittest.TestCase):
    def test_default_root_is_anchored_to_the_project_not_the_working_directory(self):
        self.assertEqual(DEFAULT_ARTIFACT_ROOT, PROJECT_ROOT / "artifacts/deep_cfr")
        self.assertTrue(DEFAULT_ARTIFACT_ROOT.is_absolute())

    def test_run_names_are_positive_integer_indices(self):
        self.assertEqual(run_name(1), "run_01")
        self.assertEqual(run_name(101), "run_101")
        self.assertEqual(parse_run_name("run_01"), 1)
        for invalid in (
            "run_00",
            "run_001",
            "run_1",
            "run_-1",
            "run_01.tmp",
            "other",
        ):
            with self.subTest(invalid=invalid):
                self.assertIsNone(parse_run_name(invalid))
        for invalid in (0, -1, True, "1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                run_name(invalid)  # type: ignore[arg-type]

    def test_auto_allocation_uses_max_plus_one_and_explicit_collision_fails(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            first = reserve_run("lite", artifact_root=root)
            occupied = root / "lite" / "run_07"
            occupied.mkdir()
            (root / "lite" / "run_06.tmp").mkdir()

            next_ref = reserve_run("lite", artifact_root=root)
            explicit = reserve_run("lite", run=3, artifact_root=root)

            self.assertEqual(first.run, 1)
            self.assertEqual(next_ref.run, 8)
            self.assertEqual(explicit.run, 3)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                reserve_run("lite", run=7, artifact_root=root)

    def test_reservation_writes_a_valid_v1_manifest(self):
        with TemporaryDirectory() as temporary:
            ref = reserve_run("lite", artifact_root=temporary)
            document = json.loads(ref.manifest.read_text(encoding="utf-8"))

            self.assertEqual(document["schema_version"], RUN_SCHEMA_VERSION)
            self.assertEqual(document["source"], "managed")
            self.assertEqual(document["mode"], "lite")
            self.assertEqual(document["kind"], "normal")
            self.assertEqual(document["run"], 1)
            self.assertEqual(document["state"], "reserved")
            self.assertEqual(document["run_id"], ref.run_id)
            self.assertEqual(document["created_at"], ref.created_at)
            self.assertEqual(len(ref.run_id), 36)
            self.assertIsNotNone(
                datetime.fromisoformat(ref.created_at.replace("Z", "+00:00")).tzinfo
            )
            self.assertEqual(read_run(ref.path, artifact_root=temporary), ref)

    def test_manifest_updates_are_atomic_monotonic_and_preserve_identity(self):
        with TemporaryDirectory() as temporary:
            reserved = reserve_run("lite", artifact_root=temporary)
            active = update_run_manifest(
                reserved,
                state="active",
                provenance={"parent": "fixture"},
            )
            completed = update_run_manifest(active, state="completed")

            self.assertEqual(active.run_id, reserved.run_id)
            self.assertEqual(completed.created_at, reserved.created_at)
            self.assertEqual(completed.state, "completed")
            self.assertEqual(completed.resume_provenance, {"parent": "fixture"})
            self.assertEqual(
                read_run(completed.path, artifact_root=temporary),
                completed,
            )
            self.assertFalse(
                any(path.name.endswith(".tmp") for path in completed.path.iterdir())
            )
            with self.assertRaisesRegex(ValueError, "backward"):
                update_run_manifest(completed, state="active")

    def test_reader_rejects_incomplete_v1_manifests(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "lite/run_01"
            path.mkdir(parents=True)
            (path / RUN_MANIFEST_NAME).write_text(
                json.dumps(
                    {
                        "schema_version": RUN_SCHEMA_VERSION,
                        "source": "managed",
                        "mode": "lite",
                        "kind": "normal",
                        "run": 1,
                    }
                ),
                encoding="utf-8",
            )

            self.assertIsNone(read_run(path, artifact_root=root))

    def test_smoke_has_an_independent_numbered_namespace(self):
        with TemporaryDirectory() as temporary:
            normal = reserve_run("lite", artifact_root=temporary)
            smoke = reserve_run("lite", smoke=True, artifact_root=temporary)

            root = Path(temporary).resolve()
            self.assertEqual(normal.path, root / "lite/run_01")
            self.assertEqual(smoke.path, root / "smoke/run_01")
            self.assertEqual(smoke.run, 1)
            self.assertTrue(smoke.smoke)

    def test_concurrent_automatic_reservations_are_unique(self):
        with TemporaryDirectory() as temporary:
            def allocate(_index: int):
                return reserve_run("lite", artifact_root=temporary)

            with ThreadPoolExecutor(max_workers=8) as executor:
                refs = tuple(executor.map(allocate, range(8)))

            self.assertEqual(sorted(ref.run for ref in refs), list(range(1, 9)))
            self.assertEqual(len({ref.path for ref in refs}), 8)
            self.assertTrue(all(ref.manifest.is_file() for ref in refs))

    def test_reservation_rejects_symlinked_root_and_namespace_parents(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            real_root = base / "real-root"
            real_root.mkdir()
            root_alias = base / "root-alias"
            root_alias.symlink_to(real_root, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "artifact root.*symlink"):
                reserve_run("lite", artifact_root=root_alias)
            self.assertEqual(tuple(real_root.iterdir()), ())

            root = base / "managed"
            root.mkdir()
            external_namespace = base / "external-namespace"
            external_namespace.mkdir()
            (root / "lite").symlink_to(
                external_namespace,
                target_is_directory=True,
            )
            with self.assertRaisesRegex(ValueError, "namespace.*symlink"):
                reserve_run("lite", artifact_root=root)
            self.assertEqual(tuple(external_namespace.iterdir()), ())


class RunDiscoveryTests(unittest.TestCase):
    def test_discovery_uses_manifest_and_saved_legacy_mode(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            normal = reserve_run("lite", artifact_root=root)
            smoke = reserve_run("lite", smoke=True, artifact_root=root)
            _saved_config(root / "default", "classic")
            _saved_config(root / "smoke", "lite")

            malformed = root / "lite/run_03"
            malformed.mkdir()
            (malformed / RUN_MANIFEST_NAME).write_text("{broken", encoding="utf-8")
            temporary_run = root / "lite/run_04.tmp"
            temporary_run.mkdir()
            (temporary_run / RUN_MANIFEST_NAME).write_text("{}", encoding="utf-8")

            lite = discover_runs(mode="lite", artifact_root=root)
            classic = discover_runs(mode="classic", artifact_root=root)

            self.assertIn(normal, lite)
            self.assertIn(smoke, lite)
            self.assertIn(root / "smoke", {ref.path for ref in lite if ref.legacy})
            self.assertEqual(
                {ref.path for ref in classic if ref.legacy},
                {root / "default"},
            )
            self.assertNotIn(malformed, {ref.path for ref in lite})
            self.assertNotIn(temporary_run, {ref.path for ref in lite})
            self.assertFalse((root / "default/run.json").exists())
            self.assertFalse((root / "smoke/run.json").exists())

    def test_malformed_or_temporary_legacy_metadata_is_ignored_read_only(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            default = root / "default"
            default.mkdir()
            (default / "config.json").write_text("not json", encoding="utf-8")
            smoke = root / "smoke"
            smoke.mkdir()
            (smoke / "config.json.tmp").write_text(
                json.dumps({"base_game": {"mode": "lite"}}),
                encoding="utf-8",
            )

            before = sorted(path.name for path in root.iterdir())
            self.assertEqual(discover_legacy_runs(artifact_root=root), ())
            self.assertEqual(sorted(path.name for path in root.iterdir()), before)

    def test_resolve_run_supports_latest_legacy_fallback_and_ambiguity(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            first = reserve_run("lite", artifact_root=root)
            second = reserve_run("lite", artifact_root=root)
            smoke = reserve_run("lite", smoke=True, artifact_root=root)

            self.assertEqual(resolve_run("lite", artifact_root=root), second)
            self.assertEqual(
                resolve_run("lite", run=1, smoke=True, artifact_root=root),
                smoke,
            )
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                resolve_run("lite", run=1, smoke=None, artifact_root=root)

            another_root = root / "legacy-only"
            _saved_config(another_root / "default", "lite")
            legacy = resolve_run(
                "lite",
                artifact_root=another_root,
                include_legacy=True,
            )
            self.assertTrue(legacy.legacy)
            self.assertEqual(legacy.path, another_root / "default")
            self.assertEqual(first.run, 1)

    def test_symlink_alias_cannot_impersonate_an_explicit_run_number(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            original = reserve_run("lite", run=1, artifact_root=root)
            alias = root / "lite/run_02"
            alias.symlink_to(original.path, target_is_directory=True)

            self.assertIsNone(read_run(alias, artifact_root=root))
            with self.assertRaises(FileNotFoundError):
                resolve_run("lite", run=2, artifact_root=root)
            self.assertEqual(
                [ref.run for ref in discover_runs(
                    mode="lite",
                    artifact_root=root,
                    include_legacy=False,
                )],
                [1],
            )


class OutputAndProvenanceTests(unittest.TestCase):
    def test_fresh_output_reserves_managed_runs_or_accepts_explicit_unmanaged(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "managed"
            external = Path(temporary).resolve() / "external"

            managed = resolve_fresh_output("lite", artifact_root=root)
            unmanaged = resolve_fresh_output(
                "lite",
                output_dir=external,
                artifact_root=root,
            )

            self.assertTrue(managed.managed)
            self.assertEqual(managed.run, 1)
            self.assertEqual(unmanaged.source, "unmanaged")
            self.assertEqual(unmanaged.path, external)
            self.assertFalse(external.exists())

    def test_overwrite_is_only_valid_for_explicit_unmanaged_paths(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "managed"
            external = Path(temporary).resolve() / "external"
            external.mkdir()
            (external / "existing.txt").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "explicit unmanaged"):
                resolve_fresh_output("lite", overwrite=True, artifact_root=root)
            with self.assertRaisesRegex(ValueError, "not empty"):
                resolve_fresh_output(
                    "lite",
                    output_dir=external,
                    artifact_root=root,
                )
            accepted = resolve_fresh_output(
                "lite",
                output_dir=external,
                overwrite=True,
                artifact_root=root,
            )
            self.assertEqual(accepted.source, "unmanaged")
            self.assertTrue((external / "existing.txt").is_file())

            for reserved in (root / "lite/run_09", root / "default", root / "smoke"):
                with self.subTest(reserved=reserved), self.assertRaisesRegex(
                    ValueError, "managed|unmanaged"
                ):
                    resolve_fresh_output(
                        "lite",
                        output_dir=reserved,
                        overwrite=True,
                        artifact_root=root,
                    )
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                resolve_fresh_output(
                    "lite",
                    output_dir=external,
                    run=2,
                    artifact_root=root,
                )

    def test_explicit_unmanaged_output_must_be_outside_the_artifact_root(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "managed"
            root.mkdir()
            destinations = (
                root,
                root / "misc",
                root / "default/nested",
                root / "smoke/nested",
                root / "lite/run_09/nested",
            )
            for destination in destinations:
                with self.subTest(destination=destination), self.assertRaisesRegex(
                    ValueError,
                    "outside the managed artifact root",
                ):
                    resolve_fresh_output(
                        "lite",
                        output_dir=destination,
                        overwrite=True,
                        artifact_root=root,
                    )
                self.assertFalse(destination.exists() and destination != root)

    def test_explicit_output_rejects_a_symlink_alias(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "managed"
            external = base / "external"
            external.mkdir()
            alias = base / "external-alias"
            alias.symlink_to(external, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "symlink alias"):
                resolve_fresh_output(
                    "lite",
                    output_dir=alias,
                    artifact_root=root,
                )

    def test_path_provenance_identifies_all_three_sources(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "managed"
            managed = reserve_run("lite", artifact_root=root)
            managed_checkpoint = _checkpoint(managed.path / "round_01/full.pt")

            legacy_root = root / "default"
            _saved_config(legacy_root, "classic")
            legacy_checkpoint = _checkpoint(legacy_root / "round_06/full.pt")

            external = Path(temporary).resolve() / "external"
            _saved_config(external, "deluxe")
            external_checkpoint = _checkpoint(external / "round_02/full.pt")

            self.assertEqual(
                find_run_for_path(managed_checkpoint, artifact_root=root),
                managed,
            )
            legacy = find_run_for_path(legacy_checkpoint, artifact_root=root)
            unmanaged = find_run_for_path(external_checkpoint, artifact_root=root)
            self.assertEqual(legacy.source, "legacy")
            self.assertEqual(legacy.mode, "classic")
            self.assertEqual(unmanaged.source, "unmanaged")
            self.assertEqual(unmanaged.mode, "deluxe")
            self.assertEqual(
                provenance_dict(managed),
                {
                    "source": "managed",
                    "mode": "lite",
                    "kind": "normal",
                    "run": 1,
                    "run_id": managed.run_id,
                    "state": "reserved",
                    "created_at": managed.created_at,
                    "path": str(managed.path),
                    "manifest_schema": 1,
                    "resume_provenance": None,
                },
            )


class ResumePlanningTests(unittest.TestCase):
    def test_managed_resume_is_in_place_until_a_different_selector_forks(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "managed"
            source = reserve_run("lite", artifact_root=root)
            checkpoint = _checkpoint(source.path / "round_01/full.pt")

            in_place = plan_resume_destination(checkpoint, artifact_root=root)
            same_run = plan_resume_destination(
                checkpoint,
                run=1,
                artifact_root=root,
            )
            forked = plan_resume_destination(
                checkpoint,
                run=2,
                artifact_root=root,
            )

            self.assertTrue(in_place.in_place)
            self.assertEqual(in_place.output_dir, source.path)
            self.assertTrue(same_run.in_place)
            self.assertTrue(forked.fork)
            self.assertEqual(forked.destination.run, 2)
            self.assertNotEqual(forked.output_dir, source.path)
            self.assertEqual(
                forked.provenance["checkpoint_sha256"],
                hashlib.sha256(b"checkpoint fixture").hexdigest(),
            )
            self.assertEqual(forked.destination.resume_provenance, forked.provenance)
            with self.assertRaisesRegex(ValueError, "explicit unmanaged"):
                plan_resume_destination(
                    checkpoint,
                    overwrite=True,
                    artifact_root=root,
                )

    def test_explicit_different_output_forks_only_to_unmanaged_paths(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "managed"
            source = reserve_run("lite", artifact_root=root)
            checkpoint = _checkpoint(source.path / "round_01/full.pt")
            external = Path(temporary).resolve() / "external"

            plan = plan_resume_destination(
                checkpoint,
                output_dir=external,
                artifact_root=root,
            )
            self.assertTrue(plan.fork)
            self.assertEqual(plan.destination.source, "unmanaged")
            self.assertEqual(plan.output_dir, external)
            provenance_path = external / RESUME_PROVENANCE_NAME
            self.assertTrue(provenance_path.is_file())
            provenance_document = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                provenance_document["schema_version"],
                RESUME_PROVENANCE_SCHEMA_VERSION,
            )
            self.assertEqual(provenance_document["resume"], plan.provenance)
            self.assertFalse(
                any(path.name.endswith(".tmp") for path in external.iterdir())
            )

            with self.assertRaisesRegex(ValueError, "unmanaged"):
                plan_resume_destination(
                    checkpoint,
                    output_dir=root / "lite/run_09",
                    artifact_root=root,
                )

    def test_explicit_unmanaged_fork_persists_provenance_without_source_writes(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "managed"
            source = reserve_run("lite", artifact_root=root)
            checkpoint = _checkpoint(source.path / "round_01/full.pt")
            (source.path / "config.json").write_bytes(b"source config bytes\n")
            before = _file_bytes(source.path)
            destination = base / "fork"

            plan = plan_resume_destination(
                checkpoint,
                output_dir=destination,
                artifact_root=root,
            )

            self.assertTrue(plan.fork)
            self.assertEqual(_file_bytes(source.path), before)
            document = json.loads(
                (destination / RESUME_PROVENANCE_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(document["resume"], plan.provenance)
            self.assertEqual(
                document["resume"]["checkpoint_sha256"],
                hashlib.sha256(before["round_01/full.pt"]).hexdigest(),
            )

    def test_unmarked_unmanaged_source_forks_but_current_v2_resumes_in_place(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "managed"
            unmarked = base / "unmarked"
            _saved_config(unmarked, "lite")
            old_checkpoint = _checkpoint(unmarked / "round_01/full.pt")
            before = _file_bytes(unmarked)

            old_plan = plan_resume_destination(
                old_checkpoint,
                artifact_root=root,
            )

            self.assertTrue(old_plan.fork)
            self.assertEqual(old_plan.source.source, "unmanaged")
            self.assertEqual(old_plan.output_dir, root / "lite/run_01")
            self.assertEqual(_file_bytes(unmarked), before)

            current = base / "current"
            _current_saved_config(current, "lite")
            current_checkpoint = _checkpoint(current / "round_01/full.pt")
            current_plan = plan_resume_destination(
                current_checkpoint,
                artifact_root=root,
            )

            self.assertTrue(current_plan.in_place)
            self.assertEqual(current_plan.output_dir, current)
            self.assertFalse((current / RESUME_PROVENANCE_NAME).exists())

    def test_completed_managed_run_forks_while_active_and_reserved_resume_in_place(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            reserved = reserve_run("lite", artifact_root=root)
            reserved_checkpoint = _checkpoint(
                reserved.path / "round_01/full.pt"
            )
            self.assertTrue(
                plan_resume_destination(
                    reserved_checkpoint,
                    artifact_root=root,
                ).in_place
            )

            active = update_run_manifest(reserved, state="active")
            self.assertTrue(
                plan_resume_destination(
                    reserved_checkpoint,
                    artifact_root=root,
                ).in_place
            )

            completed = update_run_manifest(active, state="completed")
            source_before = _file_bytes(completed.path)
            completed_plan = plan_resume_destination(
                reserved_checkpoint,
                artifact_root=root,
            )

            self.assertTrue(completed_plan.fork)
            self.assertEqual(completed_plan.destination.run, 2)
            self.assertEqual(completed_plan.destination.state, "reserved")
            self.assertEqual(_file_bytes(completed.path), source_before)

    def test_legacy_normal_and_smoke_checkpoints_always_fork(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            legacy_default = root / "default"
            _saved_config(legacy_default, "lite")
            default_checkpoint = _checkpoint(
                legacy_default / "round_06/full.pt"
            )

            default_plan = plan_resume_destination(
                default_checkpoint,
                artifact_root=root,
            )
            self.assertTrue(default_plan.source.legacy)
            self.assertTrue(default_plan.fork)
            self.assertEqual(default_plan.output_dir, root / "lite/run_01")
            self.assertEqual(
                default_plan.destination.resume_provenance,
                default_plan.provenance,
            )
            self.assertEqual(
                default_plan.provenance["checkpoint_path"],
                str(default_checkpoint.resolve()),
            )
            self.assertEqual(
                default_plan.provenance["checkpoint_sha256"],
                hashlib.sha256(b"checkpoint fixture").hexdigest(),
            )
            self.assertTrue(default_plan.provenance["legacy"])
            self.assertFalse((legacy_default / RUN_MANIFEST_NAME).exists())

            legacy_smoke = root / "smoke"
            _saved_config(legacy_smoke, "lite")
            smoke_checkpoint = _checkpoint(legacy_smoke / "round_01/full.pt")
            smoke_plan = plan_resume_destination(
                smoke_checkpoint,
                artifact_root=root,
            )
            self.assertTrue(smoke_plan.source.legacy)
            self.assertTrue(smoke_plan.destination.smoke)
            self.assertEqual(smoke_plan.output_dir, root / "smoke/run_01")
            self.assertEqual(
                smoke_plan.destination.resume_provenance,
                smoke_plan.provenance,
            )
            self.assertFalse((legacy_smoke / RUN_MANIFEST_NAME).exists())

    def test_overwrite_never_applies_to_an_automatic_resume_destination(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            legacy = root / "default"
            _saved_config(legacy, "lite")
            checkpoint = _checkpoint(legacy / "round_01/full.pt")

            with self.assertRaisesRegex(ValueError, "explicit unmanaged"):
                plan_resume_destination(
                    checkpoint,
                    overwrite=True,
                    artifact_root=root,
                )
            self.assertFalse((root / "lite").exists())

    def test_temporary_or_missing_checkpoints_are_never_planned(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            temporary_checkpoint = _checkpoint(
                root / "lite/run_01/round_01/full.pt.tmp"
            )
            with self.assertRaisesRegex(ValueError, "temporary"):
                plan_resume_destination(
                    temporary_checkpoint,
                    mode="lite",
                    artifact_root=root,
                )
            with self.assertRaises(FileNotFoundError):
                plan_resume_destination(
                    root / "missing.pt",
                    mode="lite",
                    artifact_root=root,
                )

    def test_nested_checkpoint_in_a_malformed_managed_run_is_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            malformed = root / "lite/run_01"
            checkpoint = _checkpoint(malformed / "nested/full.pt")

            with self.assertRaisesRegex(ValueError, "malformed managed"):
                plan_resume_destination(
                    checkpoint,
                    mode="lite",
                    artifact_root=root,
                )


if __name__ == "__main__":
    unittest.main()
