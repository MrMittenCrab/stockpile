"""Persistent complexity-memory and rigorous-bound contract tests."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import stockpile


def _information_result(
    configuration,
    *,
    exact: bool = False,
    information_sets: int = 12,
    information_set_actions: int = 34,
    max_states: int = 100,
    max_seconds: float = 2.0,
) -> stockpile.InformationSetComplexity:
    configured = getattr(configuration, "configured_game", configuration)
    players = configured.parameters.player_count
    per_player_sets = {player: 0 for player in range(players)}
    per_player_actions = {player: 0 for player in range(players)}
    per_player_sets[0] = information_sets
    per_player_actions[0] = information_set_actions
    return stockpile.InformationSetComplexity(
        parameters=configured.parameters,
        exact=exact,
        count_kind="exact" if exact else "lower_bound",
        information_sets=information_sets,
        information_set_actions=information_set_actions,
        max_actions_per_information_set=min(4, information_set_actions),
        per_player_information_sets=per_player_sets,
        per_player_information_set_actions=per_player_actions,
        states_visited=max_states - int(exact),
        terminal_states=3,
        chance_nodes=5,
        elapsed_seconds=0.125,
        max_states=max_states,
        max_seconds=max_seconds,
        truncation_reason=None if exact else "max_states",
    )


def _interface_result(
    configuration: stockpile.InterfaceConfiguration,
    **result_values,
) -> stockpile.InterfaceComplexity:
    report = stockpile.complexity_report(configuration.configured_game)
    return stockpile.InterfaceComplexity(
        configuration=configuration,
        action_catalog=stockpile.ActionCatalogComplexity(
            num_distinct_actions=int(report["num_distinct_actions"]),
            max_legal_actions=int(report["max_legal_actions"]),
            max_chance_outcomes=int(report["max_chance_outcomes"]),
            shared_action_head=int(report["shared_action_head"]),
            max_game_length=int(report["max_game_length"]),
            observation_size=int(report["observation_size"]),
        ),
        information_set_complexity=_information_result(
            configuration,
            **result_values,
        ),
    )


def _configuration(**changes) -> stockpile.InterfaceConfiguration:
    values = {
        "profile": "lite",
        "player_count": 2,
        "round_count": 2,
        "lite_options": ("dividends",),
        "action_space_mode": "compact",
    }
    values.update(changes)
    return stockpile.create_configuration(**values)


def _rewrite_as_legacy(
    current_path: Path,
    legacy_path: Path,
    *,
    schema_version: int,
    omit_saved_round: bool = False,
) -> None:
    """Convert a test v3 document into a valid historical fixture."""

    if schema_version not in {1, 2}:
        raise ValueError("legacy test schema must be 1 or 2")

    document = json.loads(current_path.read_text(encoding="utf-8"))
    old_digest = "1" * 64
    document["schema_version"] = schema_version
    document["platform_digest"] = old_digest
    legacy_entries = {}
    for entry in document["entries"].values():
        entry["provenance"]["schema_version"] = schema_version
        entry["provenance"]["platform_digest"] = old_digest
        if schema_version == 1:
            entry["semantic_rules"].pop("starting_shares_per_player", None)
        entry["semantic_rules"].pop("sequential_observable_selling", None)
        encoded = json.dumps(
            entry["semantic_rules"],
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        entry["fingerprint"] = hashlib.sha256(encoded).hexdigest()
        if omit_saved_round:
            entry["result"]["parameters"].pop("round_count", None)
        legacy_entries[entry["fingerprint"]] = entry
    document["entries"] = legacy_entries
    legacy_path.write_text(json.dumps(document), encoding="utf-8")


class TemporaryCacheCase(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self._temporary_directory.name)
        self.seed_path = root / "preset_v3.json"
        self.learned_path = root / "learned_v3.json"
        self.legacy_v1_seed_path = root / "preset_v1.json"
        self.legacy_v2_seed_path = root / "preset_v2.json"
        self.cache = stockpile.ComplexityCache(
            seed_path=self.seed_path,
            learned_path=self.learned_path,
        )

    def tearDown(self):
        self._temporary_directory.cleanup()


class SemanticFingerprintTests(TemporaryCacheCase):
    def test_schema_three_reads_both_prior_schemas(self):
        self.assertEqual(stockpile.CACHE_SCHEMA_VERSION, 3)
        self.assertEqual(stockpile.LEGACY_CACHE_SCHEMA_VERSIONS, (1, 2))

    def test_twelve_trees_serve_twenty_four_compact_and_shared_views(self):
        from stockpile.tools.generate_preset_complexity import PRESET_MATRIX

        self.assertEqual(
            tuple(PRESET_MATRIX),
            tuple(
                (profile, players)
                for profile in ("lite", "classic", "deluxe")
                for players in range(2, 6)
            ),
        )

        fingerprints = set()
        for index, (profile, players) in enumerate(
            PRESET_MATRIX,
            start=1,
        ):
            arguments = {
                "profile": profile,
                "player_count": players,
                "round_count": 6,
            }
            compact = stockpile.create_configuration(
                **arguments,
                action_space_mode="compact",
            )
            shared = stockpile.create_configuration(
                **arguments,
                action_space_mode="shared",
            )
            self.assertEqual(
                compact.rule_set.starting_shares_per_player,
                0 if profile == "lite" else 1,
            )
            self.assertEqual(compact.rule_set.investors, profile == "deluxe")
            self.assertEqual(
                compact.rule_set.sequential_observable_selling,
                profile != "lite",
            )
            compact_key = stockpile.semantic_fingerprint(compact.configured_game)
            shared_key = stockpile.semantic_fingerprint(shared.configured_game)
            self.assertEqual(compact_key, shared_key)
            fingerprints.add(compact_key)

            self.cache.save(
                compact.configured_game,
                _information_result(
                    compact,
                    information_sets=10 * index,
                    information_set_actions=30 * index,
                ),
            )
            compact_hit = self.cache.lookup(compact.configured_game)
            shared_hit = self.cache.lookup(shared.configured_game)
            self.assertIsNotNone(compact_hit)
            self.assertIsNotNone(shared_hit)
            assert compact_hit is not None and shared_hit is not None
            self.assertEqual(
                compact_hit.result.information_sets,
                shared_hit.result.information_sets,
            )
            self.assertEqual(
                compact_hit.result.parameters.action_space_mode,
                "compact",
            )
            self.assertEqual(
                shared_hit.result.parameters.action_space_mode,
                "shared",
            )

        self.assertEqual(len(fingerprints), 12)

    def test_identical_explicit_configuration_reuses_one_entry(self):
        first = _configuration()
        second = _configuration()
        self.assertIsNot(first.configured_game, second.configured_game)
        self.assertEqual(
            stockpile.semantic_fingerprint(first.configured_game),
            stockpile.semantic_fingerprint(second.configured_game),
        )

        self.cache.save(first.configured_game, _information_result(first))
        hit = self.cache.lookup(second.configured_game)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.result.information_sets, 12)

    def test_round_and_lite_option_changes_are_safe_cache_misses(self):
        original = _configuration()
        self.cache.save(original.configured_game, _information_result(original))
        changes = (
            _configuration(round_count=3),
            _configuration(lite_options=("trading_fees",)),
            _configuration(lite_options=("starting_share",)),
        )
        for changed in changes:
            with self.subTest(configuration=changed.parameters):
                self.assertIsNone(self.cache.lookup(changed.configured_game))

    def test_sell_order_is_automatically_fingerprinted(self):
        sequential = stockpile.resolve_configuration(
            "lite",
            player_count=2,
            round_count=2,
            dividend=True,
            sell_order=True,
        )
        nonsequential = stockpile.resolve_configuration(
            "lite",
            player_count=2,
            round_count=2,
            dividend=True,
            sell_order=False,
        )

        sequential_payload = stockpile.semantic_rule_payload(
            sequential.configured_game
        )
        nonsequential_payload = stockpile.semantic_rule_payload(
            nonsequential.configured_game
        )
        self.assertIs(
            sequential_payload["sequential_observable_selling"],
            True,
        )
        self.assertIs(
            nonsequential_payload["sequential_observable_selling"],
            False,
        )
        self.assertNotEqual(
            stockpile.semantic_fingerprint(sequential.configured_game),
            stockpile.semantic_fingerprint(nonsequential.configured_game),
        )

        self.cache.save(
            sequential.configured_game,
            _information_result(sequential),
        )
        self.assertIsNone(self.cache.lookup(nonsequential.configured_game))


class CachePersistenceTests(TemporaryCacheCase):
    def test_corrupt_json_is_ignored_and_replaced_by_a_valid_save(self):
        configuration = _configuration()
        self.learned_path.write_text("{ definitely not JSON", encoding="utf-8")

        self.assertIsNone(self.cache.lookup(configuration.configured_game))
        self.cache.save(
            configuration.configured_game,
            _information_result(configuration),
        )
        self.assertIsNotNone(self.cache.lookup(configuration.configured_game))
        document = json.loads(self.learned_path.read_text(encoding="utf-8"))
        self.assertEqual(document["schema_version"], 3)

    def test_semantics_are_authoritative_and_digest_is_auditable_provenance(self):
        configuration = _configuration()
        self.cache.save(
            configuration.configured_game,
            _information_result(configuration),
        )
        document = json.loads(self.learned_path.read_text(encoding="utf-8"))
        document["semantics_version"] = -1
        self.learned_path.write_text(json.dumps(document), encoding="utf-8")
        self.assertIsNone(self.cache.lookup(configuration.configured_game))

        self.cache.save(
            configuration.configured_game,
            _information_result(configuration),
        )
        document = json.loads(self.learned_path.read_text(encoding="utf-8"))
        historical_digest = "0" * 64
        document["platform_digest"] = historical_digest
        for entry in document["entries"].values():
            entry["provenance"]["platform_digest"] = historical_digest
        self.learned_path.write_text(json.dumps(document), encoding="utf-8")
        hit = self.cache.lookup(configuration.configured_game)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.provenance.platform_digest, historical_digest)

    def test_save_is_atomic_and_stronger_records_win(self):
        configuration = _configuration()
        with patch(
            "stockpile.complexity_cache.os.replace",
            wraps=os.replace,
        ) as replace_call:
            self.cache.save(
                configuration.configured_game,
                _information_result(configuration, information_sets=20),
            )

        replace_call.assert_called()
        self.assertTrue(self.learned_path.is_file())
        self.assertFalse(tuple(self.learned_path.parent.glob("*.tmp")))

        self.cache.save(
            configuration.configured_game,
            _information_result(
                configuration,
                exact=True,
                information_sets=3,
                information_set_actions=7,
            ),
        )
        self.cache.save(
            configuration.configured_game,
            _information_result(configuration, information_sets=999),
        )
        hit = self.cache.lookup(configuration.configured_game)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertTrue(hit.result.exact)
        self.assertEqual(hit.result.information_sets, 3)

    def test_current_cache_identity_survives_public_parameter_default_drift(self):
        configuration = stockpile.create_configuration(
            "classic",
            player_count=3,
            round_count=7,
        )
        self.cache.save(
            configuration.configured_game,
            _information_result(configuration, information_sets=71),
        )
        document = json.loads(self.learned_path.read_text(encoding="utf-8"))
        entry = next(iter(document["entries"].values()))
        entry["result"]["parameters"].pop("round_count", None)
        self.learned_path.write_text(json.dumps(document), encoding="utf-8")

        hit = self.cache.lookup(configuration.configured_game)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.result.information_sets, 71)
        self.assertEqual(hit.result.parameters.round_count, 7)

    def test_new_writes_can_coexist_with_historical_entry_provenance(self):
        historical = _configuration(round_count=2)
        current = _configuration(round_count=3)
        self.cache.save(
            historical.configured_game,
            _information_result(historical, information_sets=22),
        )
        document = json.loads(self.learned_path.read_text(encoding="utf-8"))
        old_digest = "2" * 64
        document["platform_digest"] = old_digest
        for entry in document["entries"].values():
            entry["provenance"]["platform_digest"] = old_digest
        self.learned_path.write_text(json.dumps(document), encoding="utf-8")

        self.cache.save(
            current.configured_game,
            _information_result(current, information_sets=33),
        )

        old_hit = self.cache.lookup(historical.configured_game)
        new_hit = self.cache.lookup(current.configured_game)
        self.assertIsNotNone(old_hit)
        self.assertIsNotNone(new_hit)
        assert old_hit is not None and new_hit is not None
        self.assertEqual(old_hit.provenance.platform_digest, old_digest)
        self.assertNotEqual(new_hit.provenance.platform_digest, old_digest)


class LegacyMigrationTests(TemporaryCacheCase):
    def _legacy_cache(self, *paths: Path) -> stockpile.ComplexityCache:
        return stockpile.ComplexityCache(
            seed_path=self.seed_path.parent / "empty_v3.json",
            learned_path=None,
            legacy_seed_paths=paths,
            legacy_learned_paths=(),
        )

    def test_v1_exact_semantic_match_reuses_old_digest_read_only(self):
        configuration = stockpile.create_configuration(
            "classic",
            player_count=3,
            round_count=7,
        )
        self.cache.save(
            configuration.configured_game,
            _information_result(configuration, information_sets=73),
            source="preset",
        )
        _rewrite_as_legacy(
            self.seed_path,
            self.legacy_v1_seed_path,
            schema_version=1,
            omit_saved_round=True,
        )
        original = self.legacy_v1_seed_path.read_bytes()

        hit = self._legacy_cache(self.legacy_v1_seed_path).lookup(
            configuration.configured_game
        )

        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.provenance.schema_version, 1)
        self.assertEqual(hit.provenance.platform_digest, "1" * 64)
        self.assertEqual(hit.result.information_sets, 73)
        self.assertEqual(hit.result.parameters.round_count, 7)
        self.assertEqual(self.legacy_v1_seed_path.read_bytes(), original)

    def test_v2_exact_match_reuses_mixed_historical_digests_read_only(self):
        configuration = stockpile.create_configuration(
            "classic",
            player_count=4,
            round_count=5,
        )
        self.cache.save(
            configuration.configured_game,
            _information_result(configuration, information_sets=52),
            source="preset",
        )
        _rewrite_as_legacy(
            self.seed_path,
            self.legacy_v2_seed_path,
            schema_version=2,
        )
        document = json.loads(
            self.legacy_v2_seed_path.read_text(encoding="utf-8")
        )
        for entry in document["entries"].values():
            entry["provenance"]["platform_digest"] = "2" * 64
        self.legacy_v2_seed_path.write_text(
            json.dumps(document),
            encoding="utf-8",
        )
        original = self.legacy_v2_seed_path.read_bytes()

        hit = self._legacy_cache(self.legacy_v2_seed_path).lookup(
            configuration.configured_game
        )

        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.provenance.schema_version, 2)
        self.assertEqual(hit.provenance.platform_digest, "2" * 64)
        self.assertEqual(hit.result.information_sets, 52)
        self.assertEqual(self.legacy_v2_seed_path.read_bytes(), original)

    def test_legacy_neighbouring_rules_and_tampering_are_misses(self):
        configuration = stockpile.create_configuration(
            "classic",
            player_count=3,
            round_count=7,
        )
        self.cache.save(
            configuration.configured_game,
            _information_result(configuration),
            source="preset",
        )
        _rewrite_as_legacy(
            self.seed_path,
            self.legacy_v2_seed_path,
            schema_version=2,
        )
        legacy_cache = self._legacy_cache(self.legacy_v2_seed_path)
        self.assertIsNone(
            legacy_cache.lookup(
                stockpile.create_configuration(
                    "classic",
                    player_count=3,
                    round_count=6,
                ).configured_game
            )
        )

        document = json.loads(
            self.legacy_v2_seed_path.read_text(encoding="utf-8")
        )
        entry = next(iter(document["entries"].values()))
        entry["semantic_rules"]["round_count"] = 6
        self.legacy_v2_seed_path.write_text(
            json.dumps(document),
            encoding="utf-8",
        )
        self.assertIsNone(legacy_cache.lookup(configuration.configured_game))

    def test_v1_missing_share_field_means_one_and_cannot_match_zero_share_lite(self):
        configuration = stockpile.create_configuration("lite", round_count=2)
        self.cache.save(
            configuration.configured_game,
            _information_result(configuration),
            source="preset",
        )
        _rewrite_as_legacy(
            self.seed_path,
            self.legacy_v1_seed_path,
            schema_version=1,
        )

        self.assertEqual(configuration.rule_set.starting_shares_per_player, 0)
        self.assertIsNone(
            self._legacy_cache(self.legacy_v1_seed_path).lookup(
                configuration.configured_game
            )
        )

    def test_v1_and_v2_missing_sell_order_mean_sequential_and_miss_false(self):
        configuration = stockpile.resolve_configuration(
            "classic",
            player_count=2,
            round_count=2,
            sell_order=False,
        )
        self.cache.save(
            configuration.configured_game,
            _information_result(configuration),
            source="preset",
        )

        for schema_version, path in (
            (1, self.legacy_v1_seed_path),
            (2, self.legacy_v2_seed_path),
        ):
            with self.subTest(schema_version=schema_version):
                _rewrite_as_legacy(
                    self.seed_path,
                    path,
                    schema_version=schema_version,
                )
                self.assertIsNone(
                    self._legacy_cache(path).lookup(
                        configuration.configured_game
                    )
                )


class CacheAwareResolverTests(TemporaryCacheCase):
    def test_prefer_hit_avoids_live_traversal(self):
        configuration = _configuration()
        self.cache.save(
            configuration.configured_game,
            _information_result(configuration, information_sets=73),
        )
        with patch(
            "stockpile.stockpile_interface.compute_interface_complexity",
            side_effect=AssertionError("cache hit performed a traversal"),
        ) as live:
            resolved = stockpile.resolve_interface_complexity(
                configuration,
                cache=self.cache,
                cache_policy="prefer",
            )
        live.assert_not_called()
        self.assertEqual(resolved.information_set_complexity.information_sets, 73)
        self.assertEqual(resolved.provenance.source, "learned")

    def test_prefer_miss_runs_once_then_remembers_identical_configuration(self):
        first = _configuration()
        equivalent = _configuration()
        live_result = _interface_result(first, information_sets=44)
        with patch(
            "stockpile.stockpile_interface.compute_interface_complexity",
            return_value=live_result,
        ) as live:
            first_resolved = stockpile.resolve_interface_complexity(
                first,
                cache=self.cache,
                cache_policy="prefer",
                max_states=1_000,
                max_seconds=3.0,
            )
        live.assert_called_once_with(
            first,
            max_states=1_000,
            max_seconds=3.0,
            require_exact=False,
        )
        self.assertEqual(first_resolved.provenance.source, "live")

        with patch(
            "stockpile.stockpile_interface.compute_interface_complexity",
            side_effect=AssertionError("identical configuration should be remembered"),
        ) as second_live:
            remembered = stockpile.resolve_interface_complexity(
                equivalent,
                cache=self.cache,
                cache_policy="prefer",
            )
        second_live.assert_not_called()
        self.assertEqual(remembered.provenance.source, "learned")
        self.assertEqual(remembered.information_set_complexity.information_sets, 44)

    def test_refresh_recalculates_while_off_never_reads_or_writes(self):
        configuration = _configuration()
        self.cache.save(
            configuration.configured_game,
            _information_result(configuration, information_sets=10),
        )
        with patch(
            "stockpile.stockpile_interface.compute_interface_complexity",
            return_value=_interface_result(configuration, information_sets=20),
        ):
            refreshed = stockpile.resolve_interface_complexity(
                configuration,
                cache=self.cache,
                cache_policy="refresh",
            )
        self.assertEqual(refreshed.information_set_complexity.information_sets, 20)

        with patch(
            "stockpile.stockpile_interface.compute_interface_complexity",
            return_value=_interface_result(configuration, information_sets=30),
        ):
            uncached = stockpile.resolve_interface_complexity(
                configuration,
                cache=self.cache,
                cache_policy="off",
            )
        self.assertEqual(uncached.information_set_complexity.information_sets, 30)
        hit = self.cache.lookup(configuration.configured_game)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.result.information_sets, 20)


class StructuralBoundsTests(unittest.TestCase):
    def test_runtime_branching_invariants_guard_bound_inputs(self):
        configuration = stockpile.create_configuration("lite", round_count=1)
        chance_state = configuration.game.new_initial_state()
        chance_state.rule_set = replace(chance_state.rule_set, max_chance_outcomes=1)
        with self.assertRaisesRegex(RuntimeError, "max_chance_outcomes"):
            chance_state.chance_outcomes()

        player_state = configuration.game.new_initial_state()
        while player_state.is_chance_node():
            player_state.apply_action(player_state.chance_outcomes()[0][0])
        player_state.rule_set = replace(player_state.rule_set, max_legal_actions=1)
        with self.assertRaisesRegex(RuntimeError, "max_legal_actions"):
            player_state.legal_actions()

    def test_formula_includes_fixed_deluxe_investor_setup_chance_nodes(self):
        for profile in ("classic", "deluxe"):
            with self.subTest(profile=profile):
                configuration = stockpile.create_configuration(
                    profile,
                    player_count=3,
                    round_count=2,
                )
                rules = configuration.rule_set
                self.assertEqual(rules.investors, profile == "deluxe")
                investor_deals = (
                    rules.player_count * 2 if rules.investors else 0
                )
                chance_nodes = (
                    rules.player_count * rules.starting_shares_per_player
                    + 1
                    + investor_deals
                    + rules.round_count
                    * (
                        2 * rules.company_count
                        + rules.stockpile_count
                        + 2 * rules.player_count * rules.supply_batches
                    )
                )
                expected_log10 = (
                    math.log10(rules.max_game_length)
                    + rules.max_game_length * math.log10(rules.max_legal_actions)
                    + chance_nodes * math.log10(rules.max_chance_outcomes)
                )
                bounds = stockpile.compute_complexity_bounds(
                    configuration.configured_game
                )
                self.assertEqual(bounds.chance_node_bound, chance_nodes)
                self.assertAlmostEqual(
                    bounds.upper_information_sets_log10,
                    expected_log10,
                )

    def test_lite_starting_share_option_adds_one_setup_draw_per_player(self):
        bounds = []
        for options in ((), ("starting_share",)):
            configuration = stockpile.create_configuration(
                "lite",
                player_count=4,
                round_count=1,
                lite_options=options,
            )
            bounds.append(
                stockpile.compute_complexity_bounds(
                    configuration.configured_game
                )
            )
        self.assertEqual(
            bounds[1].chance_node_bound - bounds[0].chance_node_bound,
            4,
        )

    def test_exact_result_collapses_both_bounds(self):
        configuration = _configuration()
        exact = _information_result(
            configuration,
            exact=True,
            information_sets=3,
            information_set_actions=7,
        )
        bounds = stockpile.compute_complexity_bounds(
            configuration.configured_game,
            result=exact,
        )
        self.assertTrue(bounds.exact)
        self.assertEqual(bounds.lower_information_sets, 3)
        self.assertEqual(bounds.upper_information_sets, 3)
        self.assertEqual(bounds.lower_information_set_actions, 7)
        self.assertEqual(bounds.upper_information_set_actions, 7)


class PresetGeneratorTests(TemporaryCacheCase):
    def test_generator_is_resumable_and_uses_the_twelve_tree_matrix(self):
        from stockpile.tools import generate_preset_complexity as generator

        def calculate(configured, *, max_states, max_seconds, require_exact=False):
            self.assertFalse(require_exact)
            self.assertEqual(configured.parameters.round_count, 6)
            return _information_result(
                configured,
                information_sets=(
                    configured.parameters.player_count * 10
                    + int(configured.rule_set.investors)
                ),
                max_states=max_states,
                max_seconds=max_seconds,
            )

        with patch.object(
            generator.platform,
            "compute_information_set_complexity",
            side_effect=calculate,
        ) as compute:
            first = generator.generate_preset_complexity(
                output_path=self.seed_path,
                max_states=10_000,
                max_seconds=120.0,
            )
        self.assertEqual(compute.call_count, 12)
        self.assertEqual(len(first), 12)
        first_bytes = self.seed_path.read_bytes()

        with patch.object(
            generator.platform,
            "compute_information_set_complexity",
            side_effect=AssertionError("complete presets must resume from memory"),
        ) as compute:
            second = generator.generate_preset_complexity(
                output_path=self.seed_path,
                max_states=10_000,
                max_seconds=120.0,
            )
        compute.assert_not_called()
        self.assertEqual(first, second)
        self.assertEqual(self.seed_path.read_bytes(), first_bytes)

    def test_generator_promotes_only_exact_matching_legacy_entries(self):
        from stockpile.tools import generate_preset_complexity as generator

        legacy_configuration = stockpile.create_configuration(
            "classic",
            player_count=2,
            round_count=6,
        )
        self.cache.save(
            legacy_configuration.configured_game,
            _information_result(
                legacy_configuration,
                information_sets=777,
                information_set_actions=888,
                max_states=10_000,
                max_seconds=120.0,
            ),
            source="preset",
        )
        _rewrite_as_legacy(
            self.seed_path,
            self.legacy_v2_seed_path,
            schema_version=2,
        )
        output = self.seed_path.parent / "generated_v3.json"

        def calculate(configured, *, max_states, max_seconds, require_exact=False):
            return _information_result(
                configured,
                max_states=max_states,
                max_seconds=max_seconds,
            )

        with patch.object(
            generator.platform,
            "compute_information_set_complexity",
            side_effect=calculate,
        ) as compute:
            generated = generator.generate_preset_complexity(
                output_path=output,
                max_states=10_000,
                max_seconds=120.0,
                legacy_paths=(self.legacy_v2_seed_path,),
            )

        self.assertEqual(compute.call_count, 11)
        self.assertEqual(generated["classic:2"].result.information_sets, 777)
        self.assertEqual(len(generated), 12)


if __name__ == "__main__":
    unittest.main()
