"""Contract tests for the resolved Stockpile interface facade."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, replace
import unittest
from unittest.mock import patch

import stockpile


def _complexity_result(configuration, *, exact=False):
    players = configuration.player_count
    sets = {player: 0 for player in range(players)}
    actions = {player: 0 for player in range(players)}
    sets[0] = 3
    actions[0] = 7
    return stockpile.InformationSetComplexity(
        parameters=configuration.parameters,
        exact=exact,
        count_kind="exact" if exact else "lower_bound",
        information_sets=3,
        information_set_actions=7,
        max_actions_per_information_set=4,
        per_player_information_sets=sets,
        per_player_information_set_actions=actions,
        states_visited=9,
        terminal_states=1,
        chance_nodes=2,
        elapsed_seconds=0.01,
        max_states=10,
        max_seconds=1.0,
        truncation_reason=None if exact else "max_states",
    )


class ResolvedConfigurationTests(unittest.TestCase):
    def test_public_type_is_frozen_and_old_type_name_is_an_exact_alias(self):
        self.assertIs(stockpile.InterfaceConfiguration, stockpile.GameConfig)
        configuration = stockpile.resolve_configuration("lite")
        with self.assertRaises(FrozenInstanceError):
            configuration.hand = True

    def test_profile_aliases_are_canonicalized(self):
        self.assertEqual(
            tuple(mode.value for mode in stockpile.ConfigurationMode),
            ("lite", "classic", "deluxe"),
        )
        aliases = {
            "minimal_training": "lite",
            "core": "classic",
            "full": "deluxe",
            "expanded": "deluxe",
            "expanded_variants": "deluxe",
        }
        for requested, canonical in aliases.items():
            with self.subTest(requested=requested):
                configuration = stockpile.resolve_configuration(requested)
                self.assertEqual(configuration.mode.value, canonical)
                self.assertEqual(configuration.parameters.rules_profile, canonical)
                self.assertEqual(configuration.rule_set.profile, canonical)

    def test_none_uses_the_strict_mode_default_table(self):
        expected = {
            "lite": {
                "hand": False,
                "fees": False,
                "dividend": False,
                "split": False,
                "majority": False,
                "stock_tracks": False,
                "sell_order": False,
                "impact": False,
                "investor": False,
            },
            "classic": {
                "hand": True,
                "fees": True,
                "dividend": True,
                "split": True,
                "majority": True,
                "stock_tracks": False,
                "sell_order": True,
                "impact": True,
                "investor": False,
            },
            "deluxe": {
                "hand": True,
                "fees": True,
                "dividend": True,
                "split": True,
                "majority": True,
                "stock_tracks": False,
                "sell_order": True,
                "impact": True,
                "investor": True,
            },
        }
        for mode, values in expected.items():
            with self.subTest(mode=mode):
                configuration = stockpile.resolve_configuration(mode)
                for name, value in values.items():
                    self.assertIs(getattr(configuration, name), value, name)
                self.assertEqual(configuration.player_count, 2)
                self.assertEqual(configuration.round_count, 6)
                self.assertEqual(configuration.action_space_mode, "compact")
                self.assertIs(configuration.parameters, configuration.configured_game.parameters)
                self.assertIs(configuration.rule_set, configuration.configured_game.rule_set)
                self.assertIs(configuration.game, configuration.configured_game.game)
                if mode == "deluxe":
                    self.assertEqual(len(configuration.rule_set.enabled_investors), 10)

    def test_explicit_switches_override_defaults_without_changing_fixed_layers(self):
        cases = (
            ("lite", True, False, False),
            ("classic", False, True, False),
            ("deluxe", False, True, True),
        )
        for mode, value, impact, investor in cases:
            with self.subTest(mode=mode):
                configuration = stockpile.resolve_configuration(
                    mode,
                    hand=value,
                    fees=value,
                    dividend=value,
                    split=value,
                    majority=value,
                    stock_tracks=value,
                    sell_order=value,
                )
                for name in (
                    "hand",
                    "fees",
                    "dividend",
                    "split",
                    "majority",
                    "stock_tracks",
                    "sell_order",
                ):
                    self.assertIs(getattr(configuration, name), value, name)
                self.assertIs(configuration.impact, impact)
                self.assertIs(configuration.investor, investor)

    def test_friendly_switches_map_to_exact_grouped_engine_rules(self):
        configuration = stockpile.resolve_configuration(
            "classic",
            player_count=5,
            round_count=10,
            hand=False,
            fees=False,
            dividend=True,
            split=False,
            majority=True,
            stock_tracks=True,
            sell_order=False,
            action_space_mode="shared",
        )
        rules = configuration.rule_set
        self.assertEqual(rules.starting_shares_per_player, 0)
        self.assertFalse(rules.trading_fees)
        self.assertTrue(rules.forecast_dividends)
        self.assertTrue(rules.dividend_reveal_choice)
        self.assertFalse(rules.stock_splits)
        self.assertFalse(rules.repeat_split_bonus)
        self.assertTrue(rules.majority_bonus)
        self.assertTrue(rules.advanced_price_tracks)
        self.assertTrue(rules.advanced_track_dividends)
        self.assertFalse(rules.sequential_observable_selling)
        self.assertEqual(configuration.player_count, 5)
        self.assertEqual(configuration.round_count, 10)
        self.assertEqual(configuration.action_space_mode, "shared")
        self.assertEqual(
            configuration.parameters.rule_overrides,
            {
                "hand": False,
                "fees": False,
                "dividend": True,
                "split": False,
                "majority": True,
                "stock_tracks": True,
                "sell_order": False,
            },
        )

    def test_track_dividends_require_both_friendly_layers(self):
        no_dividend = stockpile.resolve_configuration(
            "lite",
            dividend=False,
            stock_tracks=True,
        )
        no_tracks = stockpile.resolve_configuration(
            "classic",
            dividend=True,
            stock_tracks=False,
        )
        self.assertFalse(no_dividend.rule_set.advanced_track_dividends)
        self.assertFalse(no_tracks.rule_set.advanced_track_dividends)

    def test_platform_configuration_is_built_once(self):
        from stockpile import stockpile_interface as interface

        with patch.object(
            interface.platform,
            "configure_game",
            wraps=interface.platform.configure_game,
        ) as configure:
            configuration = interface.resolve_configuration("classic")
        configure.assert_called_once()
        self.assertIsInstance(configuration, stockpile.GameConfig)

    def test_invariants_reject_a_public_view_that_disagrees_with_engine(self):
        configuration = stockpile.resolve_configuration("classic")
        with self.assertRaisesRegex(ValueError, "trading_fees"):
            replace(configuration, fees=False)
        with self.assertRaisesRegex(ValueError, "Market Impact"):
            replace(configuration, mode=stockpile.ConfigurationMode.LITE)

    def test_invalid_values_are_rejected(self):
        for name in (
            "hand",
            "fees",
            "dividend",
            "split",
            "majority",
            "stock_tracks",
            "sell_order",
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(TypeError, name):
                    stockpile.resolve_configuration("lite", **{name: "off"})
        for rounds in (0, 11):
            with self.subTest(rounds=rounds):
                with self.assertRaises(ValueError):
                    stockpile.resolve_configuration("lite", round_count=rounds)
        for players in (1, 6):
            with self.subTest(players=players):
                with self.assertRaises(ValueError):
                    stockpile.resolve_configuration("classic", player_count=players)
        with self.assertRaisesRegex(ValueError, "rules profile"):
            stockpile.resolve_configuration("flex")
        with self.assertRaises(ValueError):
            stockpile.resolve_configuration("lite", action_space_mode="wide")


class CompatibilityTests(unittest.TestCase):
    def test_legacy_lite_options_translate_to_resolved_switches(self):
        configuration = stockpile.create_configuration(
            "lite",
            player_count=3,
            round_count=4,
            lite_options=(
                "majority_bonus",
                "dividends",
                "starting_share",
            ),
            action_space_mode="shared",
        )
        self.assertTrue(configuration.hand)
        self.assertFalse(configuration.fees)
        self.assertTrue(configuration.dividend)
        self.assertFalse(configuration.split)
        self.assertTrue(configuration.majority)
        self.assertFalse(configuration.stock_tracks)
        self.assertFalse(configuration.sell_order)
        self.assertEqual(
            tuple(option.value for option in configuration.lite_options),
            ("starting_share", "dividends", "majority_bonus"),
        )

    def test_legacy_deluxe_flag_cannot_disable_fixed_investors(self):
        disabled_request = stockpile.create_configuration(
            "deluxe",
            deluxe_investors=False,
        )
        enabled_request = stockpile.create_configuration(
            "deluxe",
            deluxe_investors=True,
        )
        self.assertTrue(disabled_request.investor)
        self.assertTrue(enabled_request.investor)
        self.assertTrue(disabled_request.deluxe_investors)
        with self.assertRaisesRegex(ValueError, "fixed on only"):
            stockpile.create_configuration(
                "classic",
                deluxe_investors=True,
            )

    def test_legacy_options_remain_lite_only(self):
        with self.assertRaisesRegex(ValueError, "lite_options"):
            stockpile.create_configuration(
                "classic",
                lite_options=("dividends",),
            )
        with self.assertRaisesRegex(ValueError, "optional rule"):
            stockpile.create_configuration(
                "lite",
                lite_options=("market_impact",),
            )


class ExplanationTests(unittest.TestCase):
    def test_explanation_uses_resolved_rules(self):
        lite = stockpile.resolve_configuration(
            "lite",
            hand=False,
        )
        explanation = stockpile.explain_configuration(lite)
        self.assertIn("without starting shares", " ".join(explanation.setup))
        self.assertNotIn("Action phase", " ".join(explanation.turns))
        self.assertIn("Without observing the others", " ".join(explanation.turns))

        sequential_lite = stockpile.resolve_configuration(
            "lite",
            sell_order=True,
        )
        sequential_explanation = stockpile.explain_configuration(sequential_lite)
        self.assertIn(
            "sale proceeds and public cash update immediately",
            " ".join(sequential_explanation.turns),
        )

        deluxe = stockpile.resolve_configuration("deluxe")
        explanation = stockpile.explain_configuration(deluxe)
        self.assertIn("Investors", " ".join(explanation.setup))
        self.assertIn("Action phase", " ".join(explanation.turns))


class InterfaceComplexityTests(unittest.TestCase):
    def test_live_complexity_consumes_and_preserves_game_config(self):
        configuration = stockpile.resolve_configuration(
            "lite",
            player_count=5,
            round_count=9,
            hand=True,
            dividend=True,
        )
        traversal = _complexity_result(configuration)
        static = {
            "num_distinct_actions": 21,
            "max_legal_actions": 8,
            "max_chance_outcomes": 6,
            "shared_action_head": 29,
            "max_game_length": 123,
            "observation_size": 256,
        }
        with (
            patch(
                "stockpile.stockpile_interface.platform."
                "compute_information_set_complexity",
                return_value=traversal,
            ) as compute,
            patch(
                "stockpile.stockpile_interface.platform.complexity_report",
                return_value=static,
            ) as report,
        ):
            result = stockpile.compute_interface_complexity(
                configuration,
                max_states=100,
                max_seconds=2.0,
                require_exact=False,
            )
        self.assertIs(result.information_set_complexity, traversal)
        self.assertIs(result.configuration, configuration)
        self.assertEqual(asdict(result.action_catalog), static)
        compute.assert_called_once_with(
            configuration.configured_game,
            max_states=100,
            max_seconds=2.0,
            require_exact=False,
        )
        report.assert_called_once_with(configuration.configured_game)

    def test_complexity_rejects_an_unresolved_platform_game(self):
        configuration = stockpile.resolve_configuration("lite")
        with self.assertRaisesRegex(TypeError, "resolved GameConfig"):
            stockpile.compute_interface_complexity(
                configuration.configured_game,
            )
        with self.assertRaisesRegex(TypeError, "resolved GameConfig"):
            stockpile.resolve_interface_complexity(
                configuration.configured_game,
                cache_policy="off",
            )

    def test_cache_off_resolution_preserves_game_config(self):
        configuration = stockpile.resolve_configuration("classic")
        traversal = _complexity_result(configuration)
        with patch(
            "stockpile.stockpile_interface.platform."
            "compute_information_set_complexity",
            return_value=traversal,
        ):
            resolved = stockpile.resolve_interface_complexity(
                configuration,
                cache_policy="off",
                max_states=10,
                max_seconds=1.0,
            )
        self.assertIs(resolved.configuration, configuration)
        self.assertIs(resolved.information_set_complexity, traversal)


if __name__ == "__main__":
    unittest.main()
