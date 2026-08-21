"""Tests for Stockpile information-set enumeration and parameter presets."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import stockpile


def _model_dict(model):
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    to_dict = getattr(model, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return dict(vars(model))


@dataclass
class _TinyState:
    """A tiny imperfect-information tree used for exact count assertions.

    Player 1's ``shared`` information set is reached along two histories.  Its
    legal action set is intentionally identical unless ``mismatch`` is true.
    """

    node: str = "root"
    mismatch: bool = False

    _TRANSITIONS = {
        ("root", 10): "chance",
        ("root", 11): "player_one_b",
        ("chance", 100): "player_one_a",
        ("chance", 101): "terminal_chance",
        ("player_one_a", 20): "terminal_a",
        ("player_one_b", 20): "terminal_b",
        ("player_one_b", 21): "terminal_b",
    }

    def clone(self):
        return _TinyState(self.node, self.mismatch)

    def apply_action(self, action):
        self.node = self._TRANSITIONS[(self.node, action)]

    def is_terminal(self):
        return self.node.startswith("terminal")

    def is_chance_node(self):
        return self.node == "chance"

    def chance_outcomes(self):
        return [(100, 0.5), (101, 0.5)] if self.is_chance_node() else []

    def current_player(self):
        if self.node == "root":
            return 0
        if self.node in {"player_one_a", "player_one_b"}:
            return 1
        return -1

    def legal_actions(self, player_id=None):
        del player_id
        if self.node == "root":
            return [10, 11]
        if self.node == "player_one_a":
            return [20]
        if self.node == "player_one_b":
            return [21] if self.mismatch else [20]
        return []

    def information_state_string(self, player_id=None):
        player_id = self.current_player() if player_id is None else player_id
        if self.node == "root" and player_id == 0:
            return "root-information"
        if self.node in {"player_one_a", "player_one_b"} and player_id == 1:
            return "shared-information"
        raise AssertionError("information_state_string called outside a player node")


@dataclass
class _TinyGame:
    mismatch: bool = False

    def new_initial_state(self):
        return _TinyState(mismatch=self.mismatch)


def _tiny_config(parameters, *, mismatch=False):
    return SimpleNamespace(
        parameters=parameters,
        game=_TinyGame(mismatch=mismatch),
        rule_set=None,
    )


class InformationSetEnumerationTests(unittest.TestCase):
    def setUp(self):
        self.parameters = stockpile.get_parameter_preset("lite")

    def _compute_tiny(self, *, mismatch=False, **kwargs):
        configured = _tiny_config(self.parameters, mismatch=mismatch)
        with patch.object(
            stockpile.stockpile_platform,
            "configure_game",
            return_value=configured,
        ):
            return stockpile.compute_information_set_complexity(
                self.parameters,
                **kwargs,
            )

    def test_exact_count_excludes_chance_and_terminal_nodes(self):
        result = self._compute_tiny(
            max_states=100,
            max_seconds=10.0,
            require_exact=True,
        )

        self.assertIsInstance(result, stockpile.InformationSetComplexity)
        self.assertTrue(result.exact)
        self.assertEqual(result.count_kind, "exact")
        self.assertIsNone(result.truncation_reason)
        self.assertEqual(result.states_visited, 7)
        self.assertEqual(result.terminal_states, 3)
        self.assertEqual(result.chance_nodes, 1)
        self.assertEqual(result.information_sets, 2)
        self.assertEqual(result.information_set_actions, 3)
        self.assertEqual(result.max_actions_per_information_set, 2)
        self.assertEqual(result.parameters.rules_profile, "lite")

    def test_per_player_counts_sum_to_the_totals(self):
        result = self._compute_tiny(max_states=100, max_seconds=10.0)

        self.assertEqual(result.per_player_information_sets, {0: 1, 1: 1})
        self.assertEqual(result.per_player_information_set_actions, {0: 2, 1: 1})
        self.assertEqual(
            sum(result.per_player_information_sets.values()),
            result.information_sets,
        )
        self.assertEqual(
            sum(result.per_player_information_set_actions.values()),
            result.information_set_actions,
        )

    def test_action_pairs_count_distinct_legal_ids_once_per_information_set(self):
        result = self._compute_tiny(max_states=100, max_seconds=10.0)

        # Root contributes (I0, 10) and (I0, 11).  The two histories reaching
        # player 1 share one information set and therefore contribute (I1, 20)
        # only once.
        self.assertEqual(result.information_sets, 2)
        self.assertEqual(result.information_set_actions, 3)

    def test_legal_action_mismatch_within_an_information_set_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "legal"):
            self._compute_tiny(
                mismatch=True,
                max_states=100,
                max_seconds=10.0,
            )

    def test_state_budget_returns_deterministic_lower_bound(self):
        first = self._compute_tiny(max_states=3, max_seconds=10.0)
        second = self._compute_tiny(max_states=3, max_seconds=10.0)

        self.assertFalse(first.exact)
        self.assertEqual(first.count_kind, "lower_bound")
        self.assertEqual(first.truncation_reason, "max_states")
        self.assertEqual(first.states_visited, 3)
        self.assertEqual(first.max_states, 3)
        self.assertEqual(first.max_seconds, 10.0)
        deterministic_fields = (
            "exact",
            "count_kind",
            "information_sets",
            "information_set_actions",
            "max_actions_per_information_set",
            "per_player_information_sets",
            "per_player_information_set_actions",
            "states_visited",
            "terminal_states",
            "chance_nodes",
            "max_states",
            "max_seconds",
            "truncation_reason",
        )
        for field in deterministic_fields:
            with self.subTest(field=field):
                self.assertEqual(getattr(first, field), getattr(second, field))

    def test_require_exact_turns_a_budget_limit_into_an_exception(self):
        with self.assertRaises(stockpile.InformationSetEnumerationLimit):
            self._compute_tiny(
                max_states=3,
                max_seconds=10.0,
                require_exact=True,
            )

    def test_real_lite_traversal_honors_a_deterministic_state_budget(self):
        first = stockpile.compute_information_set_complexity(
            self.parameters,
            max_states=25,
            max_seconds=10.0,
        )
        second = stockpile.compute_information_set_complexity(
            self.parameters,
            max_states=25,
            max_seconds=10.0,
        )

        self.assertEqual(first.states_visited, 25)
        self.assertEqual(first.count_kind, "lower_bound")
        self.assertEqual(first.truncation_reason, "max_states")
        self.assertEqual(first.information_sets, second.information_sets)
        self.assertEqual(first.information_set_actions, second.information_set_actions)
        self.assertEqual(
            first.per_player_information_sets,
            second.per_player_information_sets,
        )
        self.assertEqual(
            first.per_player_information_set_actions,
            second.per_player_information_set_actions,
        )

    def test_aliases_produce_canonical_equivalent_complexity_results(self):
        alias_groups = {
            "lite": ("minimal_training",),
            "classic": ("core",),
            "deluxe": ("full", "expanded", "expanded_variants"),
        }
        deterministic_fields = (
            "exact",
            "count_kind",
            "information_sets",
            "information_set_actions",
            "max_actions_per_information_set",
            "per_player_information_sets",
            "per_player_information_set_actions",
            "states_visited",
            "terminal_states",
            "chance_nodes",
            "max_states",
            "max_seconds",
            "truncation_reason",
        )
        for canonical, aliases in alias_groups.items():
            canonical_result = stockpile.compute_information_set_complexity(
                canonical,
                max_states=25,
                max_seconds=10.0,
            )
            self.assertEqual(canonical_result.parameters.rules_profile, canonical)
            for alias in aliases:
                with self.subTest(canonical=canonical, alias=alias):
                    alias_result = stockpile.compute_information_set_complexity(
                        alias,
                        max_states=25,
                        max_seconds=10.0,
                    )
                    self.assertEqual(alias_result.parameters.rules_profile, canonical)
                    for field in deterministic_fields:
                        self.assertEqual(
                            getattr(alias_result, field),
                            getattr(canonical_result, field),
                            field,
                        )


class ParameterPresetTests(unittest.TestCase):
    def test_canonical_and_legacy_names_normalize_to_the_same_presets(self):
        alias_groups = {
            "lite": ("minimal_training",),
            "classic": ("core",),
            "deluxe": ("full", "expanded", "expanded_variants"),
        }
        for canonical, aliases in alias_groups.items():
            canonical_parameters = stockpile.get_parameter_preset(canonical)
            self.assertEqual(canonical_parameters.rules_profile, canonical)
            for alias in aliases:
                with self.subTest(canonical=canonical, alias=alias):
                    alias_parameters = stockpile.get_parameter_preset(alias)
                    self.assertEqual(
                        _model_dict(canonical_parameters),
                        _model_dict(alias_parameters),
                    )
                    self.assertEqual(alias_parameters.rules_profile, canonical)

    def test_all_profiles_accept_every_supported_player_count(self):
        for profile in ("lite", "classic", "deluxe"):
            for player_count in range(2, 6):
                with self.subTest(profile=profile, player_count=player_count):
                    parameters = stockpile.get_parameter_preset(
                        profile,
                        player_count=player_count,
                    )
                    configured = stockpile.configure_game(parameters)
                    self.assertEqual(configured.parameters.player_count, player_count)

    def test_preset_preserves_overrides_and_action_space_mode(self):
        parameters = stockpile.get_parameter_preset(
            "lite",
            player_count=3,
            rule_overrides={"round_count": 2},
            action_space_mode="shared",
        )
        self.assertEqual(parameters.rule_overrides["round_count"], 2)
        self.assertEqual(parameters.action_space_mode, "shared")

    def test_round_count_accepts_one_through_ten_only(self):
        for round_count in (1, 10):
            with self.subTest(round_count=round_count):
                parameters = stockpile.get_parameter_preset(
                    "lite",
                    round_count=round_count,
                )
                configured = stockpile.configure_game(parameters)
                self.assertEqual(configured.rule_set.round_count, round_count)

        for round_count in (0, 11):
            with self.subTest(invalid_round_count=round_count):
                with self.assertRaisesRegex(ValueError, "round_count"):
                    stockpile.get_parameter_preset(
                        "lite",
                        round_count=round_count,
                    )

    def test_new_profiles_share_size_but_layer_rules(self):
        lite = stockpile.configure_game("lite")
        classic = stockpile.configure_game("classic")
        deluxe = stockpile.configure_game("deluxe")

        self.assertEqual(lite.parameters.rules_profile, "lite")
        self.assertEqual(classic.parameters.rules_profile, "classic")
        self.assertEqual(deluxe.parameters.rules_profile, "deluxe")
        self.assertEqual(lite.rule_set.company_count, 6)
        self.assertEqual(classic.rule_set.company_count, 6)
        self.assertEqual(lite.rule_set.starting_shares_per_player, 0)
        self.assertEqual(classic.rule_set.starting_shares_per_player, 1)
        self.assertFalse(lite.rule_set.market_action_cards)
        self.assertTrue(classic.rule_set.market_action_cards)
        self.assertFalse(classic.rule_set.advanced_price_tracks)
        self.assertFalse(deluxe.rule_set.advanced_price_tracks)
        self.assertFalse(classic.rule_set.investors)
        self.assertTrue(deluxe.rule_set.investors)

    def test_deluxe_investors_are_fixed_on_and_profile_scoped(self):
        default = stockpile.get_parameter_preset("deluxe")
        explicit = stockpile.get_parameter_preset(
            "deluxe",
            deluxe_investors=True,
        )
        self.assertTrue(default.deluxe_investors)
        self.assertTrue(stockpile.configure_game(default).rule_set.investors)
        self.assertTrue(stockpile.configure_game(explicit).rule_set.investors)
        with self.assertRaisesRegex(ValueError, "Deluxe"):
            stockpile.get_parameter_preset("classic", deluxe_investors=True)

    def test_unknown_preset_is_rejected(self):
        with self.assertRaises(ValueError):
            stockpile.get_parameter_preset("training-ish")


if __name__ == "__main__":
    unittest.main()
