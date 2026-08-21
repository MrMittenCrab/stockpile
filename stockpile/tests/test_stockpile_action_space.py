"""Contract tests for Stockpile's OpenSpiel action-space design."""

from __future__ import annotations

import json
import random
import unittest

import pyspiel

import stockpile


def _parameters(
    profile: str,
    players: int,
    *,
    rounds: int = 6,
    deluxe_investors: bool = False,
    rule_overrides=None,
    action_space_mode: str = "compact",
):
    return stockpile.get_parameter_preset(
        profile,
        player_count=players,
        round_count=rounds,
        deluxe_investors=deluxe_investors,
        rule_overrides=rule_overrides,
        action_space_mode=action_space_mode,
    )


def _configured(profile: str, players: int, **values):
    return stockpile.configure_game(_parameters(profile, players, **values))


def _legal_actions(state):
    return list(state._legal_actions(state.current_player()))


def _max_branch_on_one_path(game, *, stop_at: int, seed: int = 7) -> int:
    state = game.new_initial_state()
    rng = random.Random(seed)
    maximum = 0
    for _ in range(2_000):
        if state.is_terminal():
            return maximum
        if state.is_chance_node():
            outcomes = list(state.chance_outcomes())
            actions, probabilities = zip(*outcomes, strict=True)
            state.apply_action(rng.choices(actions, weights=probabilities, k=1)[0])
            continue
        legal = _legal_actions(state)
        if not legal:
            raise AssertionError("Non-terminal player node has no legal actions")
        maximum = max(maximum, len(legal))
        if maximum >= stop_at:
            return maximum
        state.apply_action(legal[0])
    raise AssertionError("Game did not terminate within 2,000 micro-actions")


class ActionCatalogTests(unittest.TestCase):
    def test_compact_catalogs_cover_canonical_profiles_and_players(self):
        expected_catalogs = {
            "lite": {2: 18, 3: 17, 4: 18, 5: 19},
            "classic": {2: 28, 3: 27, 4: 28, 5: 29},
            "deluxe": {2: 41, 3: 36, 4: 38, 5: 40},
        }
        for profile, counts in expected_catalogs.items():
            for players, expected in counts.items():
                with self.subTest(profile=profile, players=players):
                    configured = _configured(profile, players)
                    self.assertEqual(
                        configured.rule_set.action_codec.num_distinct_actions,
                        expected,
                    )
                    self.assertEqual(configured.game.num_distinct_actions(), expected)
                    report = stockpile.complexity_report(configured)
                    self.assertEqual(report["max_legal_actions"], 8)
                    self.assertEqual(
                        report["shared_action_head"],
                        42 if profile == "deluxe" else 29,
                    )

    def test_catalog_namespaces_follow_effective_mechanics(self):
        lite = _configured("lite", 2).rule_set.action_codec
        classic = _configured("classic", 2).rule_set.action_codec
        deluxe = _configured("deluxe", 2).rule_set.action_codec
        investors = deluxe

        self.assertNotIn("company", lite.ranges)
        self.assertNotIn("direction", lite.ranges)
        self.assertNotIn("dividend_claim", lite.ranges)
        self.assertNotEqual(classic, deluxe)
        self.assertIn("company", classic.ranges)
        self.assertIn("use_ability", deluxe.ranges)
        self.assertIn("use_ability", investors.ranges)
        self.assertIn("investor_slot", investors.ranges)

        dividend_lite = stockpile.create_configuration(
            "lite",
            lite_options=("dividends",),
        )
        self.assertIn(
            "dividend_claim",
            dividend_lite.rule_set.action_codec.ranges,
        )

    def test_shared_mode_pads_without_changing_legal_namespace_ids(self):
        for profile in ("lite", "classic", "deluxe"):
            compact = _parameters(
                profile,
                3,
                action_space_mode="compact",
            )
            shared = _parameters(
                profile,
                3,
                action_space_mode="shared",
            )
            compact_rules = stockpile.configure_game(compact).rule_set
            shared_rules = stockpile.configure_game(shared).rule_set
            self.assertEqual(compact_rules.action_codec.ranges, shared_rules.action_codec.ranges)
            self.assertLessEqual(
                compact_rules.action_codec.num_distinct_actions,
                shared_rules.action_codec.num_distinct_actions,
            )

    def test_reported_chance_maxima_and_reachable_bid_branch(self):
        cases = (("lite", 6), ("classic", 11), ("deluxe", 11))
        for profile, expected in cases:
            with self.subTest(profile=profile):
                report = stockpile.complexity_report(
                    _parameters(profile, 3)
                )
                self.assertEqual(report["max_chance_outcomes"], expected)

        for profile in ("lite", "classic"):
            with self.subTest(profile=profile):
                observed = _max_branch_on_one_path(
                    _configured(profile, 2, rounds=1).game,
                    stop_at=8,
                )
                self.assertEqual(observed, 8)


class CompatibilityAndSerializationTests(unittest.TestCase):
    def test_canonical_enums_and_legacy_attributes(self):
        self.assertEqual(stockpile.RulesProfile.LITE.value, "lite")
        self.assertEqual(stockpile.RulesProfile.CLASSIC.value, "classic")
        self.assertEqual(stockpile.RulesProfile.DELUXE.value, "deluxe")
        self.assertEqual(stockpile.RulesProfile.CORE.value, "classic")
        self.assertEqual(stockpile.RulesProfile.FULL.value, "deluxe")
        self.assertEqual(
            tuple(option.value for option in stockpile.LiteOptionalRule),
            (
                "starting_share",
                "trading_fees",
                "dividends",
                "stock_splits",
                "majority_bonus",
            ),
        )

    def test_preset_factory_exposes_rounds_and_deluxe_investors(self):
        parameters = stockpile.get_parameter_preset(
            "deluxe",
            player_count=4,
            round_count=2,
            deluxe_investors=True,
            action_space_mode="shared",
        )
        self.assertEqual(parameters.rules_profile, "deluxe")
        self.assertEqual(parameters.round_count, 2)
        self.assertTrue(parameters.deluxe_investors)
        configured = stockpile.configure_game(parameters)
        self.assertEqual(configured.rule_set.round_count, 2)
        self.assertTrue(configured.rule_set.investors)
        self.assertEqual(configured.game.num_distinct_actions(), 42)

        for profile in ("lite", "classic"):
            with self.subTest(profile=profile):
                with self.assertRaisesRegex(ValueError, "Deluxe"):
                    stockpile.get_parameter_preset(
                        profile,
                        deluxe_investors=True,
                    )

    def test_profile_aliases_normalize_to_new_canonical_names(self):
        aliases = {
            "minimal_training": "lite",
            "core": "classic",
            "full": "deluxe",
            "expanded": "deluxe",
            "expanded_variants": "deluxe",
        }
        for alias, canonical in aliases.items():
            with self.subTest(alias=alias):
                configured = stockpile.configure_game(alias)
                self.assertEqual(configured.parameters.rules_profile, canonical)
                self.assertEqual(configured.rule_set.profile, canonical)
                serialized = json.loads(configured.parameters.model_dump_json())
                self.assertEqual(serialized["rules_profile"], canonical)

    def test_default_parameters_are_six_round_zero_share_lite(self):
        parameters = stockpile.GameParameters()
        configured = stockpile.configure_game(parameters)
        self.assertEqual(parameters.rules_profile, "lite")
        self.assertEqual(parameters.round_count, 6)
        self.assertFalse(parameters.deluxe_investors)
        self.assertEqual(configured.rule_set.starting_shares_per_player, 0)
        self.assertEqual(configured.game.num_distinct_actions(), 18)

    def test_round_and_player_bounds_are_validated(self):
        for rounds in (1, 10):
            configured = stockpile.configure_game(
                _parameters("classic", 3, rounds=rounds)
            )
            self.assertEqual(configured.rule_set.round_count, rounds)
        for rounds in (0, 11):
            with self.subTest(rounds=rounds):
                with self.assertRaises(ValueError):
                    _parameters("classic", 3, rounds=rounds)
        for players in (1, 6):
            with self.subTest(players=players):
                with self.assertRaises(ValueError):
                    _parameters("lite", players)

    def test_state_serialization_round_trip(self):
        game = _configured("lite", 2, rounds=1).game
        state = game.new_initial_state()
        serialized = state.serialize()
        restored = game.deserialize_state(serialized)
        self.assertEqual(restored.current_player(), state.current_player())
        self.assertEqual(_legal_actions(restored), _legal_actions(state))

        if hasattr(state, "to_dict") and hasattr(type(state), "from_dict"):
            payload = state.to_dict()
            json.dumps(payload)
            clone = type(state).from_dict(payload)
            self.assertEqual(clone.current_player(), state.current_player())
            self.assertEqual(_legal_actions(clone), _legal_actions(state))

    def test_pyspiel_scalar_loading_and_random_sim(self):
        game = pyspiel.load_game(
            "python_stockpile",
            {
                "players": 2,
                "rules_profile": "lite",
                "rounds": 1,
            },
        )
        self.assertEqual(game.get_parameters()["rounds"], 1)
        self.assertEqual(game.num_distinct_actions(), 18)
        pyspiel.random_sim_test(
            game,
            num_sims=2,
            serialize=True,
            verbose=False,
        )

    def test_random_sim_clone_and_serialize_with_sealed_selling(self):
        for profile in ("lite", "classic", "deluxe"):
            with self.subTest(profile=profile):
                game = _configured(
                    profile,
                    2,
                    rounds=1,
                    rule_overrides={"sell_order": False},
                ).game
                pyspiel.random_sim_test(
                    game,
                    num_sims=2,
                    serialize=True,
                    verbose=False,
                )

    def test_game_strings_use_canonical_names_and_retain_new_scalars(self):
        cases = (
            ("lite", "lite", False, False, 18),
            ("minimal_training", "lite", False, False, 18),
            ("classic", "classic", False, False, 28),
            ("core", "classic", False, False, 28),
            ("deluxe", "deluxe", False, True, 41),
            ("full", "deluxe", False, True, 41),
            ("expanded", "deluxe", False, True, 41),
            ("deluxe", "deluxe", True, True, 41),
        )
        for requested, canonical, requested_investors, effective_investors, actions in cases:
            with self.subTest(requested=requested, investors=requested_investors):
                game = pyspiel.load_game(
                    "python_stockpile",
                    {
                        "players": 2,
                        "rules_profile": requested,
                        "rounds": 2,
                        "deluxe_investors": requested_investors,
                    },
                )
                parameters = game.get_parameters()
                self.assertEqual(parameters["rules_profile"], canonical)
                self.assertEqual(parameters["rounds"], 2)
                self.assertEqual(
                    parameters["deluxe_investors"],
                    effective_investors,
                )
                self.assertEqual(game.num_distinct_actions(), actions)
                game_string = str(game)
                self.assertIn(f"rules_profile={canonical}", game_string)
                self.assertIn("rounds=2", game_string)
                reloaded = pyspiel.load_game(game_string)
                self.assertEqual(reloaded.get_parameters()["rules_profile"], canonical)
                self.assertEqual(reloaded.get_parameters()["rounds"], 2)
                self.assertEqual(reloaded.num_distinct_actions(), actions)

                state = game.new_initial_state()
                restored = reloaded.deserialize_state(state.serialize())
                self.assertEqual(restored.current_player(), state.current_player())
                self.assertEqual(_legal_actions(restored), _legal_actions(state))


if __name__ == "__main__":
    unittest.main()
