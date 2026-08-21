"""Contract and rules tests for :mod:`stockpile`.

The tests intentionally use only the Python standard library.  Small adapter
helpers below accept either dataclass/Pydantic-style value objects or their
plain mapping representation, while the public function and method names are
tested exactly.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import unittest

import stockpile


OPTIONAL_FEATURES = (
    "trading_fees",
    "market_action_cards",
    "stock_boom_cards",
    "stock_bust_cards",
    "forecast_dividends",
    "dividend_reveal_choice",
    "stock_splits",
    "repeat_split_bonus",
    "bankruptcy",
    "majority_bonus",
    "blind_information_pairs",
    "partial_sales",
    "advanced_price_tracks",
    "advanced_track_dividends",
    "investors",
)


def _plain(value):
    """Return a stable, recursively JSON-compatible representation."""

    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if is_dataclass(value):
        return _plain(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _plain(model_dump(mode="json"))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: _plain(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _fingerprint(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"))


def _nested_value(value, key, default=None):
    """Find the first exact key in a nested public representation."""

    value = _plain(value)
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for child in value.values():
            found = _nested_value(child, key, default)
            if found is not default:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _nested_value(child, key, default)
            if found is not default:
                return found
    return default


def _member(value, *names):
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
        if isinstance(value, Mapping) and name in value:
            return value[name]
    raise AssertionError(f"{type(value).__name__} has none of {names!r}")


def _bool_rule(rule_set, name):
    marker = object()
    value = _nested_value(rule_set, name, marker)
    if value is marker:
        raise AssertionError(f"RuleSet does not expose {name!r}")
    return bool(value)


def _report_is_valid(report):
    data = _plain(report)
    valid = bool(data.get("valid", True))
    legal = data.get("legal")
    return valid if legal is None else valid and bool(legal)


def _action_id(action):
    return int(_member(action, "action_id", "id"))


def _configure(
    profile="lite",
    *,
    overrides=None,
    player_count=2,
    round_count=6,
    deluxe_investors=False,
):
    parameters = stockpile.GameParameters(
        player_count=player_count,
        rules_profile=profile,
        round_count=round_count,
        deluxe_investors=deluxe_investors,
        board_side="standard",
        investor_mode="none",
        rule_overrides={} if overrides is None else overrides,
    )
    return stockpile.configure_game(parameters)


def _initial_state(configured, seed=7):
    initial_input = stockpile.randomize_initial_input(configured.rule_set, seed)
    state, report = stockpile.initialize_game(
        configured.rule_set,
        configured.game,
        initial_input,
    )
    if not _report_is_valid(report):
        raise AssertionError(f"initialization failed: {_plain(report)!r}")
    return initial_input, state


def _play_first_legal(configured, seed=7, limit=1_000):
    """Play a reproducible game using the lowest currently legal action id."""

    _, state = _initial_state(configured, seed)
    action_ids = []
    for _ in range(limit):
        if state.is_terminal():
            return state, action_ids
        actor = state.current_player()
        information, actions = stockpile.observe_game_state(
            configured.rule_set,
            state,
            actor,
        )
        del information
        if not actions:
            raise AssertionError(f"non-terminal state has no legal actions: {state}")
        action = min(actions, key=_action_id)
        request = stockpile.ActionRequest(player_id=actor, action_id=_action_id(action))
        state, _, report = stockpile.advance_game(
            configured.rule_set,
            state,
            actions,
            request,
        )
        if not _report_is_valid(report):
            raise AssertionError(f"legal transition failed: {_plain(report)!r}")
        action_ids.append(_action_id(action))
    raise AssertionError(f"game did not terminate within {limit} player decisions")


def _observe(configured, state, player):
    return stockpile.observe_game_state(configured.rule_set, state, player)


def _apply_player_action(configured, state, action):
    actor = state.current_player()
    _, actions = _observe(configured, state, actor)
    state, _, report = stockpile.advance_game(
        configured.rule_set,
        state,
        actions,
        stockpile.ActionRequest(player_id=actor, action_id=_action_id(action)),
    )
    if not _report_is_valid(report):
        raise AssertionError(f"legal transition failed: {_plain(report)!r}")
    return state


def _apply_first_player_action(configured, state):
    _, actions = _observe(configured, state, state.current_player())
    if not actions:
        raise AssertionError(f"non-terminal state has no legal actions: {state}")
    return _apply_player_action(configured, state, min(actions, key=_action_id))


def _resolve_chance_to_player(state, limit=1_000):
    """Apply the lowest explicit chance outcome until a player must act."""

    count = 0
    while state.is_chance_node():
        outcomes = list(state.chance_outcomes())
        if not outcomes:
            raise AssertionError("chance node has no outcomes")
        state.apply_action(min(outcomes, key=lambda item: int(item[0]))[0])
        count += 1
        if count > limit:
            raise AssertionError("chance setup did not reach a player decision")
    return count


def _public_player_value(value, player):
    value = _plain(value)
    if isinstance(value, Mapping):
        for key in (player, str(player)):
            if key in value:
                return value[key]
    if isinstance(value, list) and player < len(value):
        return value[player]
    raise AssertionError(f"cannot find player {player} in {value!r}")


def _card_ids(cards):
    cards = _plain(cards)
    if isinstance(cards, Mapping):
        result = {
            int(key)
            for key in cards
            if str(key).lstrip("-").isdigit()
        }
        result.update(_card_ids(list(cards.values())))
        return result
    result = set()
    for card in cards or []:
        if isinstance(card, Mapping) and "card_id" in card:
            result.add(int(card["card_id"]))
        elif isinstance(card, (int, str)) and str(card).lstrip("-").isdigit():
            result.add(int(card))
    return result


def _investor_tokens(values):
    """Stable identities for Investor strings or public card records."""

    tokens = []
    for value in _plain(values) or []:
        if isinstance(value, Mapping):
            token = value.get("name", value.get("investor", value.get("value", value)))
        else:
            token = value
        tokens.append(_fingerprint(token))
    return tokens


def _find_action(actions, *, action_type=None, text=None, ordinal=None):
    for action in actions:
        data = _plain(action)
        if action_type is not None and data.get("action_type") != action_type:
            continue
        if ordinal is not None:
            payload = data.get("payload", {})
            candidate = payload.get("ordinal") if isinstance(payload, Mapping) else None
            if candidate != ordinal:
                continue
        if text is not None and text.lower() not in json.dumps(data, sort_keys=True).lower():
            continue
        return action
    raise AssertionError(
        f"no matching action type={action_type!r} text={text!r} "
        f"ordinal={ordinal!r}; actions={_plain(actions)!r}"
    )


def _complete_supply_placement(configured, state):
    for _ in range(3):
        state = _apply_first_player_action(configured, state)
    return state


class ConfigurationTests(unittest.TestCase):
    def test_mapping_and_model_inputs_normalize_identically(self):
        model = stockpile.GameParameters(
            player_count=2,
            rules_profile="lite",
            board_side="standard",
            investor_mode="none",
            rule_overrides={},
        )
        from_model = stockpile.configure_game(model)
        from_mapping = stockpile.configure_game(
            {
                "player_count": 2,
                "rules_profile": "lite",
                "board_side": "standard",
                "investor_mode": "none",
                "rule_overrides": {},
            }
        )
        self.assertEqual(_plain(from_model.parameters), _plain(from_mapping.parameters))
        self.assertEqual(_plain(from_model.rule_set), _plain(from_mapping.rule_set))
        self.assertIsInstance(from_model.parameter_schema, Mapping)
        self.assertIsInstance(from_model.state_schema, Mapping)

    def test_lite_profile_uses_shared_size_and_simplified_rules(self):
        configured = _configure()
        parameters = _plain(configured.parameters)
        rules = configured.rule_set

        self.assertEqual(parameters["rules_profile"], "lite")
        self.assertEqual(
            _plain(stockpile.GameParameters())["rules_profile"],
            "lite",
        )
        self.assertEqual(
            _plain(stockpile.configure_game({}).parameters)["rules_profile"],
            "lite",
        )
        self.assertEqual(_nested_value(stockpile.StockpileGame().rule_set, "profile"), "lite")
        self.assertEqual(_nested_value(rules, "company_count"), 6)
        self.assertEqual(_nested_value(rules, "round_count"), 6)
        self.assertEqual(_nested_value(rules, "two_player_topology"), "official")
        self.assertEqual(
            _nested_value(rules, "bid_values"),
            [0, 1, 3, 6, 10, 15, 20, 25],
        )
        self.assertEqual(
            _nested_value(rules, "forecast_values"),
            [4, 2, 1, 0, -2, -3],
        )
        self.assertEqual(_nested_value(rules, "starting_shares_per_player"), 0)
        for feature in (
            "trading_fees",
            "market_action_cards",
            "stock_boom_cards",
            "stock_bust_cards",
            "forecast_dividends",
            "dividend_reveal_choice",
            "stock_splits",
            "repeat_split_bonus",
            "majority_bonus",
            "advanced_price_tracks",
            "advanced_track_dividends",
            "investors",
        ):
            self.assertFalse(_bool_rule(rules, feature), feature)
        for feature in ("bankruptcy", "blind_information_pairs", "partial_sales"):
            self.assertTrue(_bool_rule(rules, feature), feature)
        self.assertFalse(rules.sequential_observable_selling)

    def test_lite_can_opt_into_sequential_observable_selling(self):
        rules = _configure(
            "lite",
            overrides={"sell_order": True},
        ).rule_set

        self.assertTrue(rules.sequential_observable_selling)

    def test_classic_and_deluxe_profiles_are_layered(self):
        classic = _configure("classic").rule_set
        deluxe = _configure("deluxe").rule_set

        self.assertEqual(_nested_value(classic, "company_count"), 6)
        self.assertEqual(_nested_value(classic, "round_count"), 6)
        self.assertEqual(_nested_value(classic, "starting_shares_per_player"), 1)
        for feature in (
            "trading_fees",
            "market_action_cards",
            "forecast_dividends",
            "dividend_reveal_choice",
            "stock_splits",
            "repeat_split_bonus",
            "bankruptcy",
            "majority_bonus",
            "blind_information_pairs",
            "partial_sales",
        ):
            self.assertTrue(_bool_rule(classic, feature), feature)
        for rules in (classic, deluxe):
            self.assertFalse(_bool_rule(rules, "advanced_price_tracks"))
            self.assertFalse(_bool_rule(rules, "advanced_track_dividends"))
            self.assertTrue(rules.sequential_observable_selling)
        self.assertFalse(_bool_rule(classic, "investors"))
        self.assertTrue(_bool_rule(deluxe, "investors"))
        self.assertEqual(deluxe.enabled_investors, stockpile.stockpile_platform.INVESTOR_NAMES)

    def test_grouped_optional_overrides_apply_to_every_profile(self):
        grouped = {
            "hand": False,
            "fees": False,
            "dividend": False,
            "split": False,
            "majority": False,
            "stock_tracks": True,
            "sell_order": False,
        }
        for profile in ("lite", "classic", "deluxe"):
            with self.subTest(profile=profile):
                rules = stockpile.configure_game(
                    stockpile.get_parameter_preset(
                        profile,
                        rule_overrides=grouped,
                    )
                ).rule_set
                self.assertEqual(rules.starting_shares_per_player, 0)
                self.assertFalse(rules.trading_fees)
                self.assertFalse(rules.forecast_dividends)
                self.assertFalse(rules.dividend_reveal_choice)
                self.assertFalse(rules.stock_splits)
                self.assertFalse(rules.repeat_split_bonus)
                self.assertFalse(rules.majority_bonus)
                self.assertTrue(rules.advanced_price_tracks)
                self.assertFalse(rules.advanced_track_dividends)
                self.assertFalse(rules.sequential_observable_selling)
                self.assertEqual(rules.investors, profile == "deluxe")
                self.assertEqual(rules.market_action_cards, profile != "lite")

    def test_canonical_presets_reject_fixed_layer_overrides(self):
        for key in (
            "market_action_cards",
            "stock_boom_cards",
            "stock_bust_cards",
            "investors",
            "enabled_investors",
        ):
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError,
                "does not allow",
            ):
                stockpile.get_parameter_preset(
                    "deluxe",
                    rule_overrides={key: False},
                )

    def test_core_optional_features_can_all_be_disabled(self):
        overrides = {feature: False for feature in OPTIONAL_FEATURES}
        overrides.update(
            {
                "company_count": 3,
                "round_count": 1,
                "bid_values": [0, 1, 3, 6],
                "starting_cash": 10,
                "forecast_values": [-2, 0, 2],
                "two_player_topology": "simple",
            }
        )
        rules = _configure("classic", overrides=overrides).rule_set
        for feature in OPTIONAL_FEATURES:
            self.assertFalse(_bool_rule(rules, feature), feature)

    def test_fixed_core_cannot_be_disabled(self):
        fixed_core_keys = (
            "insider_information",
            "stockpile_construction",
            "ascending_auction",
            "secret_portfolios",
            "selling_phase",
            "price_movement",
            "terminal_liquidation",
        )
        for key in fixed_core_keys:
            with self.subTest(key=key), self.assertRaises((TypeError, ValueError)):
                _configure(overrides={key: False})

    def test_feature_dependencies_are_rejected(self):
        invalid_overrides = (
            {"stock_splits": False, "repeat_split_bonus": True},
            {"forecast_dividends": False, "dividend_reveal_choice": True},
            {"advanced_price_tracks": False, "advanced_track_dividends": True},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides), self.assertRaises((TypeError, ValueError)):
                _configure("classic", overrides=overrides)

    def test_deluxe_investors_remain_valid_when_fees_and_dividends_are_off(self):
        rules = stockpile.configure_game(
            stockpile.get_parameter_preset(
                "deluxe",
                rule_overrides={"fees": False, "dividend": False},
            )
        ).rule_set
        self.assertTrue(rules.investors)
        self.assertIn("broker_bernie", rules.enabled_investors)
        self.assertIn("dividend_deborah", rules.enabled_investors)
        self.assertFalse(rules.trading_fees)
        self.assertFalse(rules.forecast_dividends)

    def test_profile_aliases_normalize_to_lite_classic_and_deluxe(self):
        aliases = {
            "minimal_training": "lite",
            "core": "classic",
            "full": "deluxe",
            "expanded": "deluxe",
            "expanded_variants": "deluxe",
        }
        for alias, canonical in aliases.items():
            with self.subTest(alias=alias):
                configured = _configure(alias)
                self.assertEqual(
                    _plain(configured.parameters)["rules_profile"],
                    canonical,
                )
                self.assertEqual(_nested_value(configured.rule_set, "profile"), canonical)


class SetupAndTransitionTests(unittest.TestCase):
    def setUp(self):
        self.configured = _configure()

    def test_lite_setup_is_seed_deterministic(self):
        first = stockpile.randomize_initial_input(self.configured.rule_set, 314159)
        second = stockpile.randomize_initial_input(self.configured.rule_set, 314159)
        self.assertEqual(_fingerprint(first), _fingerprint(second))

        state_a, report_a = stockpile.initialize_game(
            self.configured.rule_set, self.configured.game, first
        )
        state_b, report_b = stockpile.initialize_game(
            self.configured.rule_set, self.configured.game, second
        )
        self.assertTrue(_report_is_valid(report_a))
        self.assertTrue(_report_is_valid(report_b))
        self.assertEqual(_fingerprint(state_a), _fingerprint(state_b))

    def test_high_round_market_decks_always_cover_all_required_draws(self):
        for profile in ("lite", "classic", "deluxe"):
            for players in range(2, 6):
                with self.subTest(profile=profile, players=players):
                    rules = _configure(
                        profile,
                        player_count=players,
                        round_count=10,
                    ).rule_set
                    initial = stockpile.randomize_initial_input(rules, 41)
                    required_draws = rules.round_count * (
                        rules.stockpile_count
                        + 2 * rules.player_count * rules.supply_batches
                    )
                    self.assertGreaterEqual(
                        len(initial.market_deck_order),
                        required_draws,
                    )

    def test_high_round_market_decks_repeat_only_complete_templates(self):
        platform = stockpile.stockpile_platform
        for profile in ("lite", "classic", "deluxe"):
            for players in range(2, 6):
                with self.subTest(profile=profile, players=players):
                    rules = _configure(
                        profile,
                        player_count=players,
                        round_count=10,
                    ).rule_set
                    template = platform._market_templates(rules)
                    deck = platform._build_market_deck(rules)
                    self.assertEqual(len(deck) % len(template), 0)
                    copies = len(deck) // len(template)
                    expected = Counter(template)
                    actual = Counter(
                        (card.card_type, card.company_id, card.value)
                        for card in deck
                    )
                    self.assertEqual(
                        actual,
                        Counter(
                            {
                                key: count * copies
                                for key, count in expected.items()
                            }
                        ),
                    )

    def test_seeded_deluxe_prices_match_advanced_track_indices(self):
        platform = stockpile.stockpile_platform
        configured = _configure(
            "deluxe",
            player_count=5,
            round_count=10,
            overrides={"stock_tracks": True},
        )
        initial = stockpile.randomize_initial_input(configured.rule_set, 73)
        expected_prices = tuple(
            platform.ADVANCED_TRACKS[company][
                platform.ADVANCED_START_INDEX[company]
            ]
            for company in range(configured.rule_set.company_count)
        )
        self.assertEqual(initial.starting_prices, expected_prices)

        state, report = stockpile.initialize_game(
            configured.rule_set,
            configured.game,
            initial,
        )
        self.assertTrue(_report_is_valid(report))
        for company, name in enumerate(configured.rule_set.company_names):
            self.assertEqual(
                state.prices[name],
                platform.ADVANCED_TRACKS[company][state.price_indices[company]],
            )

    def test_lite_starting_share_option_controls_setup_draws(self):
        without = self.configured
        with_share = _configure(
            overrides={"starting_share": True},
        )
        without_input = stockpile.randomize_initial_input(without.rule_set, 9)
        with_input = stockpile.randomize_initial_input(with_share.rule_set, 9)
        self.assertEqual(without.rule_set.starting_shares_per_player, 0)
        self.assertEqual(with_share.rule_set.starting_shares_per_player, 1)
        self.assertEqual(tuple(without_input.starting_shares), ())
        self.assertEqual(len(with_input.starting_shares), 2)

        without_state = without.game.new_initial_state()
        with_state = with_share.game.new_initial_state()
        self.assertEqual(without_state._chance_kind, "first_player")
        self.assertEqual(with_state._chance_kind, "starting_share")

    def test_lite_deck_omits_disabled_card_types(self):
        initial = stockpile.randomize_initial_input(self.configured.rule_set, 19)
        data = _plain(initial)
        deck = None
        for key in ("market_deck_order", "deck_order", "market_deck"):
            deck = _nested_value(data, key)
            if deck is not None:
                break
        self.assertIsNotNone(deck, "InitialInput must expose the sampled Market deck")
        text = json.dumps(deck, sort_keys=True).lower()
        self.assertNotIn("trading_fee", text)
        self.assertNotIn("boom", text)
        self.assertNotIn("bust", text)
        self.assertNotIn("dividend", text)

    def test_advance_is_functional_and_rejects_unknown_action(self):
        _, state = _initial_state(self.configured, 23)
        actor = state.current_player()
        _, legal_actions = stockpile.observe_game_state(
            self.configured.rule_set, state, actor
        )
        before = _fingerprint(state)
        legal = min(legal_actions, key=_action_id)
        next_state, _, report = stockpile.advance_game(
            self.configured.rule_set,
            state,
            legal_actions,
            stockpile.ActionRequest(player_id=actor, action_id=_action_id(legal)),
        )
        self.assertTrue(_report_is_valid(report))
        self.assertEqual(_fingerprint(state), before, "advance_game mutated its input")
        self.assertIsNot(next_state, state)

        next_before = _fingerprint(next_state)
        next_actor = next_state.current_player()
        _, next_legal = stockpile.observe_game_state(
            self.configured.rule_set, next_state, next_actor
        )
        rejected_state, _, rejected_report = stockpile.advance_game(
            self.configured.rule_set,
            next_state,
            next_legal,
            stockpile.ActionRequest(player_id=next_actor, action_id=-1),
        )
        self.assertFalse(_report_is_valid(rejected_report))
        self.assertEqual(_fingerprint(rejected_state), next_before)
        self.assertEqual(_fingerprint(next_state), next_before)

    def test_seeded_lite_playthrough_terminates_deterministically(self):
        state_a, actions_a = _play_first_legal(self.configured, seed=101)
        state_b, actions_b = _play_first_legal(self.configured, seed=101)
        self.assertTrue(state_a.is_terminal())
        self.assertTrue(state_b.is_terminal())
        self.assertEqual(actions_a, actions_b)
        self.assertEqual(_fingerprint(state_a), _fingerprint(state_b))

        prices = _nested_value(state_a, "prices")
        self.assertIsInstance(prices, Mapping)
        self.assertTrue(prices)
        self.assertTrue(all(1 <= int(price) <= 10 for price in prices.values()))


class DetailedRuleBehaviorTests(unittest.TestCase):
    def test_official_two_player_second_supply_batch_is_dealt_only_after_first_batch(self):
        configured = _configure("classic", player_count=2)
        state = configured.game.new_initial_state()
        _resolve_chance_to_player(state)

        self.assertEqual(_nested_value(state, "phase"), "supply")
        self.assertEqual(_nested_value(state, "stage"), "supply_card")
        first_actor = state.current_player()
        second_actor = 1 - first_actor
        first_info, _ = _observe(configured, state, first_actor)
        second_info, _ = _observe(configured, state, second_actor)
        spectator, _ = _observe(configured, state, None)
        self.assertEqual(len(_member(first_info, "private_hand")), 2)
        self.assertEqual(len(_member(second_info, "private_hand")), 2)
        self.assertFalse(_member(spectator, "private_hand"))

        piles_before = _fingerprint(
            _nested_value(_member(second_info, "public_state"), "stockpiles")
        )
        state = _apply_first_player_action(configured, state)
        state = _apply_first_player_action(configured, state)
        pending_info, _ = _observe(configured, state, second_actor)
        self.assertEqual(
            _fingerprint(
                _nested_value(_member(pending_info, "public_state"), "stockpiles")
            ),
            piles_before,
            "a pending face-up selection must remain private until both cards commit",
        )

        state = _apply_first_player_action(configured, state)
        committed_info, _ = _observe(configured, state, second_actor)
        self.assertNotEqual(
            _fingerprint(
                _nested_value(_member(committed_info, "public_state"), "stockpiles")
            ),
            piles_before,
        )
        former_actor_info, _ = _observe(configured, state, first_actor)
        self.assertFalse(_member(former_actor_info, "private_hand"))
        self.assertEqual(len(_member(committed_info, "private_hand")), 2)

        state = _complete_supply_placement(configured, state)
        self.assertTrue(
            state.is_chance_node(),
            "official two-player cards 5-8 must be chance-dealt after batch one commits",
        )
        for player in range(2):
            information, _ = _observe(configured, state, player)
            self.assertFalse(_member(information, "private_hand"))

        second_batch_draws = _resolve_chance_to_player(state)
        self.assertEqual(second_batch_draws, 4)
        self.assertEqual(_nested_value(state, "stage"), "supply_card")
        for player in range(2):
            information, _ = _observe(configured, state, player)
            self.assertEqual(len(_member(information, "private_hand")), 2)

    def test_fee_debts_and_chosen_investors_are_public(self):
        configured = _configure(
            "deluxe",
            player_count=2,
            deluxe_investors=True,
        )
        state = configured.game.new_initial_state()
        state.players[0].fees = [1, 3]
        state.players[0].investors = ["bill", "broker_bernie"]
        state.players[1].investors = ["golden_graham", "wise_warren"]

        observations = [
            _observe(configured, state, player)[0]
            for player in (0, 1, None)
        ]
        expected_investors = {
            0: {"bill", "broker_bernie"},
            1: {"golden_graham", "wise_warren"},
        }
        for information in observations:
            public = _plain(_member(information, "public_state"))
            debts = public.get("fee_debts")
            if debts is None and isinstance(public.get("players"), (Mapping, list)):
                debts = {
                    player: _nested_value(
                        _public_player_value(public["players"], player),
                        "fee_debts",
                        [],
                    )
                    for player in range(2)
                }
            self.assertIsNotNone(debts, "public state must expose outstanding fee debts")
            self.assertEqual(_public_player_value(debts, 0), [1, 3])

            public_investors = _member(information, "public_investors")
            for player, expected in expected_investors.items():
                self.assertEqual(
                    set(_public_player_value(public_investors, player)),
                    expected,
                )

    def test_full_investor_deal_is_unique_private_and_keep_two_is_interactive(self):
        configured = _configure(
            "deluxe",
            player_count=2,
            deluxe_investors=True,
        )
        state = configured.game.new_initial_state()
        offers = {}
        keep_actions = {0: 0, 1: 0}
        selected = None

        for _ in range(100):
            if state.is_chance_node():
                outcomes = list(state.chance_outcomes())
                self.assertTrue(outcomes)
                state.apply_action(min(outcomes, key=lambda item: int(item[0]))[0])
                continue

            actor = state.current_player()
            information, actions = _observe(configured, state, actor)
            offer = _member(information, "private_investor_offer")
            if offer and not offers:
                for player in range(2):
                    player_information, _ = _observe(configured, state, player)
                    offers[player] = _investor_tokens(
                        _member(player_information, "private_investor_offer")
                    )
                    self.assertEqual(len(offers[player]), 4)
                spectator, _ = _observe(configured, state, None)
                self.assertFalse(_member(spectator, "private_investor_offer"))
                self.assertTrue(set(offers[0]).isdisjoint(offers[1]))

            investor_actions = [
                action
                for action in actions
                if _nested_value(_plain(action).get("payload", {}), "stage")
                == "investor_select"
            ]
            if not investor_actions:
                self.fail(
                    "full setup reached a player node without an interactive Investor keep"
                )
            state = _apply_player_action(
                configured,
                state,
                min(investor_actions, key=_action_id),
            )
            keep_actions[actor] += 1

            spectator, _ = _observe(configured, state, None)
            public_investors = _member(spectator, "public_investors")
            if all(len(_public_player_value(public_investors, p)) == 2 for p in range(2)):
                selected = public_investors
                break
        else:
            self.fail("Investor selection did not finish")

        self.assertEqual(keep_actions, {0: 2, 1: 2})
        self.assertEqual(set(offers), {0, 1})
        dealt = offers[0] + offers[1]
        self.assertEqual(len(dealt), 8)
        self.assertEqual(len(set(dealt)), 8, "the eight dealt cards must be unique")
        for player in range(2):
            chosen = _investor_tokens(_public_player_value(selected, player))
            self.assertEqual(len(chosen), 2)
            self.assertTrue(set(chosen).issubset(set(offers[player])))

    def test_two_same_phase_investors_can_be_ordered_and_remaining_one_skipped(self):
        configured = _configure(
            "deluxe",
            player_count=2,
            deluxe_investors=True,
        )
        state = configured.game.new_initial_state()
        state._chance_kind = ""
        state.first_player = 0
        state.players[0].investors = ["maverick_mark", "wise_warren"]
        state.players[1].investors = []
        state.stockpiles[0].face_down_cards = [
            stockpile.Card(
                card_id=900_001,
                card_type="stock",
                company_id=0,
                value=1,
                face_up=False,
                location="stockpile:0",
            )
        ]
        state._begin_investor_pre_demand()

        _, actions = _observe(configured, state, 0)
        self.assertIsNotNone(_find_action(actions, action_type="done"))
        mark = _find_action(actions, action_type="investor_slot", ordinal=0)
        warren = _find_action(actions, action_type="investor_slot", ordinal=1)
        self.assertNotEqual(_action_id(mark), _action_id(warren))

        state.apply_action(_action_id(warren))
        _, actions = _observe(configured, state, 0)
        pile_zero = _find_action(actions, action_type="pile", ordinal=0)
        state.apply_action(_action_id(pile_zero))

        self.assertEqual(state.current_player(), 0)
        self.assertEqual(_nested_value(state, "stage"), "investor_offer")
        warren_information, _ = _observe(configured, state, 0)
        self.assertIn(
            900_001,
            _card_ids(_member(warren_information, "known_cards")),
        )
        _, remaining = _observe(configured, state, 0)
        self.assertIsNotNone(
            _find_action(remaining, action_type="investor_slot", ordinal=0)
        )
        remaining_ordinals = {
            _nested_value(_plain(action).get("payload", {}), "ordinal")
            for action in remaining
            if _plain(action).get("action_type") == "investor_slot"
        }
        self.assertNotIn(1, remaining_ordinals)
        done = _find_action(remaining, action_type="done")
        state.apply_action(_action_id(done))
        self.assertNotEqual(_nested_value(state, "stage"), "investor_offer")
        self.assertEqual(_nested_value(state, "phase"), "demand")

    def test_dividend_deborah_may_decline_at_round_end(self):
        configured = _configure(
            "deluxe",
            player_count=2,
            deluxe_investors=True,
        )
        state = configured.game.new_initial_state()
        state._chance_kind = ""
        state.first_player = 0
        state.players[0].investors = ["dividend_deborah"]
        state.players[1].investors = []
        cash_before = state.players[0].cash
        state._begin_deborah_or_finish_round()

        _, actions = _observe(configured, state, 0)
        done = _find_action(actions, action_type="done")
        self.assertTrue(
            any(_plain(action).get("action_type") == "company" for action in actions)
        )
        state.apply_action(_action_id(done))
        self.assertEqual(state.players[0].cash, cash_before)
        self.assertNotEqual(_nested_value(state, "stage"), "deborah_company")

    def test_dividend_reveal_is_decided_per_physical_holding(self):
        configured = _configure("classic", player_count=2)
        _, state = _initial_state(configured, 123)
        state._chance_kind = ""
        for player in state.players:
            player.private_information.clear()
            player.revealed_information.clear()
            player.regular_portfolio = [0] * configured.rule_set.company_count
            player.split_portfolio = [0] * configured.rule_set.company_count
        state.public_information = [(0, "DIVIDEND")]
        state.blind_information = []
        state.players[0].regular_portfolio[0] = 2
        state.players[0].split_portfolio[0] = 1
        cash_before = state.players[0].cash
        state._begin_movement()

        _, actions = _observe(configured, state, 0)
        claim = _find_action(actions, action_type="dividend_claim", ordinal=1)
        state.apply_action(_action_id(claim))
        self.assertEqual(state.players[0].cash, cash_before + 2)

        _, actions = _observe(configured, state, 0)
        waive = _find_action(actions, action_type="dividend_claim", ordinal=0)
        state.apply_action(_action_id(waive))
        self.assertEqual(state.players[0].cash, cash_before + 2)

        _, actions = _observe(configured, state, 0)
        claim = _find_action(actions, action_type="dividend_claim", ordinal=1)
        state.apply_action(_action_id(claim))
        self.assertEqual(
            state.players[0].cash,
            cash_before + 6,
            "two claimed regular shares pay $2K each and one split card pays $4K",
        )
        self.assertEqual(state.players[0].regular_portfolio[0], 2)
        self.assertEqual(state.players[0].split_portfolio[0], 1)

    def test_repeat_split_pays_twenty_thousand_per_split_card(self):
        configured = _configure("classic", player_count=2)
        _, state = _initial_state(configured, 456)
        player = state.players[0]
        player.cash = 0
        player.fees.clear()
        player.regular_portfolio[0] = 0
        player.split_portfolio[0] = 1

        state._trigger_split(0)
        self.assertEqual(player.cash, 20)
        self.assertEqual(player.split_portfolio[0], 1)

    def test_private_hands_and_legally_known_cards_are_masked(self):
        configured = _configure("classic", player_count=3)
        state = configured.game.new_initial_state()
        _resolve_chance_to_player(state)
        actor = state.current_player()
        others = [player for player in range(3) if player != actor]

        hands = {}
        for player in range(3):
            information, _ = _observe(configured, state, player)
            hands[player] = _card_ids(_member(information, "private_hand"))
            self.assertEqual(len(hands[player]), 2)
        self.assertEqual(len(set().union(*hands.values())), 6)
        spectator, _ = _observe(configured, state, None)
        self.assertFalse(_member(spectator, "private_hand"))

        state = _complete_supply_placement(configured, state)
        hidden_cards = [
            card
            for pile in state.stockpiles
            for card in pile.face_down_cards
        ]
        self.assertTrue(hidden_cards)
        hidden = hidden_cards[0]
        viewer, uninformed = others
        state.players[viewer].viewed_cards.add(hidden.card_id)
        state.players[viewer].known_cards[hidden.card_id] = hidden

        viewer_info, _ = _observe(configured, state, viewer)
        uninformed_info, _ = _observe(configured, state, uninformed)
        spectator, _ = _observe(configured, state, None)
        self.assertIn(hidden.card_id, _card_ids(_member(viewer_info, "known_cards")))
        self.assertNotIn(
            hidden.card_id,
            _card_ids(_member(uninformed_info, "known_cards")),
        )
        self.assertNotIn(hidden.card_id, _card_ids(_member(spectator, "known_cards")))

    def test_sealed_sales_are_private_until_one_atomic_batch_settlement(self):
        configured = _configure(
            "lite",
            round_count=1,
        )
        state = configured.game.new_initial_state()
        state._chance_kind = ""
        state.first_player = 0
        state.players[0].regular_portfolio[0] = 2
        state.players[1].regular_portfolio[0] = 1
        state._begin_selling()

        sale = configured.rule_set.action_codec.offset("sale_mode")
        done = configured.rule_set.action_codec.offset("done")
        cash_before = tuple(player.cash for player in state.players)
        holdings_before = tuple(
            tuple(player.regular_portfolio) for player in state.players
        )
        state.apply_action(sale)

        self.assertEqual(
            tuple(tuple(player.regular_portfolio) for player in state.players),
            holdings_before,
        )
        self.assertEqual(tuple(player.cash for player in state.players), cash_before)
        self.assertFalse(state.history_records)
        self.assertEqual(state._sequence, 0)
        owner, _ = _observe(configured, state, 0)
        opponent, _ = _observe(configured, state, 1)
        spectator, _ = _observe(configured, state, None)
        self.assertEqual(_member(owner, "owned_stocks")["regular"]["Cosmic Computers"], 1)
        self.assertTrue(
            any(
                record.get("stage") == "selling_commitment"
                for record in _plain(_member(owner, "observable_history"))
            )
        )
        for hidden_view in (opponent, spectator):
            self.assertFalse(
                any(
                    record.get("stage") == "selling_commitment"
                    for record in _plain(_member(hidden_view, "observable_history"))
                )
            )

        # Player 0 holds the remainder, then player 1 commits an all-sale and
        # holds every other company. Nothing public changes before the last
        # company commitment closes the sealed batch.
        for _ in range(6):
            state.apply_action(done)
        self.assertEqual(state.current_player(), 1)
        sell_all = configured.rule_set.action_codec.offset("sale_mode") + 2
        state.apply_action(sell_all)
        for _ in range(4):
            state.apply_action(done)
        self.assertEqual(tuple(player.cash for player in state.players), cash_before)
        self.assertEqual(
            tuple(tuple(player.regular_portfolio) for player in state.players),
            holdings_before,
        )
        self.assertFalse(state.history_records)
        state.apply_action(done)

        self.assertEqual(state.players[0].regular_portfolio[0], 1)
        self.assertEqual(state.players[1].regular_portfolio[0], 0)
        self.assertEqual(state.players[0].cash, cash_before[0] + 5)
        self.assertEqual(state.players[1].cash, cash_before[1] + 5)
        self.assertEqual(state._sequence, 1)
        self.assertEqual(len(state.history_records), 1)
        batch = state.history_records[0]
        self.assertEqual(batch["stage"], "selling_batch")
        self.assertEqual(batch["sales"], {0: {0: 1}, 1: {0: 1}})
        owner_after, _ = _observe(configured, state, 0)
        owner_stages = [
            record["stage"]
            for record in _plain(_member(owner_after, "observable_history"))
        ]
        self.assertLess(
            owner_stages.index("selling_commitment"),
            owner_stages.index("selling_batch"),
        )

    def test_sealed_sale_plans_merge_for_opponents_but_not_the_owner(self):
        configured = _configure(
            "lite",
            round_count=1,
        )

        def reach_player_one(*, sell: bool):
            state = configured.game.new_initial_state()
            state._chance_kind = ""
            state.first_player = 0
            state.players[0].regular_portfolio[0] = 1
            state._begin_selling()
            if sell:
                state.apply_action(
                    configured.rule_set.action_codec.offset("sale_mode")
                )
            done = configured.rule_set.action_codec.offset("done")
            for _ in range(6):
                state.apply_action(done)
            self.assertEqual(state.current_player(), 1)
            self.assertEqual(state._selling_company, 0)
            return state

        sold = reach_player_one(sell=True)
        held = reach_player_one(sell=False)
        self.assertEqual(
            sold.information_state_string(1),
            held.information_state_string(1),
        )
        self.assertEqual(
            tuple(sold.observation_tensor(1)),
            tuple(held.observation_tensor(1)),
        )
        self.assertEqual(
            sold._legal_actions(1),
            held._legal_actions(1),
        )
        self.assertNotEqual(
            sold.information_state_string(0),
            held.information_state_string(0),
        )

    def test_sealed_selling_preserves_the_legacy_all_or_nothing_option(self):
        configured = _configure(
            "lite",
            round_count=1,
            overrides={"partial_sales": False},
        )
        state = configured.game.new_initial_state()
        state._chance_kind = ""
        state.first_player = 0
        state.players[0].regular_portfolio[0] = 1
        state._begin_selling()
        done = configured.rule_set.action_codec.offset("done")
        sell_all = configured.rule_set.action_codec.offset("sell_all")
        self.assertEqual(state._legal_actions(0), sorted((done, sell_all)))

        state.apply_action(sell_all)
        self.assertEqual(state.players[0].regular_portfolio[0], 1)
        self.assertEqual(state.players[0].cash, 30)
        for _ in range(5 + 6):
            state.apply_action(done)
        self.assertEqual(state.players[0].regular_portfolio[0], 0)
        self.assertEqual(state.players[0].cash, 35)
        self.assertEqual(state.history_records[-1]["stage"], "selling_batch")

    def test_sealed_sale_clone_and_serialization_preserve_private_commitments(self):
        configured = _configure(
            "lite",
            round_count=1,
        )
        state = configured.game.new_initial_state()
        for _ in range(1_000):
            if state.is_chance_node():
                state.apply_action(state.chance_outcomes()[0][0])
            else:
                state.apply_action(min(state._legal_actions(state.current_player())))
            if (
                state.phase == "selling"
                and any(state._private_sale_history)
            ):
                break
        else:
            self.fail("playthrough did not reach a sealed sale commitment")

        for restored in (
            state.clone(),
            configured.game.deserialize_state(state.serialize()),
        ):
            self.assertEqual(restored.to_dict(), state.to_dict())
            self.assertEqual(
                restored._selling_shadow_regular,
                state._selling_shadow_regular,
            )
            self.assertEqual(
                restored._selling_shadow_split,
                state._selling_shadow_split,
            )
            self.assertEqual(
                restored._private_sale_history,
                state._private_sale_history,
            )
            self.assertEqual(
                restored._legal_actions(restored.current_player()),
                state._legal_actions(state.current_player()),
            )


class ObservationAndScoringTests(unittest.TestCase):
    def setUp(self):
        self.configured = _configure()
        _, self.state = _initial_state(self.configured, 2718)

    def test_observations_keep_private_information_and_holdings_secret(self):
        player_zero, actions_zero = stockpile.observe_game_state(
            self.configured.rule_set, self.state, 0
        )
        player_one, actions_one = stockpile.observe_game_state(
            self.configured.rule_set, self.state, 1
        )
        spectator, spectator_actions = stockpile.observe_game_state(
            self.configured.rule_set, self.state, None
        )
        del spectator_actions

        self.assertEqual(_nested_value(player_zero, "player_id"), 0)
        self.assertEqual(_nested_value(player_one, "player_id"), 1)
        self.assertIsNone(_nested_value(spectator, "player_id"))

        private_zero = _nested_value(player_zero, "private_information")
        private_one = _nested_value(player_one, "private_information")
        if private_zero is None:
            private_zero = _nested_value(player_zero, "private_information_pairs")
        if private_one is None:
            private_one = _nested_value(player_one, "private_information_pairs")
        self.assertTrue(private_zero)
        self.assertTrue(private_one)
        self.assertNotEqual(private_zero, private_one)
        self.assertFalse(
            _nested_value(spectator, "private_information", [])
            or _nested_value(spectator, "private_information_pairs", [])
        )

        legal_ids_zero = {_action_id(action) for action in actions_zero}
        observed_ids_zero = set(_nested_value(player_zero, "legal_action_ids", []))
        self.assertEqual(legal_ids_zero, observed_ids_zero)
        acting = self.state.current_player()
        self.assertEqual(bool(actions_zero), acting == 0)
        self.assertEqual(bool(actions_one), acting == 1)

    def test_lite_hypothetical_initial_score_is_a_cash_tie(self):
        result = stockpile.score_game(self.configured.rule_set, self.state)
        winners = set(_member(result, "winner_ids", "winners"))
        final_cash = _member(result, "final_cash_by_player", "final_cash")
        values = list(_plain(final_cash).values())
        self.assertEqual(values[0], values[1])
        self.assertEqual(winners, {0, 1})
        bonuses = _plain(_member(result, "bonuses", "majority_bonuses"))
        bonus_values = list(bonuses.values()) if isinstance(bonuses, Mapping) else list(bonuses)
        self.assertTrue(
            not bonus_values or all(int(amount) == 0 for amount in bonus_values),
            bonuses,
        )

    def test_endgame_receipts_settle_fee_debt_without_mutating_state(self):
        configured = _configure("classic", player_count=2, round_count=1)
        state = configured.game.new_initial_state()
        for player in state.players:
            player.cash = 0
            player.regular_portfolio = [0] * configured.rule_set.company_count
            player.split_portfolio = [0] * configured.rule_set.company_count
            player.fees.clear()
        state.players[0].regular_portfolio[0] = 1
        state.players[0].fees = [12]

        result = stockpile.score_game(configured.rule_set, state)

        self.assertEqual(result.liquidation_values[0], 5)
        self.assertEqual(result.bonuses[0], 10)
        self.assertEqual(result.final_cash_by_player[0], 3)
        self.assertEqual(state.players[0].cash, 0)
        self.assertEqual(state.players[0].fees, [12])

    def test_endgame_fee_debts_remain_fifo_when_the_first_is_unaffordable(self):
        configured = _configure("lite", player_count=2, round_count=1)
        state = configured.game.new_initial_state()
        for player in state.players:
            player.cash = 0
            player.regular_portfolio = [0] * configured.rule_set.company_count
            player.split_portfolio = [0] * configured.rule_set.company_count
        state.players[0].regular_portfolio[0] = 2
        state.players[0].fees = [3, 5, 4]
        state.players[1].regular_portfolio[0] = 1
        state.players[1].fees = [6, 1]

        result = stockpile.score_game(configured.rule_set, state)

        self.assertEqual(result.liquidation_values, {0: 10, 1: 5})
        self.assertEqual(result.final_cash_by_player, {0: 2, 1: 5})
        self.assertEqual(state.players[0].fees, [3, 5, 4])
        self.assertEqual(state.players[1].fees, [6, 1])

    def test_terminal_result_has_bounded_zero_sum_utilities(self):
        terminal, _ = _play_first_legal(self.configured, seed=811)
        result = stockpile.score_game(self.configured.rule_set, terminal)
        utilities_value = _plain(_member(result, "utilities"))
        utilities = (
            list(utilities_value.values())
            if isinstance(utilities_value, Mapping)
            else list(utilities_value)
        )
        self.assertEqual(len(utilities), 2)
        self.assertAlmostEqual(sum(float(value) for value in utilities), 0.0)
        self.assertTrue(all(-1.0 <= float(value) <= 1.0 for value in utilities))


class OpenSpielAndWrapperTests(unittest.TestCase):
    def test_game_and_state_expose_open_spiel_contract(self):
        configured = _configure()
        initial = stockpile.randomize_initial_input(configured.rule_set, 41)
        state, report = stockpile.initialize_game(
            configured.rule_set, configured.game, initial
        )
        self.assertTrue(_report_is_valid(report))

        for name in (
            "current_player",
            "legal_actions",
            "apply_action",
            "clone",
            "is_terminal",
            "returns",
            "information_state_string",
            "observation_string",
            "observation_tensor",
        ):
            self.assertTrue(callable(getattr(state, name, None)), name)

        actor = state.current_player()
        _, actions = stockpile.observe_game_state(configured.rule_set, state, actor)
        self.assertEqual(
            set(state.legal_actions(actor)),
            {_action_id(action) for action in actions},
        )
        clone = state.clone()
        clone.apply_action(min(state.legal_actions(actor)))
        self.assertNotEqual(_fingerprint(clone), _fingerprint(state))
        self.assertTrue(state.information_state_string(actor))
        self.assertGreater(len(state.observation_tensor(actor)), 0)

    def test_functional_new_initial_state_wrapper(self):
        configured = _configure()
        state = configured.game.new_initial_state()
        self.assertFalse(state.is_terminal())
        self.assertTrue(state.legal_actions(state.current_player()))

    def test_complexity_reports_match_the_stable_action_catalog(self):
        lite = stockpile.complexity_report(
            stockpile.GameParameters(player_count=2, rules_profile="lite")
        )
        self.assertEqual(lite["num_distinct_actions"], 18)
        self.assertEqual(lite["max_legal_actions"], 8)
        self.assertEqual(lite["max_chance_outcomes"], 6)
        self.assertEqual(lite["shared_action_head"], 29)
        self.assertGreater(lite["max_game_length"], 0)
        self.assertGreater(lite["observation_size"], 0)

        classic_sizes = {2: 28, 3: 27, 4: 28, 5: 29}
        for player_count, expected_size in classic_sizes.items():
            with self.subTest(profile="classic", player_count=player_count):
                report = stockpile.complexity_report(
                    stockpile.GameParameters(
                        player_count=player_count,
                        rules_profile="classic",
                    )
                )
                self.assertEqual(report["num_distinct_actions"], expected_size)
                self.assertEqual(report["max_legal_actions"], 8)
                self.assertEqual(report["max_chance_outcomes"], 11)
                self.assertEqual(report["shared_action_head"], 29)

        deluxe_sizes = {2: 41, 3: 36, 4: 38, 5: 40}
        for player_count, expected_size in deluxe_sizes.items():
            with self.subTest(profile="deluxe", player_count=player_count):
                report = stockpile.complexity_report(
                    stockpile.GameParameters(
                        player_count=player_count,
                        rules_profile="deluxe",
                    )
                )
                self.assertEqual(report["num_distinct_actions"], expected_size)
                self.assertEqual(report["max_legal_actions"], 8)
                self.assertEqual(report["max_chance_outcomes"], 11)
                self.assertEqual(report["shared_action_head"], 42)


if __name__ == "__main__":
    unittest.main()
