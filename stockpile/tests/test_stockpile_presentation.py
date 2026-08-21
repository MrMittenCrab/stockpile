"""Presentation-only engine contracts for the local browser interface."""

from __future__ import annotations

from dataclasses import asdict
import unittest

import stockpile


def _configured(*, rounds: int = 1, impact: bool = False):
    return stockpile.configure_game(
        stockpile.GameParameters(
            player_count=2,
            rules_profile="lite",
            round_count=rounds,
            rule_overrides={"impact": impact},
        )
    )


def _play_lowest_to_terminal(configured, seed: int = 19):
    initial = stockpile.randomize_initial_input(configured.rule_set, seed)
    state, report = stockpile.initialize_game(
        configured.rule_set,
        configured.game,
        initial,
    )
    if not report.valid:
        raise AssertionError(report.errors)
    for _ in range(2_000):
        if state.is_terminal():
            return state
        if state.is_chance_node():
            state.apply_action(min(action for action, _ in state.chance_outcomes()))
        else:
            state.apply_action(min(state.legal_actions()))
    raise AssertionError("playthrough did not terminate")


class PresentationContractTests(unittest.TestCase):
    def test_lite_price_above_ten_emits_display_event_without_game_sequence(self):
        configured = _configured()
        state = configured.game.new_initial_state()
        state._set_company_price(0, 10)
        sequence_before = state._sequence

        state._move_price(
            0,
            4,
            cause="market_forecast",
            forecast=4,
        )

        self.assertEqual(state._company_price(0), 14)
        self.assertEqual(state._sequence, sequence_before)
        events = stockpile.get_presentation_events(configured.rule_set, state)
        self.assertEqual(len(events), 1)
        self.assertEqual(
            asdict(events[0]),
            {
                "presentation_sequence": 1,
                "round": 1,
                "event_type": "market_movement",
                "cause": "market_forecast",
                "company_id": 0,
                "company_name": "Cosmic Computers",
                "prior_price": 10,
                "requested_delta": 4,
                "actual_delta": 4,
                "resulting_price": 14,
                "forecast": 4,
                "effect": None,
                "actor_id": None,
                "description": "Cosmic Computers moved +4 to $14K",
            },
        )
        self.assertEqual(
            stockpile.get_presentation_events(
                configured.rule_set,
                state,
                since_sequence=1,
            ),
            (),
        )

    def test_staged_context_and_bid_marker_identity_are_viewer_safe(self):
        configured = _configured()
        state = configured.game.new_initial_state()
        state._chance_kind = ""
        state.phase = stockpile.Phase.DEMAND.value
        state.stage = "demand_bid"
        state.current_actor = 0
        state._demand_token = (0, 1)
        state._demand_pile = 2
        pile = state.stockpiles[0]
        pile.occupying_player = 1
        pile.occupying_token = 0
        pile.bid_level = 2

        owner = stockpile.get_presentation_state(configured.rule_set, state, 0)
        opponent = stockpile.get_presentation_state(configured.rule_set, state, 1)
        self.assertEqual(owner.demand_token, (0, 1))
        self.assertEqual(owner.demand_pile, 2)
        self.assertEqual(opponent.demand_token, (0, 1))
        self.assertIsNone(opponent.demand_pile)
        self.assertEqual(
            asdict(owner.stockpile_markers[0]),
            {
                "stockpile_id": 0,
                "player_id": 1,
                "marker_index": 0,
                "bid_value": 3,
                "status": "leading",
            },
        )

        state.phase = stockpile.Phase.SUPPLY.value
        state.stage = "supply_down_pile"
        state._demand_token = None
        state._demand_pile = None
        state._supply_choice = 1
        state._supply_up_pile = 3
        owner = stockpile.get_presentation_state(configured.rule_set, state, 0)
        opponent = stockpile.get_presentation_state(configured.rule_set, state, 1)
        self.assertEqual((owner.supply_choice, owner.supply_up_pile), (1, 3))
        self.assertEqual(
            (opponent.supply_choice, opponent.supply_up_pile),
            (None, None),
        )

    def test_sealed_selling_redacts_actor_and_company_from_other_views(self):
        configured = _configured()
        state = configured.game.new_initial_state()
        state._chance_kind = ""
        state.first_player = 0
        state.players[0].regular_portfolio[0] = 1
        state._begin_selling()

        owner = stockpile.get_presentation_state(configured.rule_set, state, 0)
        opponent = stockpile.get_presentation_state(configured.rule_set, state, 1)
        spectator = stockpile.get_presentation_state(configured.rule_set, state)
        self.assertEqual(owner.current_actor, 0)
        self.assertEqual(owner.stage, "selling")
        self.assertEqual(owner.selling_company, 0)
        for hidden in (opponent, spectator):
            self.assertIsNone(hidden.current_actor)
            self.assertEqual(hidden.stage, "private_selling")
            self.assertIsNone(hidden.selling_company)

    def test_sale_previews_use_shadow_holdings_and_do_not_mutate(self):
        configured = _configured()
        state = configured.game.new_initial_state()
        state._chance_kind = ""
        state.first_player = 0
        state.players[0].regular_portfolio[0] = 2
        state.players[0].split_portfolio[0] = 1
        state._begin_selling()
        before = str(state)
        sale = configured.rule_set.action_codec.offset("sale_mode")

        one = stockpile.preview_sale_action(
            configured.rule_set,
            state,
            0,
            sale,
        )
        all_shares = stockpile.preview_sale_action(
            configured.rule_set,
            state,
            0,
            sale + 2,
        )

        self.assertEqual(one.quantity_sold, 1)
        self.assertEqual(one.gross_value, 5)
        self.assertEqual(
            (one.resulting_regular, one.resulting_split, one.resulting_represented),
            (1, 1, 3),
        )
        self.assertEqual(all_shares.quantity_sold, 4)
        self.assertEqual(all_shares.gross_value, 20)
        self.assertEqual(all_shares.resulting_represented, 0)
        self.assertEqual(str(state), before)

    def test_presentation_journal_replays_through_clone(self):
        configured = _configured()
        state = _play_lowest_to_terminal(configured)
        events = stockpile.get_presentation_events(configured.rule_set, state)
        self.assertTrue(events)

        clone = state.clone()

        self.assertEqual(
            stockpile.get_presentation_events(configured.rule_set, clone),
            events,
        )
        self.assertEqual(clone._presentation_sequence, len(events))

    def test_presentation_storage_does_not_change_observation_semantics(self):
        configured = _configured()
        initial = stockpile.randomize_initial_input(configured.rule_set, 31)
        state, report = stockpile.initialize_game(
            configured.rule_set,
            configured.game,
            initial,
        )
        self.assertTrue(report.valid)
        while state.is_chance_node():
            state.apply_action(state.chance_outcomes()[0][0])
        actor = state.current_player()
        before_information, before_actions = stockpile.observe_game_state(
            configured.rule_set,
            state,
            actor,
        )
        before_tensor = tuple(state.observation_tensor(actor))
        before_sequence = state._sequence
        before_state = str(state)
        before_history = tuple(state.history())

        price = state._company_price(0)
        state._record_presentation_market_event(
            event_type="market_reveal",
            cause="test",
            company=0,
            prior_price=price,
            requested_delta=None,
            resulting_price=price,
            description="Presentation-only test event",
        )

        after_information, after_actions = stockpile.observe_game_state(
            configured.rule_set,
            state,
            actor,
        )
        self.assertEqual(
            after_information.information_state_id,
            before_information.information_state_id,
        )
        self.assertEqual(tuple(state.observation_tensor(actor)), before_tensor)
        self.assertEqual(after_information.tensor, before_information.tensor)
        self.assertEqual(after_information.legal_action_ids, before_information.legal_action_ids)
        self.assertEqual(
            tuple(action.action_id for action in after_actions),
            tuple(action.action_id for action in before_actions),
        )
        self.assertEqual(state._sequence, before_sequence)
        self.assertEqual(tuple(state.history()), before_history)
        self.assertEqual(str(state), before_state)

    def test_terminal_liquidation_has_tied_ranks_and_company_lines(self):
        configured = _configured()
        state = configured.game.new_initial_state()
        state._chance_kind = ""
        state.phase = stockpile.Phase.TERMINAL.value
        state.stage = "terminal"
        state.terminal_status = True
        state.players[0].regular_portfolio[0] = 2
        state.players[1].regular_portfolio[0] = 2

        details = stockpile.terminal_liquidation_details(
            configured.rule_set,
            state,
        )

        self.assertEqual([row.rank for row in details], [1, 1])
        self.assertTrue(all(row.winner for row in details))
        self.assertEqual(details[0].companies[0].represented_shares, 2)
        self.assertEqual(details[0].companies[0].value, 10)
        self.assertEqual(details[0].liquidation_value, 10)
        self.assertEqual(details[0].final_cash, 40)


if __name__ == "__main__":
    unittest.main()
