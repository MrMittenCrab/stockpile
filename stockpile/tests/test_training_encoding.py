"""Perfect-recall and privacy tests for the Torch-free training encoder."""

from __future__ import annotations

import unittest

import numpy as np

import stockpile
from stockpile.training.encoding import (
    ACTION_COUNT,
    CURRENT_FEATURE_SIZE,
    EMPTY_TRACE,
    EVENT_BATCH_SALES_OFFSET,
    EVENT_DEMAND_BID_AMOUNT_INDEX,
    EVENT_DEMAND_BID_LEVEL_INDEX,
    EVENT_DEMAND_PILE_INDEX,
    EVENT_FEATURE_SIZE,
    EVENT_SALE_COMPANY_INDEX,
    EVENT_SALE_MODE_OFFSET,
    EVENT_SUPPLY_CARD_PRESENT_INDEX,
    EVENT_SUPPLY_COMPANY_INDEX,
    EVENT_SUPPLY_FACE_DOWN_PILE_INDEX,
    EVENT_SUPPLY_FACE_UP_PILE_INDEX,
    HISTORY_FEATURE_SIZE,
    TraceSession,
    batch_information_inputs,
    encode_visible_event,
    reconstruct_information_input,
    reconstruct_trace,
)


def _configured(*, rounds: int = 1, sell_order: bool = False):
    return stockpile.configure_game(
        stockpile.GameParameters(
            player_count=2,
            rules_profile="lite",
            round_count=rounds,
            deluxe_investors=False,
            board_side="standard",
            investor_mode="none",
            action_space_mode="compact",
            rule_overrides={"sell_order": sell_order},
        )
    )


def _preset_state(configured, seed: int = 19):
    initial = stockpile.randomize_initial_input(configured.rule_set, seed)
    state, report = stockpile.initialize_game(
        configured.rule_set,
        configured.game,
        initial,
    )
    if not report.valid:
        raise AssertionError(report.errors)
    return state


class TrainingEncodingTests(unittest.TestCase):
    def test_empty_and_prefix_shared_history_batch_shapes(self):
        configured = _configured()
        state = configured.game.new_initial_state()
        state._chance_kind = ""
        state.first_player = 0
        state._begin_selling()
        session = TraceSession(configured.game, 0)

        initial = session.snapshot(state)
        empty_batch = batch_information_inputs([initial])
        self.assertEqual(empty_batch["current"].shape, (1, CURRENT_FEATURE_SIZE))
        self.assertEqual(empty_batch["history"].shape, (1, 0, HISTORY_FEATURE_SIZE))
        self.assertEqual(empty_batch["events"].shape, (1, 0, EVENT_FEATURE_SIZE))
        self.assertEqual(empty_batch["history_mask"].shape, (1, 0))
        self.assertEqual(empty_batch["event_mask"].shape, (1, 0))
        self.assertEqual(empty_batch["legal_mask"].shape, (1, ACTION_COUNT))
        self.assertEqual(initial.horizon_features, (0.1, 0.1, 0.1))

        # A company with no holdings has one legal action.  It is still part of
        # perfect recall even though traversal code will skip network inference.
        done = configured.rule_set.action_codec.offset("done")
        first = session.record_action(state, done)
        self.assertIsNotNone(first.step)
        self.assertTrue(first.steps()[-1].forced)
        state.apply_action(done)
        second = session.record_action(state, done)
        self.assertIsNotNone(second.step)
        self.assertTrue(second.steps()[-1].forced)
        self.assertIs(second.parent, first)
        self.assertIs(first.parent, EMPTY_TRACE)
        self.assertEqual(second.length, 2)
        state.apply_action(done)

        current = session.snapshot(state)
        populated = batch_information_inputs([initial, current])
        self.assertEqual(populated["current"].shape, (2, CURRENT_FEATURE_SIZE))
        self.assertEqual(populated["history"].shape, (2, 2, HISTORY_FEATURE_SIZE))
        self.assertEqual(populated["history_lengths"].tolist(), [0, 2])
        self.assertEqual(populated["history_mask"].tolist(), [[False, False], [True, True]])
        self.assertEqual(populated["events"].shape[0], 2)
        self.assertEqual(populated["events"].shape[2], EVENT_FEATURE_SIZE)
        self.assertEqual(np.count_nonzero(populated["history"][1, 0, -ACTION_COUNT:]), 1)

        entirely_empty = batch_information_inputs([])
        self.assertEqual(entirely_empty["current"].shape, (0, CURRENT_FEATURE_SIZE))
        self.assertEqual(entirely_empty["history"].shape, (0, 0, HISTORY_FEATURE_SIZE))
        self.assertEqual(entirely_empty["events"].shape, (0, 0, EVENT_FEATURE_SIZE))
        self.assertEqual(entirely_empty["legal_mask"].shape, (0, ACTION_COUNT))

    def test_sealed_sale_is_absent_from_later_player_trace_but_in_owner_trace(self):
        configured = _configured()
        done = configured.rule_set.action_codec.offset("done")
        sell_regular = configured.rule_set.action_codec.offset("sale_mode")

        def reach_player_one(sell: bool):
            state = configured.game.new_initial_state()
            state._chance_kind = ""
            state.first_player = 0
            state.players[0].regular_portfolio[0] = 1
            state._begin_selling()
            owner = TraceSession(configured.game, 0)
            later = TraceSession(configured.game, 1)
            if sell:
                owner.record_action(state, sell_regular)
                state.apply_action(sell_regular)
            for _company in range(6):
                owner.record_action(state, done)
                state.apply_action(done)
            self.assertEqual(state.current_player(), 1)
            return state, owner, later.snapshot(state)

        sold_state, sold_owner, sold_later = reach_player_one(True)
        held_state, held_owner, held_later = reach_player_one(False)

        self.assertEqual(sold_later.current_observation, held_later.current_observation)
        self.assertEqual(sold_later.information_state_id, held_later.information_state_id)
        self.assertEqual(sold_later.perfect_recall_id, held_later.perfect_recall_id)
        self.assertEqual(sold_later.legal_mask, held_later.legal_mask)
        self.assertEqual(
            tuple(event.payload_json for event in sold_later.visible_event_features),
            tuple(event.payload_json for event in held_later.visible_event_features),
        )
        self.assertFalse(
            any(
                event.kind == "selling_commitment"
                for event in sold_later.visible_event_features
            )
        )
        self.assertNotEqual(sold_owner.trace.digest, held_owner.trace.digest)
        self.assertNotEqual(sold_owner.trace.steps(), held_owner.trace.steps())

        # The public state also remains unchanged until both plans settle.
        self.assertEqual(
            tuple(player.cash for player in sold_state.players),
            tuple(player.cash for player in held_state.players),
        )

    def test_incremental_and_replayed_inputs_are_identical(self):
        configured = _configured(rounds=2)
        state = _preset_state(configured)
        sessions = {
            player: TraceSession(configured.game, player)
            for player in range(2)
        }

        for _ in range(24):
            self.assertFalse(state.is_terminal())
            self.assertFalse(state.is_chance_node())
            actor = int(state.current_player())
            legal = tuple(int(action) for action in state.legal_actions(actor))
            action = min(legal)
            sessions[actor].snapshot(state)
            sessions[actor].record_action(state, action, forced=len(legal) == 1)
            state.apply_action(action)

        self.assertFalse(state.is_chance_node())
        actor = int(state.current_player())
        incremental = sessions[actor].snapshot(state)
        replayed = reconstruct_information_input(state, actor)
        replayed_trace = reconstruct_trace(state, actor)

        self.assertEqual(incremental.current_observation, replayed.current_observation)
        self.assertEqual(incremental.horizon_features, replayed.horizon_features)
        self.assertEqual(incremental.information_state_id, replayed.information_state_id)
        self.assertEqual(incremental.perfect_recall_id, replayed.perfect_recall_id)
        self.assertEqual(incremental.legal_mask, replayed.legal_mask)
        self.assertEqual(incremental.trace.digest, replayed.trace.digest)
        self.assertEqual(incremental.trace.steps(), replayed.trace.steps())
        self.assertEqual(replayed_trace.digest, replayed.trace.digest)
        self.assertEqual(
            tuple(event.payload_json for event in incremental.visible_event_features),
            tuple(event.payload_json for event in replayed.visible_event_features),
        )

    def test_repeated_same_state_snapshot_preserves_event_delta_and_replay(self):
        configured = _configured()
        state = _preset_state(configured)
        player = int(state.current_player())
        session = TraceSession(configured.game, player)

        first_action = min(int(action) for action in state.legal_actions(player))
        session.record_action(state, first_action)
        state.apply_action(first_action)
        self.assertEqual(state.current_player(), player)

        first_query = session.snapshot(state, player)
        repeated_query = session.snapshot(state, player)
        self.assertEqual(first_query, repeated_query)
        self.assertGreater(len(first_query.visible_event_features), 0)

        second_action = min(int(action) for action in state.legal_actions(player))
        session.record_action(state, second_action)
        self.assertGreater(len(session.trace.steps()[-1].new_visible_events), 0)
        state.apply_action(second_action)

        replayed = reconstruct_trace(state, player)
        self.assertEqual(session.trace.digest, replayed.digest)
        self.assertEqual(session.trace.steps(), replayed.steps())

    def test_visible_event_schema_covers_lite_supply_demand_and_sales(self):
        rules = _configured().rule_set
        supply = encode_visible_event(
            {
                "sequence": 1,
                "player": 0,
                "phase": "supply",
                "stage": "supply_commit",
                "public_supply_commit": {
                    "player": 0,
                    "face_up_pile": 2,
                    "face_down_pile": 3,
                    "face_up_card": {
                        "card_id": 123,
                        "card_type": "stock",
                        "company_id": 4,
                        "value": 1,
                    },
                    "both_face_down": False,
                },
            },
            rules,
        )
        self.assertEqual(supply.kind, "supply_public")
        self.assertGreater(supply.features[EVENT_SUPPLY_FACE_UP_PILE_INDEX], 0.0)
        self.assertGreater(supply.features[EVENT_SUPPLY_FACE_DOWN_PILE_INDEX], 0.0)
        self.assertEqual(supply.features[EVENT_SUPPLY_CARD_PRESENT_INDEX], 1.0)
        self.assertGreater(supply.features[EVENT_SUPPLY_COMPANY_INDEX], 0.0)

        pile_action = rules.action_codec.offset("pile") + 2
        demand_pile = encode_visible_event(
            {
                "sequence": 2,
                "player": 1,
                "phase": "demand",
                "stage": "demand_pile",
                "action": pile_action,
            },
            rules,
        )
        self.assertEqual(demand_pile.kind, "demand_pile")
        self.assertGreater(demand_pile.features[EVENT_DEMAND_PILE_INDEX], 0.0)

        bid_action = rules.action_codec.offset("bid_level") + 3
        demand_bid = encode_visible_event(
            {
                "sequence": 3,
                "player": 1,
                "phase": "demand",
                "stage": "demand_bid",
                "action": bid_action,
            },
            rules,
        )
        self.assertEqual(demand_bid.kind, "demand_bid")
        self.assertGreater(demand_bid.features[EVENT_DEMAND_BID_LEVEL_INDEX], 0.0)
        self.assertGreater(demand_bid.features[EVENT_DEMAND_BID_AMOUNT_INDEX], 0.0)

        commitment = encode_visible_event(
            {
                "private_sequence": 1,
                "after_public_sequence": 3,
                "round": 1,
                "player": 0,
                "phase": "selling",
                "stage": "selling_commitment",
                "company": 2,
                "action": rules.action_codec.offset("sale_mode") + 2,
                "action_type": "sale_mode",
                "ordinal": 2,
            },
            rules,
        )
        self.assertEqual(commitment.kind, "selling_commitment")
        self.assertGreater(commitment.features[EVENT_SALE_COMPANY_INDEX], 0.0)
        self.assertEqual(commitment.features[EVENT_SALE_MODE_OFFSET + 3], 1.0)

        batch = encode_visible_event(
            {
                "sequence": 4,
                "player": -1,
                "phase": "selling",
                "stage": "selling_batch",
                "action": -1,
                "sales": {0: {2: 3}, 1: {4: 1}},
            },
            rules,
        )
        self.assertEqual(batch.kind, "selling_batch")
        self.assertAlmostEqual(
            batch.features[EVENT_BATCH_SALES_OFFSET + 2],
            0.3,
            places=6,
        )
        self.assertAlmostEqual(
            batch.features[EVENT_BATCH_SALES_OFFSET + 6 + 4],
            0.1,
            places=6,
        )

    def test_encoder_rejects_observable_sequential_selling(self):
        configured = _configured(sell_order=True)
        with self.assertRaisesRegex(ValueError, "sell_order"):
            TraceSession(configured.game, 0)


if __name__ == "__main__":
    unittest.main()
