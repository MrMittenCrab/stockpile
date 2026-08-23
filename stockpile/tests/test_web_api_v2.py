"""Contract and playthrough tests for the human-versus-computer web API."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import random
import unittest

from stockpile import stockpile_platform as platform
from stockpile.web.policy import RandomComputerPolicy

try:
    from fastapi.testclient import TestClient
    from stockpile.web.app import create_app
    from stockpile.web.sessions import SessionError
    from stockpile.web.v2_sessions import V2SessionStore
except ImportError:  # Core-only installations intentionally omit web extras.
    TestClient = None  # type: ignore[assignment,misc]
    create_app = None  # type: ignore[assignment]
    SessionError = None  # type: ignore[assignment,misc]
    V2SessionStore = None  # type: ignore[assignment,misc]


def _apply_for_test(session, state, player_id: int, action_id: int):  # type: ignore[no-untyped-def]
    _information, legal = platform.observe_game_state(
        session.rule_set, state, player_id
    )
    next_state, _record, report = platform.advance_game(
        session.rule_set,
        state,
        legal,
        platform.ActionRequest(player_id=player_id, action_id=action_id),
    )
    if not report.valid:
        raise AssertionError(report.errors)
    return next_state


class RandomComputerPolicyTest(unittest.TestCase):
    def test_seeded_policy_is_reproducible_uniform_and_strictly_legal(self) -> None:
        actions = [
            platform.LegalAction(
                action_id=action_id,
                phase="test",
                actor_ids=(1,),
                action_type="test",
            )
            for action_id in (7, 19, 41, 83)
        ]
        policy = RandomComputerPolicy()

        def sample(seed: int) -> list[int]:
            rng = random.Random(seed)
            return [
                policy.choose_action(None, None, actions, rng)  # type: ignore[arg-type]
                for _ in range(4_000)
            ]

        first = sample(20260822)
        self.assertEqual(first, sample(20260822))
        self.assertTrue(set(first).issubset({7, 19, 41, 83}))
        counts = Counter(first)
        self.assertEqual(set(counts), {7, 19, 41, 83})
        # This catches a biased placeholder while allowing deterministic random
        # variation around the expected quarter of the samples.
        self.assertTrue(all(850 <= count <= 1_150 for count in counts.values()))

    def test_policy_rejects_an_empty_legal_action_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one legal action"):
            RandomComputerPolicy().choose_action(  # type: ignore[arg-type]
                None, None, (), random.Random(1)
            )




class DeepCFRComputerPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from stockpile.web.policy import (
                DeepCFRComputerPolicy,
                resolve_computer_policy_path,
            )
            cls.checkpoint = resolve_computer_policy_path()
            cls.policy = DeepCFRComputerPolicy.load(cls.checkpoint)
        except Exception as error:  # Torch or artifacts may be absent.
            cls.checkpoint = None
            cls.policy = None
            cls.skip_reason = str(error)
        else:
            cls.skip_reason = ""

    def setUp(self) -> None:
        if self.policy is None:
            self.skipTest(self.skip_reason or "Deep CFR computer policy unavailable")

    def test_loaded_policy_samples_only_legal_actions(self) -> None:
        from stockpile.stockpile_interface import ConfigurationMode, resolve_configuration
        from stockpile.web.policy import RandomComputerPolicy

        configuration = resolve_configuration(
            ConfigurationMode.LITE,
            player_count=2,
            round_count=2,
        )
        state = platform.GameState(configuration.game)
        rng = random.Random(11)
        filler = RandomComputerPolicy()
        for _ in range(40):
            if state.is_terminal():
                break
            if state.is_chance_node():
                state.apply_action(state.chance_outcomes()[0][0])
                continue
            player = int(state.current_player())
            information, legal = platform.observe_game_state(
                configuration.rule_set, state, player
            )
            if player == 1:
                action_id = self.policy.choose_action(
                    state, information, legal, rng
                )
            else:
                action_id = filler.choose_action(state, information, legal, rng)
            self.assertIn(action_id, {action.action_id for action in legal})
            state.apply_action(action_id)


@unittest.skipIf(TestClient is None, "requirements-web.txt is not installed")
class WebApiV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.store = V2SessionStore()
        self.client = TestClient(create_app(store_v2=self.store))

    def create_game(
        self,
        *,
        options: dict[str, bool] | None = None,
        seed: int = 101,
    ) -> tuple[str, str, dict]:
        response = self.client.post(
            "/api/v2/games",
            json={"options": options or {}, "seed": seed},
        )
        self.assertEqual(response.status_code, 201, response.text)
        payload = response.json()
        token = payload["game_url"].split("#seat=", 1)[1]
        return payload["game_id"], token, payload

    def view_response(self, game_id: str, token: str):
        return self.client.get(
            f"/api/v2/games/{game_id}/view",
            headers={"Authorization": f"Bearer {token}"},
        )

    def view(self, game_id: str, token: str) -> dict:
        response = self.view_response(game_id, token)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def submit_action(self, game_id: str, token: str, view: dict, action_id: int):
        return self.client.post(
            f"/api/v2/games/{game_id}/actions",
            headers={"Authorization": f"Bearer {token}"},
            json={"action_id": action_id, "expected_revision": view["revision"]},
        )

    def submit_supply(self, game_id: str, token: str, view: dict, plan_id: str):
        return self.client.post(
            f"/api/v2/games/{game_id}/supply",
            headers={"Authorization": f"Bearer {token}"},
            json={"plan_id": plan_id, "expected_revision": view["revision"]},
        )

    def submit_decision(self, game_id: str, token: str, view: dict, plan_id: str):
        return self.client.post(
            f"/api/v2/games/{game_id}/decisions",
            headers={"Authorization": f"Bearer {token}"},
            json={"plan_id": plan_id, "expected_revision": view["revision"]},
        )

    def acknowledge(self, game_id: str, token: str, view: dict):
        checkpoint = view["checkpoint"]
        self.assertIsNotNone(checkpoint)
        return self.client.post(
            f"/api/v2/games/{game_id}/acknowledgements",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "checkpoint_id": checkpoint["checkpoint_id"],
                "expected_revision": view["revision"],
            },
        )

    def advance_once(self, game_id: str, token: str, view: dict) -> dict:
        if view["checkpoint"] is not None:
            response = self.acknowledge(game_id, token, view)
        elif view["supply_batch"] is not None:
            response = self.submit_supply(
                game_id,
                token,
                view,
                view["supply_batch"]["plans"][0]["plan_id"],
            )
        elif view["decision_batch"] is not None:
            response = self.submit_decision(
                game_id,
                token,
                view,
                view["decision_batch"]["plans"][0]["plan_id"],
            )
        elif view["legal_actions"]:
            response = self.submit_action(
                game_id, token, view, view["legal_actions"][0]["action_id"]
            )
        else:
            self.fail(
                "non-terminal V2 view offered no human action, atomic plan, or "
                f"checkpoint: {view['phase']} / {view['phase_step']}"
            )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def drive_until(
        self,
        game_id: str,
        token: str,
        predicate,  # type: ignore[no-untyped-def]
        *,
        limit: int = 2_000,
    ) -> dict:
        current = self.view(game_id, token)
        for _step in range(limit):
            if predicate(current):
                return current
            if current["terminal_results"] is not None:
                break
            current = self.advance_once(game_id, token, current)
        self.fail(f"V2 game did not reach requested state within {limit} actions")

    @staticmethod
    def _engine_snapshot(session) -> tuple:  # type: ignore[no-untyped-def]
        state = session.state
        return (
            state.round,
            state.phase,
            state.stage,
            int(state.current_player()),
            tuple(state.prices),
            tuple(player.cash for player in state.players),
            tuple(
                tuple(card.card_id for card in hand)
                for hand in state._hands
            ),
            tuple(
                (
                    tuple(card.card_id for card in pile.face_up_cards),
                    tuple(card.card_id for card in pile.face_down_cards),
                    pile.bid_level,
                    pile.occupying_player,
                    pile.occupying_token,
                    pile.locked,
                    pile.purchaser,
                )
                for pile in state.stockpiles
            ),
            deepcopy(state.history_records),
            tuple(
                tuple((entry.card_id, entry.face_up) for entry in cards)
                for cards in session.ordered_piles.values()
            ),
            session.chance_rng.getstate(),
            session.policy_rng.getstate(),
            session.demand_cash_before,
            session.demand_position_before,
            tuple(session.latest_cash_deltas),
            session.latest_human_position_delta,
            tuple(sorted(session.latest_market_deltas.items())),
            session.latest_presentation_event_sequence,
            session.latest_automatic_event_sequence,
            tuple(
                (
                    company_id,
                    snapshot.round,
                    snapshot.company_id,
                    snapshot.prior_price,
                    snapshot.human_shares,
                    snapshot.presentation_sequence,
                )
                for company_id, snapshot in sorted(
                    session.bankruptcy_snapshots.items()
                )
            ),
            tuple(
                (
                    sequence,
                    snapshot.round,
                    snapshot.company_id,
                    snapshot.prior_price,
                    snapshot.human_shares,
                    snapshot.presentation_sequence,
                )
                for sequence, snapshot in sorted(
                    session.bankruptcy_event_corrections.items()
                )
            ),
        )

    def natural_bankruptcy_checkpoint(
        self,
        game_id: str,
        token: str,
        *,
        round_number: int,
        company_id: int,
    ) -> tuple[dict, object]:
        """Reach a clone-stable bankruptcy produced only by public API actions."""

        result = self.drive_until(
            game_id,
            token,
            lambda item: item["checkpoint"] is not None
            and item["checkpoint"]["kind"] == "round_result"
            and item["checkpoint"]["round"] == round_number,
            limit=4_000,
        )
        session = self.store.get(game_id)
        self.assertIn(
            company_id,
            session.bankruptcy_snapshots,
            "deterministic seed no longer reaches the expected real bankruptcy",
        )
        return result, session

    def test_setup_and_creation_are_fixed_private_and_no_store(self) -> None:
        setup_response = self.client.get("/api/v2/setup")
        self.assertEqual(setup_response.status_code, 200)
        self.assertEqual(setup_response.headers["cache-control"], "no-store")
        setup = setup_response.json()
        self.assertEqual(setup["schema_version"], "2.0")
        self.assertEqual(setup["mode"], "lite")
        self.assertEqual(setup["round_count"], 2)
        self.assertEqual(
            [(item["key"], item["default"]) for item in setup["options"]],
            [
                ("dividends", False),
                ("trading_fees", False),
                ("sell_order", False),
            ],
        )

        game_id, token, created = self.create_game(seed=17)
        self.assertEqual(set(created), {"schema_version", "game_id", "game_url"})
        self.assertNotIn("seed", created)
        self.assertNotIn("seats", created)
        self.assertIn("#seat=", created["game_url"])
        view_response = self.view_response(game_id, token)
        view = view_response.json()
        self.assertEqual(view["configuration"]["player_count"], 2)
        self.assertEqual(view["configuration"]["round_count"], 2)
        self.assertFalse(view["configuration"]["options"].get("starting_share", False))
        self.assertEqual(view["viewer"], {"player_id": 0, "name": "YOU"})
        self.assertEqual([player["name"] for player in view["players"]], ["YOU", "COMPUTER"])
        self.assertEqual(view["players"][0]["role"], "human")
        self.assertEqual(view["players"][1]["role"], "computer")
        self.assertNotIn("position_value_thousands", view["players"][1])
        self.assertNotIn("position_delta_thousands", view["players"][1])
        self.assertNotIn(token, view_response.text)
        for forbidden in (
            '"seed"',
            '"card_id"',
            "information_state_id",
            '"tensor"',
            "history_records",
            "known_pile_cards",
        ):
            self.assertNotIn(forbidden, view_response.text)

        for header in (None, "Bearer wrong", f"Basic {token}"):
            headers = {} if header is None else {"Authorization": header}
            rejected = self.client.get(
                f"/api/v2/games/{game_id}/view", headers=headers
            )
            self.assertEqual(rejected.status_code, 401)
            self.assertEqual(rejected.headers["cache-control"], "no-store")

        for extra in (
            {"player_count": 2},
            {"player_names": ["YOU", "COMPUTER"]},
            {"round_count": 1},
            {"options": {"starting_share": True}},
        ):
            with self.subTest(extra=extra):
                rejected = self.client.post("/api/v2/games", json=extra)
                self.assertEqual(rejected.status_code, 422)

    def test_seeded_sessions_have_reproducible_independent_rng_streams(self) -> None:
        first, _token, _payload = self.create_game(seed=9981)
        second, _token, _payload = self.create_game(seed=9981)
        one = self.store.get(first)
        two = self.store.get(second)
        self.assertIsNot(one.chance_rng, one.policy_rng)
        self.assertEqual(one.chance_rng.getstate(), two.chance_rng.getstate())
        self.assertEqual(one.policy_rng.getstate(), two.policy_rng.getstate())
        chance_before = one.chance_rng.getstate()
        one.policy_rng.random()
        self.assertEqual(one.chance_rng.getstate(), chance_before)
        self.assertEqual(
            (
                one.state.first_player,
                one.state.round,
                one.state.phase,
                one.state.stage,
                tuple(one.state.prices),
                tuple(
                    tuple(card.card_type for card in hand)
                    for hand in one.state._hands
                ),
            ),
            (
                two.state.first_player,
                two.state.round,
                two.state.phase,
                two.state.stage,
                tuple(two.state.prices),
                tuple(
                    tuple(card.card_type for card in hand)
                    for hand in two.state._hands
                ),
            ),
        )

    def test_computer_policy_is_player_scoped_automatic_and_legal(self) -> None:
        class TrackingPolicy:
            def __init__(self) -> None:
                self.calls: list[tuple[int | None, tuple[int, ...], int]] = []

            def choose_action(self, state, information, legal_actions, rng):  # type: ignore[no-untyped-def]
                del state, rng
                selected = legal_actions[0].action_id
                self.calls.append(
                    (
                        information.player_id,
                        tuple(action.action_id for action in legal_actions),
                        selected,
                    )
                )
                return selected

        policy = TrackingPolicy()
        store = V2SessionStore(policy=policy)
        client = TestClient(create_app(store_v2=store))
        response = client.post(
            "/api/v2/games", json={"options": {}, "seed": 23}
        )
        self.assertEqual(response.status_code, 201, response.text)
        payload = response.json()
        token = payload["game_url"].split("#seat=", 1)[1]
        view = client.get(
            f"/api/v2/games/{payload['game_id']}/view",
            headers={"Authorization": f"Bearer {token}"},
        ).json()
        if not policy.calls:
            self.assertIsNotNone(view["supply_batch"])
            response = client.post(
                f"/api/v2/games/{payload['game_id']}/supply",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "plan_id": view["supply_batch"]["plans"][0]["plan_id"],
                    "expected_revision": view["revision"],
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            view = response.json()
        self.assertNotEqual(view["active_player_id"], 1)
        self.assertTrue(policy.calls)
        for player_id, legal_ids, selected in policy.calls:
            self.assertEqual(player_id, 1)
            self.assertIn(selected, legal_ids)

    def test_supply_is_complete_atomic_canonical_and_rolls_back(self) -> None:
        game_id, token, _payload = self.create_game(seed=23)
        before = self.view(game_id, token)
        batch = before["supply_batch"]
        self.assertIsNotNone(batch)
        self.assertEqual(len(batch["cards"]), 2)
        card_refs = {item["card_ref"] for item in batch["cards"]}
        self.assertEqual(len(card_refs), 2)
        for plan in batch["plans"]:
            self.assertEqual(len(plan["placements"]), 2)
            self.assertEqual(
                {item["visibility"] for item in plan["placements"]},
                {"face_up", "face_down"},
            )
            self.assertEqual(
                {item["card_ref"] for item in plan["placements"]}, card_refs
            )

        session = self.store.get(game_id)
        snapshot = self._engine_snapshot(session)
        raw_supply_action = int(session.state.legal_actions(0)[0])
        micro_action = self.submit_action(
            game_id, token, before, raw_supply_action
        )
        self.assertEqual(micro_action.status_code, 422)
        self.assertEqual(
            micro_action.json()["error"]["code"], "supply_plan_required"
        )
        self.assertEqual(self._engine_snapshot(session), snapshot)
        invalid = self.submit_supply(game_id, token, before, "not-a-plan")
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(invalid.json()["error"]["code"], "invalid_supply_plan")
        self.assertEqual(self._engine_snapshot(session), snapshot)
        self.assertEqual(self.view(game_id, token), before)

        same_pile = next(
            plan
            for plan in batch["plans"]
            if len({item["stockpile_id"] for item in plan["placements"]}) == 1
        )
        # Building the view installs the exact server-side canonical action map.
        record = session.supply_plans[same_pile["plan_id"]]
        expected = session.state
        expected_stages: list[str] = []
        for action_id in record.action_ids:
            expected_stages.append(expected.stage)
            information, legal = platform.observe_game_state(
                session.rule_set, expected, 0
            )
            del information
            next_state, _record, report = platform.advance_game(
                session.rule_set,
                expected,
                legal,
                platform.ActionRequest(player_id=0, action_id=action_id),
            )
            self.assertTrue(report.valid, report.errors)
            expected = next_state
        self.assertEqual(
            expected_stages,
            ["supply_card", "supply_up_pile", "supply_down_pile"],
        )
        prior_sequence = len(session.state.history_records)
        accepted = self.submit_supply(game_id, token, before, same_pile["plan_id"])
        self.assertEqual(accepted.status_code, 200, accepted.text)
        human_records = [
            item
            for item in session.state.history_records[prior_sequence:]
            if item["player"] == 0
        ]
        self.assertEqual(
            [item["stage"] for item in human_records[:3]], expected_stages
        )
        self.assertEqual(
            [item["action"] for item in human_records[:3]], list(record.action_ids)
        )

        stale = self.submit_supply(game_id, token, before, same_pile["plan_id"])
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.json()["error"]["code"], "stale_revision")

    def test_concurrent_supply_submission_is_serialized(self) -> None:
        game_id, token, _payload = self.create_game(seed=5)
        view = self.view(game_id, token)
        plan_id = view["supply_batch"]["plans"][0]["plan_id"]

        def post_once() -> int:
            return self.submit_supply(game_id, token, view, plan_id).status_code

        with ThreadPoolExecutor(max_workers=2) as executor:
            statuses = sorted(executor.map(lambda _value: post_once(), range(2)))
        self.assertEqual(statuses, [200, 409])

    def test_demand_decision_is_one_atomic_browser_action_with_engine_history(self) -> None:
        game_id, token, _payload = self.create_game(seed=23)
        demand = self.drive_until(
            game_id,
            token,
            lambda item: item["decision_batch"] is not None
            and item["decision_batch"]["kind"] == "demand",
        )
        self.assertEqual(demand["legal_actions"], [])
        plans = demand["decision_batch"]["plans"]
        self.assertTrue(plans)
        self.assertTrue(
            all(
                set(plan)
                == {
                    "plan_id",
                    "stockpile_id",
                    "amount_thousands",
                    "marker_index",
                }
                for plan in plans
            )
        )
        session = self.store.get(game_id)
        snapshot = self._engine_snapshot(session)
        raw_first_action = int(session.state.legal_actions(0)[0])
        rejected = self.submit_action(game_id, token, demand, raw_first_action)
        self.assertEqual(rejected.status_code, 422)
        self.assertEqual(
            rejected.json()["error"]["code"], "decision_plan_required"
        )
        invalid = self.submit_decision(game_id, token, demand, "not-a-plan")
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(invalid.json()["error"]["code"], "invalid_decision_plan")
        self.assertEqual(self._engine_snapshot(session), snapshot)

        selected = plans[0]
        record = session.decision_plans[selected["plan_id"]]
        expected = session.state
        expected_stages: list[str] = []
        for action_id in record.action_ids:
            expected_stages.append(expected.stage)
            expected = _apply_for_test(session, expected, 0, action_id)
        self.assertEqual(expected_stages, ["demand_pile", "demand_bid"])
        prior_sequence = len(session.state.history_records)
        accepted = self.submit_decision(
            game_id, token, demand, selected["plan_id"]
        )
        self.assertEqual(accepted.status_code, 200, accepted.text)
        human_records = [
            item
            for item in session.state.history_records[prior_sequence:]
            if item["player"] == 0
        ]
        self.assertEqual(
            [item["stage"] for item in human_records[:2]], expected_stages
        )
        self.assertEqual(
            [item["action"] for item in human_records[:2]], list(record.action_ids)
        )
        self.assertEqual(accepted.json()["revision"], demand["revision"] + 1)
        stale = self.submit_decision(
            game_id, token, demand, selected["plan_id"]
        )
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.json()["error"]["code"], "stale_revision")

    def test_market_impact_decision_is_atomic_and_moves_authoritative_price(self) -> None:
        game_id, token, _payload = self.create_game(
            options={
                "market_impact": True,
                "trading_fees": True,
                "dividends": True,
                "sell_order": True,
            },
            seed=2,
        )
        impact = self.drive_until(
            game_id,
            token,
            lambda item: item["decision_batch"] is not None
            and item["decision_batch"]["kind"] == "market_impact",
            limit=4_000,
        )
        self.assertEqual(impact["legal_actions"], [])
        selected = impact["decision_batch"]["plans"][0]
        self.assertEqual(abs(selected["movement"]), 2)
        session = self.store.get(game_id)
        record = session.decision_plans[selected["plan_id"]]
        before_price = session.state._company_price(selected["company_id"])
        before_history = len(session.state.history_records)
        response = self.submit_decision(
            game_id, token, impact, selected["plan_id"]
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            session.state._company_price(selected["company_id"]),
            before_price + selected["movement"],
        )
        human_records = [
            item
            for item in session.state.history_records[before_history:]
            if item["player"] == 0
        ]
        self.assertEqual(
            [item["stage"] for item in human_records[:2]],
            ["action_direction", "action_company"],
        )
        self.assertEqual(
            [item["action"] for item in human_records[:2]], list(record.action_ids)
        )

    def test_ordered_piles_interleave_and_redact_face_down_cards(self) -> None:
        game_id, token, _payload = self.create_game(seed=23)
        first = self.view(game_id, token)
        batch = first["supply_batch"]
        first_plan = next(
            plan
            for plan in batch["plans"]
            if len({item["stockpile_id"] for item in plan["placements"]}) == 1
        )
        target = first_plan["placements"][0]["stockpile_id"]
        session = self.store.get(game_id)
        first_record = session.supply_plans[first_plan["plan_id"]]
        hand = session.state._hands[0][:2]
        _namespace, first_card_ordinal = session.rule_set.action_codec.decode(
            first_record.action_ids[0]
        )
        first_hidden_id = hand[1 - first_card_ordinal].card_id
        response = self.submit_supply(game_id, token, first, first_plan["plan_id"])
        self.assertEqual(response.status_code, 200, response.text)

        second = response.json()
        if second["supply_batch"] is None:
            second = self.drive_until(
                game_id, token, lambda item: item["supply_batch"] is not None
            )
        second_plan = next(
            plan
            for plan in second["supply_batch"]["plans"]
            if next(
                item for item in plan["placements"] if item["visibility"] == "face_up"
            )["stockpile_id"]
            == target
        )
        second_record = session.supply_plans[second_plan["plan_id"]]
        second_hand = session.state._hands[0][:2]
        _namespace, second_card_ordinal = session.rule_set.action_codec.decode(
            second_record.action_ids[0]
        )
        second_visible_id = second_hand[second_card_ordinal].card_id
        response = self.submit_supply(game_id, token, second, second_plan["plan_id"])
        self.assertEqual(response.status_code, 200, response.text)
        current = response.json()

        ledger_ids = [entry.card_id for entry in session.ordered_piles[target]]
        self.assertLess(ledger_ids.index(first_hidden_id), ledger_ids.index(second_visible_id))
        pile = next(
            item for item in current["stockpiles"] if item["stockpile_id"] == target
        )
        hidden_index = ledger_ids.index(first_hidden_id)
        visible_index = ledger_ids.index(second_visible_id)
        self.assertEqual(pile["cards_bottom_to_top"][hidden_index]["visibility"], "remembered")
        self.assertTrue(pile["cards_bottom_to_top"][hidden_index]["face_down"])
        self.assertEqual(pile["cards_bottom_to_top"][visible_index]["visibility"], "visible")
        unknown = [
            card
            for stockpile in current["stockpiles"]
            for card in stockpile["cards_bottom_to_top"]
            if card["visibility"] == "hidden"
        ]
        self.assertTrue(unknown)
        self.assertTrue(all(card == {"visibility": "hidden"} for card in unknown))
        self.assertNotIn('"card_id"', response.text)

    def test_explicit_units_and_human_position_are_authoritative(self) -> None:
        game_id, token, _payload = self.create_game(
            options={"trading_fees": True, "dividends": True}, seed=2
        )
        view = self.view(game_id, token)
        self.assertTrue(
            all("price_dollars_per_share" in company for company in view["companies"])
        )
        cards = [item["card"] for item in view["supply_batch"]["cards"]]
        for card in cards:
            if card["kind"] == "stock":
                self.assertGreaterEqual(card["shares_thousands"], 1)
                self.assertNotIn("quantity", card)
            elif card["kind"] == "trading_fee":
                self.assertLess(card["cash_effect_thousands"], 0)
        for slot in view["private"]["market_information"]:
            card = slot["card"]
            if card.get("forecast") == "DIVIDEND":
                self.assertGreater(card["cash_effect_thousands"], 0)
        for holding in view["private"]["holdings"]:
            self.assertEqual(
                holding["market_value_thousands"],
                holding["shares_thousands"]
                * holding["price_dollars_per_share"],
            )
        self.assertEqual(
            view["players"][0]["position_value_thousands"],
            sum(item["market_value_thousands"] for item in view["private"]["holdings"]),
        )
        self.assertNotIn("position_value_thousands", view["players"][1])

    def test_demand_checkpoint_cash_deltas_gate_all_game_actions(self) -> None:
        game_id, token, _payload = self.create_game(
            options={"trading_fees": True}, seed=23
        )
        demand = self.drive_until(
            game_id,
            token,
            lambda item: item["phase"] == "demand" and item["checkpoint"] is None,
        )
        session = self.store.get(game_id)
        self.assertEqual(session.state.round, 1)
        self.assertTrue(all(not player.fees for player in session.state.players))
        before_cash = [player["cash_thousands"] for player in demand["players"]]
        before_position = demand["players"][0]["position_value_thousands"]
        checkpoint = self.drive_until(
            game_id,
            token,
            lambda item: item["checkpoint"] is not None
            and item["checkpoint"]["kind"] == "demand_result",
        )
        self.assertEqual(checkpoint["phase"], "DEMAND_RESULT")
        self.assertEqual(checkpoint["pending_decision"]["kind"], "acknowledge")
        self.assertEqual(checkpoint["legal_actions"], [])
        self.assertIsNone(checkpoint["supply_batch"])
        acquired_fees: list[int] = []
        for index, player in enumerate(checkpoint["players"]):
            expected = player["cash_thousands"] - before_cash[index]
            self.assertEqual(player["cash_delta_thousands"], expected or None)
            bid_total = sum(session.state.players[index].bids)
            deferred_fee_total = sum(session.state.players[index].fees)
            paid_fee_total = -expected - bid_total
            self.assertGreaterEqual(paid_fee_total, 0)
            acquired_fees.append(paid_fee_total + deferred_fee_total)
        discarded_fee_total = sum(
            abs(int(card.value or 0))
            for card in session.state.discards
            if card.card_type == platform.CardType.TRADING_FEE.value
        )
        self.assertGreater(discarded_fee_total, 0)
        self.assertEqual(sum(acquired_fees), discarded_fee_total)
        expected_position = (
            checkpoint["players"][0]["position_value_thousands"] - before_position
        )
        self.assertNotEqual(expected_position, 0)
        self.assertEqual(
            checkpoint["players"][0]["position_delta_thousands"],
            expected_position,
        )

        repeated = self.view(game_id, token)
        self.assertEqual(repeated, checkpoint)
        blocked_action = self.client.post(
            f"/api/v2/games/{game_id}/actions",
            headers={"Authorization": f"Bearer {token}"},
            json={"action_id": 0, "expected_revision": checkpoint["revision"]},
        )
        self.assertEqual(blocked_action.status_code, 409)
        self.assertEqual(blocked_action.json()["error"]["code"], "checkpoint_pending")
        blocked_supply = self.client.post(
            f"/api/v2/games/{game_id}/supply",
            headers={"Authorization": f"Bearer {token}"},
            json={"plan_id": "nope", "expected_revision": checkpoint["revision"]},
        )
        self.assertEqual(blocked_supply.status_code, 409)
        self.assertEqual(blocked_supply.json()["error"]["code"], "checkpoint_pending")
        wrong_ack = self.client.post(
            f"/api/v2/games/{game_id}/acknowledgements",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "checkpoint_id": "wrong",
                "expected_revision": checkpoint["revision"],
            },
        )
        self.assertEqual(wrong_ack.status_code, 409)
        self.assertEqual(wrong_ack.json()["error"]["code"], "stale_checkpoint")
        self.assertEqual(self.view(game_id, token), checkpoint)

        accepted_ack = self.acknowledge(game_id, token, checkpoint)
        self.assertEqual(accepted_ack.status_code, 200, accepted_ack.text)
        after_ack = accepted_ack.json()
        self.assertIsNone(after_ack["checkpoint"])
        self.assertEqual(
            [player["cash_delta_thousands"] for player in after_ack["players"]],
            [player["cash_delta_thousands"] for player in checkpoint["players"]],
        )
        self.assertEqual(
            after_ack["players"][0]["position_delta_thousands"],
            checkpoint["players"][0]["position_delta_thousands"],
        )

    def test_dividend_claim_replaces_and_waive_clears_human_cash_delta(self) -> None:
        for claim in (True, False):
            with self.subTest(claim=claim):
                game_id, token, _payload = self.create_game(
                    options={"dividends": True}, seed=0
                )
                decision = self.drive_until(
                    game_id,
                    token,
                    lambda item: item["pending_decision"]["kind"]
                    == "dividend_claim",
                )
                selected = next(
                    action
                    for action in decision["legal_actions"]
                    if action["control"] == "dividend"
                    and action["label"].lower().startswith(
                        "claim" if claim else "waive"
                    )
                )
                session = self.store.get(game_id)
                session.latest_cash_deltas = (-9, 4)
                before_cash = decision["players"][0]["cash_thousands"]
                response = self.submit_action(
                    game_id, token, decision, selected["action_id"]
                )
                self.assertEqual(response.status_code, 200, response.text)
                after = response.json()
                actual_delta = after["players"][0]["cash_thousands"] - before_cash
                if claim:
                    self.assertGreater(actual_delta, 0)
                    self.assertEqual(
                        after["players"][0]["cash_delta_thousands"], actual_delta
                    )
                else:
                    self.assertEqual(actual_delta, 0)
                    self.assertIsNone(
                        after["players"][0]["cash_delta_thousands"]
                    )

    def test_observable_sale_updates_both_metrics_and_hold_clears_both(self) -> None:
        def selling_choice(
            current: dict,
            *,
            require_sale: bool,
        ) -> dict | None:
            session = self.store.get(current["game_id"])
            if (
                current["phase"] != "selling"
                or session.state.current_player() != 0
            ):
                return None
            for action in current["legal_actions"]:
                preview = action.get("sale_preview")
                if preview is None:
                    continue
                is_sale = preview["shares_thousands"] > 0
                if is_sale != require_sale:
                    continue
                successor = _apply_for_test(
                    session, session.state, 0, action["action_id"]
                )
                # Keep the assertion local to the submitted sale/HOLD.  If the
                # cursor passed to COMPUTER or Movement, automatic work could
                # legitimately produce a later metric event in the response.
                if (
                    successor.phase == platform.Phase.SELLING.value
                    and successor.current_player() == 0
                ):
                    return action
            return None

        for require_sale in (True, False):
            with self.subTest(require_sale=require_sale):
                game_id, token, _payload = self.create_game(
                    options={"sell_order": True}, seed=23
                )
                demand_result = self.drive_until(
                    game_id,
                    token,
                    lambda item: item["checkpoint"] is not None
                    and item["checkpoint"]["kind"] == "demand_result",
                )
                purchase_human = demand_result["players"][0]
                self.assertIsNotNone(purchase_human["cash_delta_thousands"])
                self.assertIsNotNone(purchase_human["position_delta_thousands"])
                acknowledged = self.acknowledge(game_id, token, demand_result)
                self.assertEqual(acknowledged.status_code, 200, acknowledged.text)
                self.assertEqual(
                    acknowledged.json()["players"][0]["cash_delta_thousands"],
                    purchase_human["cash_delta_thousands"],
                )
                self.assertEqual(
                    acknowledged.json()["players"][0]["position_delta_thousands"],
                    purchase_human["position_delta_thousands"],
                )

                selected: dict | None = None

                def has_selling_choice(item: dict) -> bool:
                    nonlocal selected
                    selected = selling_choice(item, require_sale=require_sale)
                    return selected is not None

                selling = self.drive_until(game_id, token, has_selling_choice)
                self.assertIsNotNone(selected)
                before_human = selling["players"][0]
                response = self.submit_action(
                    game_id, token, selling, selected["action_id"]
                )
                self.assertEqual(response.status_code, 200, response.text)
                after = response.json()
                after_human = after["players"][0]
                self.assertEqual(after["phase"], "selling")

                preview = selected["sale_preview"]
                if require_sale:
                    self.assertGreater(preview["shares_thousands"], 0)
                    self.assertEqual(
                        after_human["cash_thousands"]
                        - before_human["cash_thousands"],
                        preview["gross_value_thousands"],
                    )
                    self.assertEqual(
                        after_human["position_value_thousands"]
                        - before_human["position_value_thousands"],
                        -preview["gross_value_thousands"],
                    )
                    self.assertEqual(
                        after_human["cash_delta_thousands"],
                        preview["gross_value_thousands"],
                    )
                    self.assertEqual(
                        after_human["position_delta_thousands"],
                        -preview["gross_value_thousands"],
                    )
                else:
                    self.assertEqual(preview["shares_thousands"], 0)
                    self.assertEqual(
                        after_human["cash_thousands"],
                        before_human["cash_thousands"],
                    )
                    self.assertEqual(
                        after_human["position_value_thousands"],
                        before_human["position_value_thousands"],
                    )
                    self.assertIsNone(after_human["cash_delta_thousands"])
                    self.assertIsNone(after_human["position_delta_thousands"])

    def test_latest_metric_slots_replace_independently_and_market_zero_is_local(self) -> None:
        game_id, token, _payload = self.create_game(seed=23)
        session = self.store.get(game_id)
        state = session.state
        state.players[0].regular_portfolio[0] = 2
        session.latest_cash_deltas = (-9, 4)
        session.latest_human_position_delta = 6
        session.latest_market_deltas = {0: 2}
        session.latest_presentation_event_sequence = state._presentation_sequence

        # A cash-capable zero replaces only that player's cash annotation.
        self.store._replace_cash_delta(session, 0, 0)
        self.assertEqual(session.latest_cash_deltas, (None, 4))
        self.assertEqual(session.latest_human_position_delta, 6)
        self.assertEqual(session.latest_market_deltas, {0: 2})
        self.store._replace_cash_delta(session, 1, 5)
        self.assertEqual(session.latest_cash_deltas, (None, 5))
        self.assertEqual(session.latest_human_position_delta, 6)

        def append_movement(company_id: int, actual_delta: int) -> None:
            prior = state._company_price(company_id)
            state._presentation_sequence += 1
            state._presentation_events.append(
                platform.PresentationEventRecord(
                    presentation_sequence=state._presentation_sequence,
                    round=state.round,
                    event_type="market_movement",
                    cause="test",
                    company_id=company_id,
                    company_name=session.rule_set.company_names[company_id],
                    prior_price=prior,
                    requested_delta=actual_delta,
                    actual_delta=actual_delta,
                    resulting_price=prior + actual_delta,
                )
            )
            self.store._record_price_events(session, state)

        # An unrelated company replaces only its own slot and, because YOU do
        # not hold it, cannot disturb the current POSITION annotation.
        append_movement(1, -1)
        self.assertEqual(session.latest_market_deltas, {0: 2, 1: -1})
        self.assertEqual(session.latest_human_position_delta, 6)

        # A newer event replaces only the same company.  YOU hold 2K Cosmic,
        # so its price event is also position-capable.
        append_movement(0, -3)
        self.assertEqual(session.latest_market_deltas, {0: -3, 1: -1})
        self.assertEqual(session.latest_human_position_delta, -6)
        self.assertEqual(session.latest_cash_deltas, (None, 5))

        # A zero result clears the capable company, not any unrelated company
        # or metric.  A zero for a held company also clears POSITION.
        append_movement(1, 0)
        self.assertEqual(session.latest_market_deltas, {0: -3})
        self.assertEqual(session.latest_human_position_delta, -6)
        append_movement(0, 0)
        self.assertEqual(session.latest_market_deltas, {})
        self.assertIsNone(session.latest_human_position_delta)
        self.assertEqual(session.latest_cash_deltas, (None, 5))

        view = self.view(game_id, token)
        self.assertTrue(
            all(
                company["price_delta_dollars_per_share"] is None
                for company in view["companies"]
            )
        )

    def test_resolved_auction_snapshot_persists_until_round_acknowledgement(self) -> None:
        game_id, token, _payload = self.create_game(seed=23)
        demand_result = self.drive_until(
            game_id,
            token,
            lambda item: item["checkpoint"] is not None
            and item["checkpoint"]["kind"] == "demand_result",
        )
        resolved = demand_result["stockpiles"]
        self.assertEqual(len(resolved), 4)
        self.assertTrue(all(pile["resolved"] for pile in resolved))
        self.assertTrue(all(pile["locked"] for pile in resolved))
        self.assertTrue(all(pile["bid"] is not None for pile in resolved))
        self.assertTrue(all(pile["purchaser_id"] in {0, 1} for pile in resolved))
        self.assertEqual(
            Counter(pile["purchaser_id"] for pile in resolved), Counter({0: 2, 1: 2})
        )
        self.assertEqual(
            len({pile["stockpile_id"] for pile in resolved}), 4
        )
        hidden = [
            card
            for pile in resolved
            for card in pile["cards_bottom_to_top"]
            if card["visibility"] == "hidden"
        ]
        self.assertTrue(all(card == {"visibility": "hidden"} for card in hidden))
        for pile in resolved:
            if pile["purchaser_id"] == 0:
                self.assertTrue(
                    all(
                        card["visibility"] != "hidden"
                        for card in pile["cards_bottom_to_top"]
                    )
                )
        final_markers = {
            (marker["player_id"], marker["marker_index"])
            for player in demand_result["players"]
            for marker in player["bid_markers"]
            if marker["status"] == "locked"
        }
        self.assertEqual(final_markers, {(0, 0), (0, 1), (1, 0), (1, 1)})

        acknowledged = self.acknowledge(game_id, token, demand_result)
        self.assertEqual(acknowledged.status_code, 200, acknowledged.text)
        after_demand = acknowledged.json()
        self.assertEqual(after_demand["stockpiles"], resolved)
        round_result = self.drive_until(
            game_id,
            token,
            lambda item: item["checkpoint"] is not None
            and item["checkpoint"]["kind"] == "round_result",
        )
        self.assertEqual(round_result["stockpiles"], resolved)
        next_round_response = self.acknowledge(game_id, token, round_result)
        self.assertEqual(next_round_response.status_code, 200, next_round_response.text)
        next_round = next_round_response.json()
        self.assertTrue(all(not pile["resolved"] for pile in next_round["stockpiles"]))
        self.assertNotEqual(next_round["stockpiles"], resolved)

    def test_concurrent_acknowledgement_is_serialized(self) -> None:
        game_id, token, _payload = self.create_game(seed=23)
        checkpoint = self.drive_until(
            game_id,
            token,
            lambda item: item["checkpoint"] is not None,
        )

        def post_once() -> int:
            return self.acknowledge(game_id, token, checkpoint).status_code

        with ThreadPoolExecutor(max_workers=2) as executor:
            statuses = sorted(executor.map(lambda _value: post_once(), range(2)))
        self.assertEqual(statuses, [200, 409])

    def test_illegal_and_stale_generic_actions_do_not_mutate_state(self) -> None:
        game_id, token, _payload = self.create_game(seed=23)
        decision = self.drive_until(
            game_id,
            token,
            lambda item: bool(item["legal_actions"])
            and item["supply_batch"] is None
            and item["decision_batch"] is None,
        )
        session = self.store.get(game_id)
        snapshot = self._engine_snapshot(session)
        illegal = self.submit_action(game_id, token, decision, 999_999)
        self.assertEqual(illegal.status_code, 422)
        self.assertEqual(illegal.json()["error"]["code"], "illegal_action")
        self.assertEqual(self._engine_snapshot(session), snapshot)
        action_id = decision["legal_actions"][0]["action_id"]
        accepted = self.submit_action(game_id, token, decision, action_id)
        self.assertEqual(accepted.status_code, 200, accepted.text)
        stale = self.submit_action(game_id, token, decision, action_id)
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.json()["error"]["code"], "stale_revision")

    def test_round_checkpoint_preserves_latest_delta_until_actual_next_round(self) -> None:
        game_id, token, _payload = self.create_game(seed=23)
        checkpoint = self.drive_until(
            game_id,
            token,
            lambda item: item["checkpoint"] is not None
            and item["checkpoint"]["kind"] == "round_result",
        )
        human = checkpoint["players"][0]
        self.assertEqual(checkpoint["round"], 1)
        self.assertEqual(checkpoint["checkpoint"]["round"], 1)
        self.assertEqual(checkpoint["pending_decision"]["kind"], "acknowledge")
        self.assertIsNone(checkpoint["terminal_results"])
        session = self.store.get(game_id)
        self.assertEqual(
            human["position_delta_thousands"],
            session.latest_human_position_delta,
        )
        self.assertEqual(
            [player["cash_delta_thousands"] for player in checkpoint["players"]],
            list(session.latest_cash_deltas),
        )
        self.assertEqual(
            {
                company["company_id"]: company["price_delta_dollars_per_share"]
                for company in checkpoint["companies"]
                if company["price_delta_dollars_per_share"] is not None
            },
            session.latest_market_deltas,
        )
        self.assertTrue(session.state.is_chance_node())
        chance_state = session.chance_rng.getstate()
        self.assertEqual(self.view(game_id, token), checkpoint)
        self.assertEqual(session.chance_rng.getstate(), chance_state)
        acknowledged = self.acknowledge(game_id, token, checkpoint)
        self.assertEqual(acknowledged.status_code, 200, acknowledged.text)
        next_round = acknowledged.json()
        self.assertEqual(next_round["round"], 2)
        self.assertIsNone(next_round["players"][0]["cash_delta_thousands"])
        self.assertIsNone(next_round["players"][0]["position_delta_thousands"])
        self.assertIsNone(next_round["players"][1]["cash_delta_thousands"])
        self.assertTrue(
            all(
                company["price_delta_dollars_per_share"] is None
                for company in next_round["companies"]
            )
        )
        self.assertTrue(
            all(event["round"] == 2 for event in next_round["recent_events"])
        )

    def test_bankruptcy_round_result_is_refresh_safe_and_private(self) -> None:
        company_id = 3
        game_id, token, _payload = self.create_game(seed=5)
        result, session = self.natural_bankruptcy_checkpoint(
            game_id, token, round_number=2, company_id=company_id
        )

        self.assertEqual(result["checkpoint"]["kind"], "round_result")
        self.assertEqual(result["checkpoint"]["round"], 2)
        self.assertEqual(result["pending_decision"]["kind"], "acknowledge")
        self.assertEqual(result["legal_actions"], [])
        self.assertIsNone(result["supply_batch"])
        self.assertIsNone(result["decision_batch"])
        self.assertIsNone(result["terminal_results"])

        company = next(
            item for item in result["companies"]
            if item["company_id"] == company_id
        )
        self.assertEqual(company["price_dollars_per_share"], 0)
        self.assertEqual(company["price_delta_dollars_per_share"], -3)
        holding = next(
            item for item in result["private"]["holdings"]
            if item["company_id"] == company_id
        )
        self.assertEqual(
            (
                holding["shares_thousands"],
                holding["price_dollars_per_share"],
                holding["market_value_thousands"],
            ),
            (1, 0, 0),
        )
        self.assertNotIn("position_value_thousands", result["players"][1])
        self.assertNotIn("position_delta_thousands", result["players"][1])
        self.assertNotIn("holdings", result["players"][1])
        wire = self.view_response(game_id, token).text
        for internal_name in (
            "bankruptcy_snapshots",
            "bankruptcy_event_corrections",
            "human_shares",
        ):
            self.assertNotIn(internal_name, wire)

        bankruptcy_events = [
            event
            for event in result["recent_events"]
            if event["event_type"] == "bankruptcy"
            and event["company_id"] == company_id
        ]
        self.assertEqual(len(bankruptcy_events), 1)
        event = bankruptcy_events[0]
        self.assertEqual(
            (
                event["prior_price_dollars_per_share"],
                event["price_delta"],
                event["resulting_price_dollars_per_share"],
                event["direction"],
            ),
            (3, -3, 0, "down"),
        )

        # The engine has already completed the real bankruptcy.  Only the
        # Round Result view temporarily restores the human's now-worthless card.
        self.assertEqual(session.state._company_price(company_id), 5)
        self.assertEqual(session.state._represented_shares(0, company_id), 0)
        snapshot = session.bankruptcy_snapshots[company_id]
        self.assertEqual(
            (snapshot.round, snapshot.prior_price, snapshot.human_shares),
            (2, 3, 1),
        )
        engine_snapshot = self._engine_snapshot(session)
        self.assertEqual(self.view(game_id, token), result)
        self.assertEqual(self._engine_snapshot(session), engine_snapshot)
        self.assertTrue(session.state.is_terminal())

    def test_final_round_bankruptcy_result_precedes_terminal_until_acknowledged(self) -> None:
        company_id = 2
        game_id, token, _payload = self.create_game(seed=9)
        result, session = self.natural_bankruptcy_checkpoint(
            game_id, token, round_number=2, company_id=company_id
        )

        self.assertTrue(session.state.is_terminal())
        self.assertEqual(result["checkpoint"]["kind"], "round_result")
        self.assertEqual(result["checkpoint"]["round"], 2)
        self.assertIsNone(result["terminal_results"])
        result_company = next(
            item for item in result["companies"]
            if item["company_id"] == company_id
        )
        self.assertEqual(result_company["price_dollars_per_share"], 0)
        self.assertEqual(result_company["price_delta_dollars_per_share"], -3)
        result_holding = next(
            item for item in result["private"]["holdings"]
            if item["company_id"] == company_id
        )
        self.assertEqual(result_holding["shares_thousands"], 1)
        self.assertEqual(self.view(game_id, token), result)
        unrelated_final_deltas = {
            company["company_id"]: company["price_delta_dollars_per_share"]
            for company in result["companies"]
            if company["company_id"] != company_id
            and company["price_delta_dollars_per_share"] is not None
        }

        acknowledged = self.acknowledge(game_id, token, result)
        self.assertEqual(acknowledged.status_code, 200, acknowledged.text)
        terminal = acknowledged.json()
        self.assertEqual(terminal["phase"], "terminal")
        self.assertIsNone(terminal["checkpoint"])
        self.assertIsNotNone(terminal["terminal_results"])
        terminal_company = next(
            item for item in terminal["companies"]
            if item["company_id"] == company_id
        )
        self.assertEqual(terminal_company["price_dollars_per_share"], 5)
        self.assertIsNone(terminal_company["price_delta_dollars_per_share"])
        self.assertEqual(
            {
                company["company_id"]: company["price_delta_dollars_per_share"]
                for company in terminal["companies"]
                if company["company_id"] != company_id
                and company["price_delta_dollars_per_share"] is not None
            },
            unrelated_final_deltas,
        )
        terminal_holding = next(
            item for item in terminal["private"]["holdings"]
            if item["company_id"] == company_id
        )
        self.assertEqual(terminal_holding["shares_thousands"], 0)
        human_result = next(
            player
            for player in terminal["terminal_results"]["players"]
            if player["player_id"] == 0
        )
        bankrupt_line = next(
            line
            for line in human_result["liquidation"]
            if line["company_id"] == company_id
        )
        self.assertEqual(
            (bankrupt_line["shares_thousands"], bankrupt_line["value_thousands"]),
            (0, 0),
        )
        self.assertEqual(session.bankruptcy_snapshots, {})

    def test_sealed_selling_keeps_computer_private_and_settles_atomically(self) -> None:
        game_id, token, _payload = self.create_game(seed=11)
        selling = self.drive_until(
            game_id,
            token,
            lambda item: item["phase"] == "selling"
            and item["pending_decision"]["kind"] == "sell",
        )
        self.assertEqual(selling["phase_step"], "private_selling")
        self.assertNotIn("position_value_thousands", selling["players"][1])
        self.assertNotIn("holdings", selling["players"][1])
        for action in selling["legal_actions"]:
            preview = action["sale_preview"]
            self.assertIsNotNone(preview)
            self.assertEqual(
                preview["gross_value_thousands"],
                preview["shares_thousands"]
                * preview["price_dollars_per_share"],
            )
        session = self.store.get(game_id)
        authoritative_before = self._engine_snapshot(session)
        response = self.submit_action(
            game_id, token, selling, selling["legal_actions"][0]["action_id"]
        )
        self.assertEqual(response.status_code, 200, response.text)
        after = response.json()
        if after["phase"] == "selling":
            # A private commitment may change the human preview, but must not
            # mutate authoritative cash/history until both plans settle.
            self.assertEqual(
                [item["cash_thousands"] for item in after["players"]],
                [item["cash_thousands"] for item in selling["players"]],
            )
            self.assertEqual(
                self._engine_snapshot(session)[0:9], authoritative_before[0:9]
            )
        self.assertEqual(
            after["players"][0]["position_value_thousands"],
            sum(
                holding["market_value_thousands"]
                for holding in after["private"]["holdings"]
            ),
        )
        self.assertNotIn("position_value_thousands", after["players"][1])

    def test_sell_actions_are_deduplicated_and_prefer_cursor_advancement(self) -> None:
        game_id, token, _payload = self.create_game(
            options={"sell_order": True}, seed=23
        )
        session = self.store.get(game_id)
        current = self.view(game_id, token)
        found_duplicate = False
        for _step in range(2_000):
            if (
                current["phase"] == "selling"
                and current["legal_actions"]
                and session.state.current_player() == 0
            ):
                _information, raw = platform.observe_game_state(
                    session.rule_set, session.state, 0
                )
                raw_groups: dict[tuple[int, int, int, int], list[int]] = {}
                for action in raw:
                    preview = platform.preview_sale_action(
                        session.rule_set, session.state, 0, action.action_id
                    )
                    key = (
                        preview.company_id,
                        preview.quantity_sold,
                        preview.gross_value,
                        preview.resulting_represented,
                    )
                    raw_groups.setdefault(key, []).append(action.action_id)
                projected_keys = [
                    (
                        action["sale_preview"]["company_id"],
                        action["sale_preview"]["shares_thousands"],
                        action["sale_preview"]["gross_value_thousands"],
                        action["sale_preview"]["resulting_shares_thousands"],
                    )
                    for action in current["legal_actions"]
                ]
                self.assertEqual(len(projected_keys), len(set(projected_keys)))
                duplicates = {
                    key: ids for key, ids in raw_groups.items() if len(ids) > 1
                }
                if duplicates:
                    found_duplicate = True
                    canonical_ids = {
                        action["action_id"] for action in current["legal_actions"]
                    }
                    for key in duplicates:
                        projected = next(
                            action
                            for action in current["legal_actions"]
                            if (
                                action["sale_preview"]["company_id"],
                                action["sale_preview"]["shares_thousands"],
                                action["sale_preview"]["gross_value_thousands"],
                                action["sale_preview"]["resulting_shares_thousands"],
                            )
                            == key
                        )
                        before_cursor = (
                            session.state.phase,
                            int(session.state.current_player()),
                            int(session.state._selling_company),
                        )
                        after = _apply_for_test(
                            session,
                            session.state,
                            0,
                            projected["action_id"],
                        )
                        after_cursor = (
                            after.phase,
                            int(after.current_player()),
                            int(after._selling_company),
                        )
                        self.assertNotEqual(after_cursor, before_cursor)
                        hidden_equivalent = next(
                            action_id
                            for action_id in duplicates[key]
                            if action_id not in canonical_ids
                        )
                        rejected = self.submit_action(
                            game_id, token, current, hidden_equivalent
                        )
                        self.assertEqual(rejected.status_code, 422)
                        self.assertEqual(
                            rejected.json()["error"]["code"], "illegal_action"
                        )
                    break
            if current["terminal_results"] is not None:
                break
            current = self.advance_once(game_id, token, current)
        self.assertTrue(found_duplicate, "seed did not exercise duplicate sell actions")

    def test_resignation_is_authoritative_stale_safe_and_closes_queued_references(self) -> None:
        game_id, token, _payload = self.create_game(seed=23)
        view = self.view(game_id, token)
        session = self.store.get(game_id)
        stale = self.client.post(
            f"/api/v2/games/{game_id}/resignations",
            headers={"Authorization": f"Bearer {token}"},
            json={"expected_revision": view["revision"] + 1},
        )
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.json()["error"]["code"], "stale_revision")
        self.assertFalse(session.closed)
        self.assertEqual(self.view(game_id, token), view)

        resigned = self.client.post(
            f"/api/v2/games/{game_id}/resignations",
            headers={"Authorization": f"Bearer {token}"},
            json={"expected_revision": view["revision"]},
        )
        self.assertEqual(resigned.status_code, 204, resigned.text)
        self.assertEqual(resigned.content, b"")
        self.assertEqual(resigned.headers["cache-control"], "no-store")
        self.assertTrue(session.closed)
        missing = self.view_response(game_id, token)
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"]["code"], "game_not_found")
        with self.assertRaises(SessionError) as context:
            self.store.view(session)
        self.assertEqual(context.exception.code, "game_closed")

    def test_complete_default_and_all_options_games_include_every_checkpoint(self) -> None:
        for options, seed in (
            ({}, 101),
            (
                {
                    "market_impact": True,
                    "trading_fees": True,
                    "dividends": True,
                    "sell_order": True,
                },
                2,
            ),
        ):
            with self.subTest(options=options):
                game_id, token, _payload = self.create_game(options=options, seed=seed)
                current = self.view(game_id, token)
                checkpoints: list[tuple[str, int]] = []
                phases: set[str] = set()
                max_price = max(
                    item["price_dollars_per_share"] for item in current["companies"]
                )
                causes: set[str] = set()
                for _step in range(4_000):
                    phases.add(current["phase"])
                    max_price = max(
                        max_price,
                        *(item["price_dollars_per_share"] for item in current["companies"]),
                    )
                    causes.update(
                        event["cause"]
                        for event in current["recent_events"]
                        if event["cause"] is not None
                    )
                    if current["checkpoint"] is not None:
                        checkpoint_key = (
                            current["checkpoint"]["kind"],
                            current["checkpoint"]["round"],
                        )
                        if not checkpoints or checkpoints[-1] != checkpoint_key:
                            checkpoints.append(checkpoint_key)
                    if current["terminal_results"] is not None:
                        break
                    current = self.advance_once(game_id, token, current)
                else:
                    self.fail("V2 two-round playthrough exceeded action limit")

                self.assertEqual(
                    checkpoints,
                    [
                        item
                        for round_number in range(1, 3)
                        for item in (
                            ("demand_result", round_number),
                            ("round_result", round_number),
                        )
                    ],
                )
                terminal = current["terminal_results"]
                self.assertEqual(len(terminal["players"]), 2)
                self.assertTrue(terminal["winner_ids"])
                for player in terminal["players"]:
                    self.assertEqual(
                        player["final_cash_thousands"],
                        player["cash_before_liquidation_thousands"]
                        + player["liquidation_value_thousands"],
                    )
                    self.assertEqual(
                        player["liquidation_value_thousands"],
                        sum(line["value_thousands"] for line in player["liquidation"]),
                    )
                if options:
                    self.assertIn("action", phases)
                    self.assertIn("market_impact", causes)
                    self.assertGreater(max_price, 10)

    def test_final_round_result_precedes_terminal_liquidation(self) -> None:
        game_id, token, _payload = self.create_game(seed=101)
        result = self.drive_until(
            game_id,
            token,
            lambda item: item["checkpoint"] is not None
            and item["checkpoint"]["kind"] == "round_result"
            and item["checkpoint"]["round"] == 2,
            limit=4_000,
        )
        self.assertEqual(result["round"], 2)
        self.assertIsNone(result["terminal_results"])
        self.assertEqual(result["pending_decision"]["kind"], "acknowledge")
        final_round_deltas = (
            result["players"][0]["cash_delta_thousands"],
            result["players"][0]["position_delta_thousands"],
            result["players"][1]["cash_delta_thousands"],
        )
        self.assertIsNotNone(final_round_deltas[1])
        session = self.store.get(game_id)
        self.assertTrue(session.state.is_terminal())
        terminal_response = self.acknowledge(game_id, token, result)
        self.assertEqual(terminal_response.status_code, 200, terminal_response.text)
        terminal = terminal_response.json()
        self.assertEqual(terminal["phase"], "terminal")
        self.assertIsNone(terminal["checkpoint"])
        self.assertIsNotNone(terminal["terminal_results"])
        self.assertEqual(
            (
                terminal["players"][0]["cash_delta_thousands"],
                terminal["players"][0]["position_delta_thousands"],
                terminal["players"][1]["cash_delta_thousands"],
            ),
            final_round_deltas,
        )


if __name__ == "__main__":
    unittest.main()
