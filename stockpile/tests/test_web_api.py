"""Boundary, privacy, and complete-playthrough tests for the local web API."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import unittest

try:
    from fastapi.testclient import TestClient
    from stockpile.web.app import create_app
    from stockpile.web.sessions import SessionStore
except ImportError:  # Core-only installations intentionally omit web extras.
    TestClient = None  # type: ignore[assignment,misc]
    create_app = None  # type: ignore[assignment]
    SessionStore = None  # type: ignore[assignment]


_FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "web_playthrough_seeds.json").read_text(
        encoding="utf-8"
    )
)["scenarios"]


@unittest.skipIf(TestClient is None, "requirements-web.txt is not installed")
class WebApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SessionStore()
        self.client = TestClient(create_app(self.store))

    def create_game(
        self,
        *,
        players: int = 2,
        rounds: int = 1,
        options: dict[str, bool] | None = None,
        seed: int = 101,
    ) -> tuple[str, list[str], dict]:
        response = self.client.post(
            "/api/v1/games",
            json={
                "player_count": players,
                "player_names": [f"Player {index + 1}" for index in range(players)],
                "round_count": rounds,
                "options": options or {},
                "seed": seed,
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        payload = response.json()
        tokens = [
            seat["url"].split("#seat=", 1)[1] for seat in payload["seats"]
        ]
        return payload["game_id"], tokens, payload

    def view(self, game_id: str, token: str):
        return self.client.get(
            f"/api/v1/games/{game_id}/view",
            headers={"Authorization": f"Bearer {token}"},
        )

    def submit(self, game_id: str, token: str, view: dict, action_id: int):
        return self.client.post(
            f"/api/v1/games/{game_id}/actions",
            headers={"Authorization": f"Bearer {token}"},
            json={"action_id": action_id, "expected_revision": view["revision"]},
        )

    def drive_to_terminal(
        self, game_id: str, tokens: list[str], *, limit: int = 4_000
    ) -> tuple[dict, set[str]]:
        phases: set[str] = set()
        for _step in range(limit):
            views = [self.view(game_id, token).json() for token in tokens]
            phases.add(views[0]["phase"])
            if views[0]["phase"] == "terminal":
                return views[0], phases
            candidates = [
                (player_id, view)
                for player_id, view in enumerate(views)
                if view["legal_actions"]
            ]
            self.assertTrue(candidates, [(v["phase"], v["phase_step"]) for v in views])
            player_id, current = candidates[0]
            response = self.submit(
                game_id,
                tokens[player_id],
                current,
                current["legal_actions"][0]["action_id"],
            )
            self.assertEqual(response.status_code, 200, response.text)
        self.fail(f"game did not finish within {limit} player actions")

    def test_setup_is_authoritative_and_no_store(self) -> None:
        response = self.client.get("/api/v1/setup")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        payload = response.json()
        self.assertEqual(payload["mode"], "lite")
        self.assertEqual(payload["defaults"], {"player_count": 2, "round_count": 6})
        self.assertEqual(payload["player_limits"], {"minimum": 2, "maximum": 5})
        self.assertEqual(
            {option["key"] for option in payload["options"]},
            {
                "market_impact",
                "starting_share",
                "trading_fees",
                "dividends",
                "sell_order",
            },
        )
        self.assertTrue(all(not option["default"] for option in payload["options"]))

    def test_setup_accepts_two_to_five_players_and_strict_names(self) -> None:
        for players in range(2, 6):
            with self.subTest(players=players):
                _game, _tokens, payload = self.create_game(players=players)
                self.assertEqual(len(payload["seats"]), players)
        duplicate = self.client.post(
            "/api/v1/games",
            json={
                "player_count": 2,
                "player_names": ["Alice", " alice "],
                "round_count": 1,
                "options": {},
            },
        )
        self.assertEqual(duplicate.status_code, 422)
        extra = self.client.post(
            "/api/v1/games",
            json={
                "player_count": 2,
                "player_names": ["A", "B"],
                "round_count": 1,
                "options": {},
                "player_id": 0,
            },
        )
        self.assertEqual(extra.status_code, 422)

    def test_fragment_token_is_bound_to_one_fixed_seat(self) -> None:
        game_id, tokens, create_payload = self.create_game(seed=7)
        first = self.view(game_id, tokens[0])
        second = self.view(game_id, tokens[1])
        self.assertEqual(first.json()["viewer"]["player_id"], 0)
        self.assertEqual(second.json()["viewer"]["player_id"], 1)
        self.assertNotIn(tokens[0], first.text)
        self.assertNotIn("seed", create_payload)
        self.assertIn("#seat=", create_payload["seats"][0]["url"])

        for header in (None, "Bearer wrong", "Basic " + tokens[0]):
            headers = {} if header is None else {"Authorization": header}
            rejected = self.client.get(
                f"/api/v1/games/{game_id}/view", headers=headers
            )
            self.assertEqual(rejected.status_code, 401)
            self.assertEqual(rejected.headers["cache-control"], "no-store")

    def test_company_patterns_and_action_direction_are_server_authored(self) -> None:
        scenario = _FIXTURES["all_lite_options_market_impact"]
        game_id, tokens, _payload = self.create_game(
            rounds=scenario["round_count"],
            options=scenario["options"],
            seed=scenario["seed"],
        )
        first = self.view(game_id, tokens[0]).json()
        self.assertEqual(
            [company["display_name"] for company in first["companies"]],
            ["COSMIC", "BOTTOMLINE", "LEADING", "AMERICAN", "STANFORD", "EPIC"],
        )
        self.assertEqual(
            [company["pattern"] for company in first["companies"]],
            ["matrix", "ledger", "molecular", "chevron", "crosshatch", "wave"],
        )
        self.assertEqual(
            {company["color"] for company in first["companies"]}, {"#002FA7"}
        )

        for _step in range(200):
            views = [self.view(game_id, token).json() for token in tokens]
            action_cards = [
                card
                for view in views
                for card in (
                    view["private"]["hand"]
                    + view["private"]["available_action_cards"]
                    + [
                        visible
                        for stockpile in view["stockpiles"]
                        for visible in stockpile["visible_cards"]
                    ]
                )
                if card["kind"] == "action"
            ]
            if action_cards:
                for card in action_cards:
                    expected_direction = "up" if card["effect"] == "boom" else "down"
                    self.assertEqual(card["direction"], expected_direction)
                    self.assertEqual(card["movement"], 2)
                break
            player_id, current = next(
                (index, view)
                for index, view in enumerate(views)
                if view["legal_actions"]
            )
            response = self.submit(
                game_id,
                tokens[player_id],
                current,
                current["legal_actions"][0]["action_id"],
            )
            self.assertEqual(response.status_code, 200, response.text)
        else:
            self.fail("no Market Impact card became visible")

    def test_cross_seat_private_information_and_hidden_cards_are_strict(self) -> None:
        game_id, tokens, _payload = self.create_game(seed=13)
        first = self.view(game_id, tokens[0]).json()
        second = self.view(game_id, tokens[1]).json()
        private0 = {
            (slot["card"]["company_id"], str(slot["card"]["forecast"]))
            for slot in first["private"]["market_information"]
            if slot["visibility"] == "private"
        }
        private1 = {
            (slot["card"]["company_id"], str(slot["card"]["forecast"]))
            for slot in second["private"]["market_information"]
            if slot["visibility"] == "private"
        }
        self.assertTrue(private0)
        self.assertTrue(private1)
        self.assertTrue(private0.isdisjoint(private1))

        for _step in range(30):
            views = [self.view(game_id, token).json() for token in tokens]
            if views[0]["phase"] == "demand":
                break
            player_id, current = next(
                (index, view)
                for index, view in enumerate(views)
                if view["legal_actions"]
            )
            response = self.submit(
                game_id,
                tokens[player_id],
                current,
                current["legal_actions"][0]["action_id"],
            )
            self.assertEqual(response.status_code, 200, response.text)
        demand = self.view(game_id, tokens[0]).json()
        hidden = [
            card
            for pile in demand["stockpiles"]
            for card in pile["hidden_cards"]
        ]
        self.assertTrue(hidden)
        self.assertTrue(all(card == {"visibility": "hidden"} for card in hidden))
        encoded = self.view(game_id, tokens[0]).text
        for forbidden in (
            "information_state_id",
            '"tensor"',
            '"card_id"',
            "history_records",
            "random_seed",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_stale_illegal_and_turn_conflict_responses(self) -> None:
        game_id, tokens, _payload = self.create_game(seed=3)
        views = [self.view(game_id, token).json() for token in tokens]
        actor = next(index for index, view in enumerate(views) if view["legal_actions"])
        waiting = 1 - actor
        conflict = self.client.post(
            f"/api/v1/games/{game_id}/actions",
            headers={"Authorization": f"Bearer {tokens[waiting]}"},
            json={"action_id": 0, "expected_revision": views[waiting]["revision"]},
        )
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.json()["error"]["code"], "turn_conflict")

        illegal = self.submit(game_id, tokens[actor], views[actor], 999_999)
        self.assertEqual(illegal.status_code, 422)
        accepted = self.submit(
            game_id,
            tokens[actor],
            views[actor],
            views[actor]["legal_actions"][0]["action_id"],
        )
        self.assertEqual(accepted.status_code, 200, accepted.text)
        stale = self.submit(
            game_id,
            tokens[actor],
            views[actor],
            views[actor]["legal_actions"][0]["action_id"],
        )
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.json()["error"]["code"], "stale_revision")

    def test_same_revision_concurrent_actions_are_serialized(self) -> None:
        game_id, tokens, _payload = self.create_game(seed=5)
        views = [self.view(game_id, token).json() for token in tokens]
        actor = next(index for index, view in enumerate(views) if view["legal_actions"])
        view = views[actor]
        action_id = view["legal_actions"][0]["action_id"]

        def post_once() -> int:
            return self.submit(game_id, tokens[actor], view, action_id).status_code

        with ThreadPoolExecutor(max_workers=2) as executor:
            statuses = sorted(executor.map(lambda _value: post_once(), range(2)))
        self.assertEqual(statuses, [200, 409])

    def test_private_multistep_selection_does_not_change_observer_revision(self) -> None:
        game_id, tokens, _payload = self.create_game(seed=7)
        views = [self.view(game_id, token).json() for token in tokens]
        actor = next(index for index, view in enumerate(views) if view["legal_actions"])
        observer = 1 - actor
        acting = views[actor]
        self.assertEqual(acting["pending_decision"]["kind"], "supply_card")
        before = views[observer]

        response = self.submit(
            game_id,
            tokens[actor],
            acting,
            acting["legal_actions"][0]["action_id"],
        )
        self.assertEqual(response.status_code, 200, response.text)
        after = self.view(game_id, tokens[observer]).json()
        self.assertEqual(after["revision"], before["revision"])
        self.assertEqual(after["phase_step"], "waiting")
        self.assertEqual(after["pending_decision"]["kind"], "waiting")

    def test_two_player_displacement_rebid_uses_exactly_two_markers(self) -> None:
        scenario = _FIXTURES["two_player_outbid_rebid"]
        game_id, tokens, _payload = self.create_game(seed=scenario["seed"])
        for _step in range(50):
            views = [self.view(game_id, token).json() for token in tokens]
            if views[0]["phase"] == "demand":
                break
            player_id, current = next(
                (index, view)
                for index, view in enumerate(views)
                if view["legal_actions"]
            )
            response = self.submit(
                game_id,
                tokens[player_id],
                current,
                current["legal_actions"][0]["action_id"],
            )
            self.assertEqual(response.status_code, 200, response.text)
        else:
            self.fail("did not reach Demand")

        first_player: int | None = None

        def bid_on(pile_id: int) -> int:
            nonlocal first_player
            views_now = [self.view(game_id, token).json() for token in tokens]
            bidder, pile_view = next(
                (index, view)
                for index, view in enumerate(views_now)
                if view["pending_decision"]["kind"] == "bid_pile"
            )
            if first_player is None:
                first_player = bidder
            pile_action = next(
                action
                for action in pile_view["legal_actions"]
                if action["target_id"] == f"stockpile:{pile_id}"
            )
            selected = self.submit(
                game_id, tokens[bidder], pile_view, pile_action["action_id"]
            )
            self.assertEqual(selected.status_code, 200, selected.text)
            bid_view = selected.json()
            bid_action = min(
                bid_view["legal_actions"],
                key=lambda action: action["amount"],
            )
            committed = self.submit(
                game_id, tokens[bidder], bid_view, bid_action["action_id"]
            )
            self.assertEqual(committed.status_code, 200, committed.text)
            return bidder

        first = bid_on(0)
        second = bid_on(0)
        self.assertNotEqual(first, second)
        displaced_view = self.view(game_id, tokens[first]).json()
        statuses = {
            marker["marker_index"]: marker["status"]
            for marker in displaced_view["players"][first]["bid_markers"]
        }
        self.assertEqual(statuses[0], "outbid")
        bid_on(1)
        bid_on(2)
        rebidder = bid_on(3)
        self.assertEqual(rebidder, first_player)

        resolved = self.view(game_id, tokens[0]).json()
        purchasers = [pile["purchaser_id"] for pile in resolved["stockpiles"]]
        self.assertEqual(len(set(purchasers)), 2)
        self.assertEqual(purchasers.count(0), 2)
        self.assertEqual(purchasers.count(1), 2)
        self.assertTrue(
            all(len(player["bid_markers"]) == 2 for player in resolved["players"])
        )

    def test_sealed_selling_does_not_publish_commitment_timing(self) -> None:
        game_id, tokens, _payload = self.create_game(seed=11)
        for _step in range(200):
            views = [self.view(game_id, token).json() for token in tokens]
            if views[0]["phase"] == "selling":
                break
            player_id, current = next(
                (index, view)
                for index, view in enumerate(views)
                if view["legal_actions"]
            )
            response = self.submit(
                game_id,
                tokens[player_id],
                current,
                current["legal_actions"][0]["action_id"],
            )
            self.assertEqual(response.status_code, 200, response.text)
        else:
            self.fail("did not reach sealed selling")

        views = [self.view(game_id, token).json() for token in tokens]
        actor = next(index for index, view in enumerate(views) if view["legal_actions"])
        observer = 1 - actor
        before = views[observer]
        acting = views[actor]
        self.assertTrue(
            all(action["sale_preview"] is not None for action in acting["legal_actions"])
        )
        response = self.submit(
            game_id,
            tokens[actor],
            acting,
            acting["legal_actions"][0]["action_id"],
        )
        self.assertEqual(response.status_code, 200, response.text)
        after = self.view(game_id, tokens[observer]).json()
        self.assertIsNone(after["active_player_id"])
        self.assertEqual(after["phase_step"], "private_selling")
        self.assertEqual(before["revision"], after["revision"])
        self.assertEqual(before["public_history"], after["public_history"])
        self.assertEqual(before["recent_events"], after["recent_events"])

    def test_completed_last_sealed_plan_cannot_expose_future_round_state(self) -> None:
        game_id, tokens, _payload = self.create_game(rounds=2, seed=11)
        for _step in range(250):
            views = [self.view(game_id, token).json() for token in tokens]
            if views[0]["phase"] == "selling":
                break
            player_id, current = next(
                (index, view)
                for index, view in enumerate(views)
                if view["legal_actions"]
            )
            response = self.submit(
                game_id,
                tokens[player_id],
                current,
                current["legal_actions"][0]["action_id"],
            )
            self.assertEqual(response.status_code, 200, response.text)
        else:
            self.fail("did not reach sealed selling")

        session = self.store.get(game_id)
        with session.lock:
            last_player = list(session.state._selling_players)[-1]
        before = self.view(game_id, tokens[last_player]).json()
        current = before
        for _decision in range(20):
            if not current["legal_actions"]:
                break
            response = self.submit(
                game_id,
                tokens[last_player],
                current,
                current["legal_actions"][0]["action_id"],
            )
            self.assertEqual(response.status_code, 200, response.text)
            current = response.json()
        else:
            self.fail("last seat did not finish its private plan")

        self.assertEqual(current["pending_decision"]["kind"], "private_selling")
        with session.lock:
            self.assertIsNotNone(session.sealed_sale)
            self.assertTrue(session.sealed_sale.plans[last_player].complete)
            self.assertFalse(
                all(plan.complete for plan in session.sealed_sale.plans.values())
            )
        self.assertEqual(current["round"], before["round"])
        self.assertEqual(current["companies"], before["companies"])
        self.assertEqual(current["private"]["hand"], before["private"]["hand"])
        self.assertEqual(current["private"]["holdings"], before["private"]["holdings"])

    def test_positive_sealed_sales_replay_and_settle_atomically(self) -> None:
        game_id, tokens, _payload = self.create_game(seed=11)
        for _step in range(200):
            views = [self.view(game_id, token).json() for token in tokens]
            if views[0]["phase"] == "selling":
                break
            player_id, current = next(
                (index, view)
                for index, view in enumerate(views)
                if view["legal_actions"]
            )
            response = self.submit(
                game_id,
                tokens[player_id],
                current,
                current["legal_actions"][0]["action_id"],
            )
            self.assertEqual(response.status_code, 200, response.text)
        else:
            self.fail("did not reach sealed selling")

        starting = [self.view(game_id, token).json() for token in tokens]
        starting_cash = [player["cash"] for player in starting[0]["players"]]
        observer_revision = starting[1]["revision"]
        observer_history = starting[1]["public_history"]
        sold_players: set[int] = set()

        for player_id in range(2):
            for _decision in range(20):
                current = self.view(game_id, tokens[player_id]).json()
                if not current["legal_actions"]:
                    break
                positive = [
                    action
                    for action in current["legal_actions"]
                    if action["sale_preview"] is not None
                    and action["sale_preview"]["quantity"] > 0
                ]
                holds = [
                    action
                    for action in current["legal_actions"]
                    if action["sale_preview"] is not None
                    and action["sale_preview"]["quantity"] == 0
                ]
                if player_id not in sold_players and positive:
                    action = max(
                        positive, key=lambda item: item["sale_preview"]["quantity"]
                    )
                    sold_players.add(player_id)
                else:
                    action = holds[0] if holds else current["legal_actions"][0]
                response = self.submit(
                    game_id,
                    tokens[player_id],
                    current,
                    action["action_id"],
                )
                self.assertEqual(response.status_code, 200, response.text)
            else:
                self.fail("private sale plan did not complete")

            if player_id == 0:
                observer = self.view(game_id, tokens[1]).json()
                self.assertEqual(observer["revision"], observer_revision)
                self.assertEqual(observer["public_history"], observer_history)
                self.assertEqual(
                    [player["cash"] for player in observer["players"]], starting_cash
                )

        self.assertEqual(sold_players, {0, 1})
        terminal = self.view(game_id, tokens[0]).json()
        self.assertIsNotNone(terminal["terminal_results"])
        self.assertTrue(
            any(
                quantity > 0
                for entry in terminal["public_history"]
                for companies in (entry["sale_totals"] or {}).values()
                for quantity in companies.values()
            )
        )
        settled_cash = [
            player["cash_before_liquidation"]
            for player in terminal["terminal_results"]["players"]
        ]
        self.assertTrue(
            all(after > before for before, after in zip(starting_cash, settled_cash))
        )

    def test_sell_order_option_keeps_engine_sequential_selling(self) -> None:
        game_id, tokens, _payload = self.create_game(
            options={"sell_order": True}, seed=29
        )
        for _step in range(200):
            views = [self.view(game_id, token).json() for token in tokens]
            if views[0]["phase"] == "selling":
                break
            player_id, current = next(
                (index, view)
                for index, view in enumerate(views)
                if view["legal_actions"]
            )
            response = self.submit(
                game_id,
                tokens[player_id],
                current,
                current["legal_actions"][0]["action_id"],
            )
            self.assertEqual(response.status_code, 200, response.text)
        else:
            self.fail("did not reach sequential selling")
        actors = [view for view in views if view["legal_actions"]]
        self.assertEqual(len(actors), 1)
        actor = actors[0]
        actor_id = actor["viewer"]["player_id"]
        self.assertTrue(all(view["active_player_id"] == actor_id for view in views))
        self.assertEqual(actor["phase_step"], "sell")
        self.assertTrue(
            all(
                view["phase_step"] == "waiting"
                for view in views
                if view["viewer"]["player_id"] != actor_id
            )
        )

    def test_chat_is_validated_ephemeral_and_not_a_game_revision(self) -> None:
        game_id, tokens, _payload = self.create_game()
        before = self.view(game_id, tokens[0]).json()
        sent = self.client.post(
            f"/api/v1/games/{game_id}/chat",
            headers={"Authorization": f"Bearer {tokens[0]}"},
            json={"message": "  Ready to trade  "},
        )
        self.assertEqual(sent.status_code, 201)
        self.assertEqual(sent.json()["chat_message"]["message"], "Ready to trade")
        after = self.view(game_id, tokens[0]).json()
        self.assertEqual(after["revision"], before["revision"])
        self.assertEqual(after["chat"][-1]["player_id"], 0)
        blank = self.client.post(
            f"/api/v1/games/{game_id}/chat",
            headers={"Authorization": f"Bearer {tokens[0]}"},
            json={"message": "   "},
        )
        self.assertEqual(blank.status_code, 422)

    def test_complete_default_six_round_game(self) -> None:
        scenario = _FIXTURES["default_two_player_six_round"]
        game_id, tokens, _payload = self.create_game(
            rounds=scenario["round_count"], seed=scenario["seed"]
        )
        terminal, phases = self.drive_to_terminal(game_id, tokens)
        self.assertTrue({"supply", "demand", "selling", "terminal"}.issubset(phases))
        self.assertNotIn("action", phases)
        # Default Movement has no player decision, so it is represented by
        # Python-emitted events rather than a transient polled phase.
        self.assertTrue(terminal["recent_events"])
        self.assertEqual(len(terminal["terminal_results"]["players"]), 2)
        self.assertTrue(terminal["terminal_results"]["winner_ids"])

    def test_complete_all_options_game_and_multiple_player_counts(self) -> None:
        scenario = _FIXTURES["all_lite_options_market_impact"]
        all_options = scenario["options"]
        game_id, tokens, _payload = self.create_game(
            rounds=scenario["round_count"],
            options=all_options,
            seed=scenario["seed"],
        )
        terminal, phases = self.drive_to_terminal(game_id, tokens)
        self.assertIn("action", phases)
        self.assertIsNotNone(terminal["terminal_results"])
        self.assertTrue(terminal["recent_events"])
        self.assertGreater(
            max(event["resulting_price"] or 0 for event in terminal["recent_events"]),
            10,
        )
        self.assertTrue(
            any(
                event["cause"] == "market_impact"
                for event in terminal["recent_events"]
            )
        )

        for players, fixture_name in (
            (3, "three_player_complete"),
            (4, "four_player_complete"),
            (5, "five_player_complete"),
        ):
            with self.subTest(players=players):
                player_scenario = _FIXTURES[fixture_name]
                game_id, tokens, _payload = self.create_game(
                    players=players,
                    rounds=player_scenario["round_count"],
                    seed=player_scenario["seed"],
                )
                terminal, _phases = self.drive_to_terminal(game_id, tokens)
                self.assertEqual(
                    len(terminal["terminal_results"]["players"]), players
                )


if __name__ == "__main__":
    unittest.main()
