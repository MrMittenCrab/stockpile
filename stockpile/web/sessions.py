"""Authoritative in-memory sessions for the local Stockpile web application."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
import hashlib
import hmac
import json
import random
import secrets
import threading
import uuid
from typing import Any, Mapping, Sequence

from .. import stockpile_interface as interface
from .. import stockpile_platform as platform
from .schemas import (
    ActionCardV1,
    BidMarkerV1,
    CapabilitiesV1,
    CardV1,
    ChatMessageV1,
    CompanyV1,
    ConfigurationV1,
    CreateGameRequest,
    CreateGameResponse,
    GameViewV1,
    HiddenCardV1,
    HoldingV1,
    InformationCardV1,
    KnownPileCardV1,
    LegalActionV1,
    LiquidationLineV1,
    LiteOptions,
    MarketInformationSlotV1,
    PendingDecisionV1,
    PublicEventV1,
    PublicHistoryEntryV1,
    PublicPlayerV1,
    SalePreviewV1,
    SeatLink,
    StockCardV1,
    StockpileV1,
    TerminalPlayerV1,
    TerminalResultsV1,
    TradingFeeCardV1,
    ViewerPrivateV1,
    ViewerV1,
)


COMPANY_COLORS = (
    "#da4f49",
    "#e7ad3c",
    "#42a874",
    "#3e83c5",
    "#8b65bd",
    "#d97942",
)
MAX_CHAT_MESSAGES = 200
MAX_PUBLIC_ITEMS = 80


class SessionError(RuntimeError):
    """A stable service-layer error converted to one HTTP response."""

    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        super().__init__(message)


@dataclass(slots=True)
class SealedPlayerPlan:
    state: platform.GameState
    action_ids: list[int] = field(default_factory=list)
    complete: bool = False


@dataclass(slots=True)
class SealedSaleBuffer:
    round: int
    plans: dict[int, SealedPlayerPlan]


@dataclass(slots=True)
class GameSession:
    game_id: str
    configuration: interface.GameConfig
    seed: int
    state: platform.GameState
    player_names: tuple[str, ...]
    token_digests: tuple[str, ...]
    rng: random.Random
    lock: threading.RLock = field(default_factory=threading.RLock)
    view_revisions: list[int] = field(default_factory=list)
    view_hashes: list[str] = field(default_factory=list)
    chat: deque[ChatMessageV1] = field(
        default_factory=lambda: deque(maxlen=MAX_CHAT_MESSAGES)
    )
    next_chat_id: int = 1
    used_bid_markers: set[tuple[int, int]] = field(default_factory=set)
    outbid_markers: set[tuple[int, int]] = field(default_factory=set)
    bid_tracking_round: int = 1
    sealed_sale: SealedSaleBuffer | None = None

    @property
    def rule_set(self) -> platform.RuleSet:
        return self.configuration.rule_set


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _apply_player_action(
    rule_set: platform.RuleSet,
    state: platform.GameState,
    player_id: int,
    action_id: int,
) -> platform.GameState:
    _information, legal = platform.observe_game_state(rule_set, state, player_id)
    next_state, _record, report = platform.advance_game(
        rule_set,
        state,
        legal,
        platform.ActionRequest(player_id=player_id, action_id=action_id),
    )
    if not report.valid:
        raise SessionError(422, "illegal_action", "; ".join(report.errors))
    return next_state


def _done_action(state: platform.GameState, player_id: int) -> int:
    for action_id in state.legal_actions(player_id):
        namespace, _ordinal = state.rule_set.action_codec.decode(int(action_id))
        if namespace == "done":
            return int(action_id)
    raise RuntimeError("selling state does not offer a hold action")


def _is_forced_zero_sale(state: platform.GameState, player_id: int) -> bool:
    if state.phase != platform.Phase.SELLING.value or state.stage != "selling":
        return False
    if state.current_player() != player_id:
        return False
    actions = [int(action) for action in state.legal_actions(player_id)]
    if len(actions) != 1:
        return False
    namespace, _ordinal = state.rule_set.action_codec.decode(actions[0])
    return namespace == "done"


def _auto_zero_sales(plan: SealedPlayerPlan, player_id: int) -> None:
    while _is_forced_zero_sale(plan.state, player_id):
        action_id = _done_action(plan.state, player_id)
        plan.action_ids.append(action_id)
        plan.state = _apply_player_action(
            plan.state.rule_set, plan.state, player_id, action_id
        )
    plan.complete = not (
        plan.state.phase == platform.Phase.SELLING.value
        and plan.state.current_player() == player_id
    )


class SessionStore:
    """Process-local session registry with per-game serialization."""

    def __init__(self) -> None:
        self._sessions: dict[str, GameSession] = {}
        self._lock = threading.RLock()

    def create(self, request: CreateGameRequest) -> tuple[GameSession, CreateGameResponse]:
        options = request.options
        configuration = interface.resolve_configuration(
            interface.ConfigurationMode.LITE,
            player_count=request.player_count,
            round_count=request.round_count,
            hand=options.starting_share,
            fees=options.trading_fees,
            dividend=options.dividends,
            split=False,
            majority=False,
            stock_tracks=False,
            sell_order=options.sell_order,
            impact=options.market_impact,
            action_space_mode="compact",
        )
        seed = request.seed if request.seed is not None else secrets.randbits(63)
        initial_input = platform.randomize_initial_input(configuration.rule_set, seed)
        state = platform.GameState(configuration.game, initial_input)
        game_id = uuid.uuid4().hex
        raw_tokens = tuple(secrets.token_urlsafe(32) for _ in request.player_names)
        session = GameSession(
            game_id=game_id,
            configuration=configuration,
            seed=seed,
            state=state,
            player_names=tuple(request.player_names),
            token_digests=tuple(_token_digest(token) for token in raw_tokens),
            rng=random.Random(seed ^ 0x53544F434B50494C),
            view_revisions=[0] * request.player_count,
            bid_tracking_round=state.round,
        )
        with session.lock:
            self._advance_automatic(session)
            self._initialize_view_hashes(session)
        with self._lock:
            self._sessions[game_id] = session
        response = CreateGameResponse(
            game_id=game_id,
            seats=[
                SeatLink(
                    player_id=player_id,
                    player_name=request.player_names[player_id],
                    url=f"/game/{game_id}#seat={token}",
                )
                for player_id, token in enumerate(raw_tokens)
            ],
        )
        return session, response

    def get(self, game_id: str) -> GameSession:
        with self._lock:
            session = self._sessions.get(game_id)
        if session is None:
            raise SessionError(404, "game_not_found", "Game not found")
        return session

    def authenticate(self, game_id: str, token: str | None) -> tuple[GameSession, int]:
        if not token:
            raise SessionError(401, "invalid_seat_token", "A valid seat token is required")
        session = self.get(game_id)
        candidate = _token_digest(token)
        for player_id, expected in enumerate(session.token_digests):
            if hmac.compare_digest(candidate, expected):
                return session, player_id
        raise SessionError(401, "invalid_seat_token", "A valid seat token is required")

    def view(self, session: GameSession, player_id: int) -> GameViewV1:
        with session.lock:
            return self._build_view(session, player_id)

    def act(
        self,
        session: GameSession,
        player_id: int,
        *,
        action_id: int,
        expected_revision: int,
    ) -> GameViewV1:
        with session.lock:
            if expected_revision != session.view_revisions[player_id]:
                raise SessionError(
                    409,
                    "stale_revision",
                    "The game view changed; refresh before acting",
                )
            if session.state.is_terminal():
                raise SessionError(409, "game_finished", "The game has ended")

            if self._is_buffered_selling(session):
                buffer = self._ensure_sealed_buffer(session)
                plan = buffer.plans[player_id]
                if plan.complete:
                    raise SessionError(
                        409,
                        "turn_conflict",
                        "This seat has finished its private selling plan",
                    )
                if plan.state.current_player() != player_id:
                    raise SessionError(409, "turn_conflict", "This seat cannot act now")
                legal_ids = {int(value) for value in plan.state.legal_actions(player_id)}
                if action_id not in legal_ids:
                    raise SessionError(422, "illegal_action", "Action is not legal now")
                plan.action_ids.append(action_id)
                plan.state = _apply_player_action(
                    session.rule_set, plan.state, player_id, action_id
                )
                _auto_zero_sales(plan, player_id)
                if all(candidate.complete for candidate in buffer.plans.values()):
                    self._commit_sealed_buffer(session)
                    self._advance_automatic(session)
            else:
                if session.state.current_player() != player_id:
                    raise SessionError(
                        409, "turn_conflict", "Another player must act first"
                    )
                legal_ids = {
                    int(value) for value in session.state.legal_actions(player_id)
                }
                if action_id not in legal_ids:
                    raise SessionError(422, "illegal_action", "Action is not legal now")
                self._track_bid_before_action(session, action_id)
                session.state = _apply_player_action(
                    session.rule_set, session.state, player_id, action_id
                )
                self._advance_automatic(session)
            self._refresh_view_revisions(session)
            return self._build_view(session, player_id)

    def add_chat(
        self, session: GameSession, player_id: int, message: str
    ) -> ChatMessageV1:
        with session.lock:
            item = ChatMessageV1(
                message_id=session.next_chat_id,
                player_id=player_id,
                player_name=session.player_names[player_id],
                message=message,
                created_at=datetime.now(UTC).isoformat(),
            )
            session.next_chat_id += 1
            session.chat.append(item)
            return item

    def _advance_automatic(self, session: GameSession) -> None:
        while not session.state.is_terminal() and session.state.is_chance_node():
            outcomes = session.state.chance_outcomes()
            if not outcomes:
                raise RuntimeError("chance state has no outcomes")
            threshold = session.rng.random()
            cumulative = 0.0
            selected = int(outcomes[-1][0])
            for action_id, probability in outcomes:
                cumulative += float(probability)
                if threshold <= cumulative:
                    selected = int(action_id)
                    break
            session.state.apply_action(selected)

        if session.bid_tracking_round != session.state.round:
            session.bid_tracking_round = session.state.round
            session.used_bid_markers.clear()
            session.outbid_markers.clear()

        if self._is_buffered_selling(session):
            self._ensure_sealed_buffer(session)
            if session.sealed_sale and all(
                plan.complete for plan in session.sealed_sale.plans.values()
            ):
                self._commit_sealed_buffer(session)
                self._advance_automatic(session)

    @staticmethod
    def _is_buffered_selling(session: GameSession) -> bool:
        return (
            session.state.phase == platform.Phase.SELLING.value
            and not session.rule_set.sequential_observable_selling
        )

    def _ensure_sealed_buffer(self, session: GameSession) -> SealedSaleBuffer:
        current = session.sealed_sale
        if current is not None and current.round == session.state.round:
            return current

        plans: dict[int, SealedPlayerPlan] = {}
        for player_id in range(session.rule_set.player_count):
            clone = session.state.clone()
            while (
                clone.phase == platform.Phase.SELLING.value
                and clone.current_player() != player_id
            ):
                actor = clone.current_player()
                if actor < 0:
                    break
                hold = _done_action(clone, actor)
                clone = _apply_player_action(session.rule_set, clone, actor, hold)
            plan = SealedPlayerPlan(state=clone)
            _auto_zero_sales(plan, player_id)
            plans[player_id] = plan
        current = SealedSaleBuffer(round=session.state.round, plans=plans)
        session.sealed_sale = current
        return current

    def _commit_sealed_buffer(self, session: GameSession) -> None:
        buffer = session.sealed_sale
        if buffer is None:
            return
        queues = {
            player: deque(plan.action_ids) for player, plan in buffer.plans.items()
        }
        state = session.state
        while state.phase == platform.Phase.SELLING.value:
            actor = state.current_player()
            if actor < 0 or not queues[actor]:
                raise RuntimeError("sealed selling plan is incomplete")
            action_id = queues[actor].popleft()
            state = _apply_player_action(session.rule_set, state, actor, action_id)
        if any(queue for queue in queues.values()):
            raise RuntimeError("sealed selling plan contains unused actions")
        session.state = state
        session.sealed_sale = None

    @staticmethod
    def _track_bid_before_action(session: GameSession, action_id: int) -> None:
        state = session.state
        if state.phase != platform.Phase.DEMAND.value or state.stage != "demand_bid":
            return
        namespace, _ordinal = session.rule_set.action_codec.decode(action_id)
        if namespace != "bid_level" or state._demand_token is None:
            return
        marker = tuple(state._demand_token)
        session.used_bid_markers.add(marker)
        session.outbid_markers.discard(marker)
        pile_id = state._demand_pile
        if pile_id is None:
            return
        pile = state.stockpiles[pile_id]
        if pile.occupying_player is not None and pile.occupying_token is not None:
            displaced = (pile.occupying_player, pile.occupying_token)
            session.used_bid_markers.add(displaced)
            session.outbid_markers.add(displaced)

    def _initialize_view_hashes(self, session: GameSession) -> None:
        session.view_hashes = [
            self._view_hash(self._build_view(session, player_id))
            for player_id in range(session.rule_set.player_count)
        ]

    def _refresh_view_revisions(self, session: GameSession) -> None:
        if not session.view_hashes:
            self._initialize_view_hashes(session)
            return
        hashes = [
            self._view_hash(self._build_view(session, player_id))
            for player_id in range(session.rule_set.player_count)
        ]
        for player_id, digest in enumerate(hashes):
            if digest != session.view_hashes[player_id]:
                session.view_revisions[player_id] += 1
        session.view_hashes = hashes

    @staticmethod
    def _view_hash(view: GameViewV1) -> str:
        payload = view.model_dump(mode="json")
        payload["revision"] = 0
        payload["chat"] = []
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _build_view(self, session: GameSession, viewer_id: int) -> GameViewV1:
        public_information, _unused = platform.observe_game_state(
            session.rule_set, session.state, None
        )
        private_state = session.state
        legal: Sequence[platform.LegalAction] = ()
        sealed = self._is_buffered_selling(session)
        plan: SealedPlayerPlan | None = None
        if sealed:
            plan = self._ensure_sealed_buffer(session).plans[viewer_id]
            # The final turn-order validator may advance through Movement and
            # into the next round after its last private commitment.  That
            # state is needed only to validate the plan; exposing it before
            # atomic settlement would reveal future cards and prices.  A
            # completed seat therefore waits on the unchanged authoritative
            # selling snapshot.
            private_state = session.state if plan.complete else plan.state
            if not plan.complete and private_state.current_player() == viewer_id:
                _private_info, legal = platform.observe_game_state(
                    session.rule_set, private_state, viewer_id
                )
            else:
                _private_info, _ = platform.observe_game_state(
                    session.rule_set, private_state, viewer_id
                )
        else:
            _private_info, observed_legal = platform.observe_game_state(
                session.rule_set, private_state, viewer_id
            )
            if session.state.current_player() == viewer_id:
                legal = observed_legal

        public = public_information.public_state
        phase = str(public["phase"])
        public_presentation = platform.get_presentation_state(
            session.rule_set, session.state, None
        )
        viewer_presentation = platform.get_presentation_state(
            session.rule_set, private_state, viewer_id
        )
        if sealed:
            active_player_id: int | None = None
            phase_step = "private_selling"
        else:
            active_player_id = public_presentation.current_actor
            # Raw engine stages include private multi-click progress (selected
            # Supply card, provisional bid pile, Action direction).  Only the
            # acting seat receives that progress through its pending decision.
            phase_step = "acting" if active_player_id == viewer_id else "waiting"

        companies = [
            CompanyV1(
                company_id=index,
                symbol=name[:1].upper(),
                name=name,
                price=int(public["prices"][name]),
                color=COMPANY_COLORS[index % len(COMPANY_COLORS)],
            )
            for index, name in enumerate(session.rule_set.company_names)
        ]
        markers = self._bid_markers(session)
        stockpiles = self._stockpiles(session, public, markers)
        players = self._players(
            session, viewer_id, public, markers, active_player_id, sealed, plan
        )
        private = self._private_view(session, viewer_id, private_state, _private_info)
        legal_actions = [
            self._legal_action(session, private_state, action) for action in legal
        ]
        pending = self._pending_decision(
            session,
            viewer_id,
            private_state,
            viewer_presentation,
            legal_actions,
            sealed,
            plan,
        )
        if not sealed:
            phase_step = pending.kind if active_player_id == viewer_id else "waiting"
        terminal = self._terminal_results(session) if session.state.is_terminal() else None

        options = LiteOptions(
            market_impact=session.configuration.impact,
            starting_share=session.configuration.hand,
            trading_fees=session.configuration.fees,
            dividends=session.configuration.dividend,
            sell_order=session.configuration.sell_order,
        )
        return GameViewV1(
            game_id=session.game_id,
            revision=session.view_revisions[viewer_id],
            configuration=ConfigurationV1(
                player_count=session.configuration.player_count,
                round_count=session.configuration.round_count,
                options=options,
            ),
            capabilities=CapabilitiesV1(
                market_impact=session.configuration.impact,
                starting_share=session.configuration.hand,
                trading_fees=session.configuration.fees,
                dividends=session.configuration.dividend,
                sequential_selling=session.configuration.sell_order,
            ),
            round=int(public["round"]),
            total_rounds=int(public["round_count"]),
            phase=phase,
            phase_step=phase_step,
            viewer=ViewerV1(
                player_id=viewer_id, name=session.player_names[viewer_id]
            ),
            active_player_id=active_player_id,
            companies=companies,
            stockpiles=stockpiles,
            players=players,
            private=private,
            pending_decision=pending,
            legal_actions=legal_actions,
            public_history=self._public_history(public_information.observable_history),
            recent_events=self._presentation_events(session.state),
            chat=list(session.chat),
            terminal_results=terminal,
        )

    def _bid_markers(self, session: GameSession) -> list[BidMarkerV1]:
        result: list[BidMarkerV1] = []
        presentation = platform.get_presentation_state(
            session.rule_set, session.state, None
        )
        active_token = presentation.demand_token
        occupied = {
            (marker.player_id, marker.marker_index): marker
            for marker in presentation.stockpile_markers
        }
        for player_id in range(session.rule_set.player_count):
            for marker_index in range(session.rule_set.meeples_per_player):
                marker = (player_id, marker_index)
                leading = occupied.get(marker)
                if leading is not None:
                    stockpile_id = leading.stockpile_id
                    bid = leading.bid_value
                    status = "locked" if leading.status == "locked" else "placed"
                elif marker in session.outbid_markers:
                    stockpile_id = None
                    bid = None
                    status = "rebidding" if active_token == marker else "outbid"
                else:
                    stockpile_id = None
                    bid = None
                    status = "available"
                result.append(
                    BidMarkerV1(
                        player_id=player_id,
                        marker_index=marker_index,
                        status=status,
                        stockpile_id=stockpile_id,
                        bid=bid,
                    )
                )
        return result

    def _stockpiles(
        self,
        session: GameSession,
        public: Mapping[str, Any],
        markers: Sequence[BidMarkerV1],
    ) -> list[StockpileV1]:
        result: list[StockpileV1] = []
        for pile in public["stockpiles"]:
            marker = next(
                (
                    item
                    for item in markers
                    if item.stockpile_id == int(pile["stockpile_id"])
                ),
                None,
            )
            visible_cards = [
                self._visible_card(session.rule_set, card)
                for card in pile["face_up_cards"]
            ]
            result.append(
                StockpileV1(
                    stockpile_id=int(pile["stockpile_id"]),
                    visible_cards=visible_cards,
                    hidden_cards=[
                        HiddenCardV1()
                        for _ in range(int(pile["face_down_count"]))
                    ],
                    marker=marker,
                    bid=(
                        None
                        if pile["bid_value"] is None
                        else int(pile["bid_value"])
                    ),
                    locked=bool(pile["locked"]),
                    purchaser_id=(
                        None
                        if pile["purchaser"] is None
                        else int(pile["purchaser"])
                    ),
                )
            )
        return result

    def _players(
        self,
        session: GameSession,
        viewer_id: int,
        public: Mapping[str, Any],
        markers: Sequence[BidMarkerV1],
        active_player_id: int | None,
        sealed: bool,
        viewer_plan: SealedPlayerPlan | None,
    ) -> list[PublicPlayerV1]:
        players: list[PublicPlayerV1] = []
        for player_id in range(session.rule_set.player_count):
            if sealed:
                if player_id == viewer_id:
                    status = (
                        "Waiting"
                        if viewer_plan is not None and viewer_plan.complete
                        else "Selling"
                    )
                else:
                    status = "Private selling"
            elif player_id == active_player_id:
                status = {
                    platform.Phase.SUPPLY.value: "Placing cards",
                    platform.Phase.DEMAND.value: "Choosing bid",
                    platform.Phase.ACTION.value: "Playing action",
                    platform.Phase.SELLING.value: "Selling",
                    platform.Phase.MOVEMENT.value: "Resolving market",
                }.get(session.state.phase, "Acting")
                token = platform.get_presentation_state(
                    session.rule_set, session.state, None
                ).demand_token
                if token and token in session.outbid_markers:
                    status = "Re-bidding"
            elif session.state.is_terminal():
                status = "Finished"
            else:
                status = "Waiting"
            players.append(
                PublicPlayerV1(
                    player_id=player_id,
                    name=session.player_names[player_id],
                    cash=int(public["cash"][player_id]),
                    active=(player_id == active_player_id),
                    status=status,
                    fee_debts=[int(value) for value in public["fee_debts"][player_id]],
                    bid_markers=[
                        item for item in markers if item.player_id == player_id
                    ],
                )
            )
        return players

    def _private_view(
        self,
        session: GameSession,
        viewer_id: int,
        state: platform.GameState,
        information: platform.InformationState,
    ) -> ViewerPrivateV1:
        holdings = [
            HoldingV1(
                company_id=company_id,
                company=name,
                regular=int(information.owned_stocks["regular"][name]),
                split=int(information.owned_stocks["split"][name]),
                represented=(
                    int(information.owned_stocks["regular"][name])
                    + 2 * int(information.owned_stocks["split"][name])
                ),
                price=state._company_price(company_id),
            )
            for company_id, name in enumerate(session.rule_set.company_names)
        ]
        known_pile_cards: list[KnownPileCardV1] = []
        for card in information.known_cards:
            if not card.location.startswith("stockpile:") or card.face_up:
                continue
            try:
                pile_id = int(card.location.split(":", 1)[1])
            except (ValueError, IndexError):
                continue
            known_pile_cards.append(
                KnownPileCardV1(
                    stockpile_id=pile_id,
                    card=self._visible_card(session.rule_set, card),
                )
            )
        action_cards = [
            ActionCardV1(effect=str(effect))
            for effect in information.acquired_actions
        ]
        return ViewerPrivateV1(
            hand=[
                self._visible_card(session.rule_set, card)
                for card in information.private_hand
            ],
            market_information=self._market_information(session, information),
            holdings=holdings,
            known_pile_cards=known_pile_cards,
            available_action_cards=action_cards,
        )

    def _market_information(
        self, session: GameSession, information: platform.InformationState
    ) -> list[MarketInformationSlotV1]:
        revealed = [tuple(pair) for pair in session.state.revealed_information]
        private = [tuple(pair) for pair in information.private_information]
        public = [tuple(pair) for pair in session.state.public_information]
        viewed = [tuple(pair) for pair in information.viewed_information_pairs]
        slots: list[MarketInformationSlotV1] = []
        seen: set[tuple[int, int | str]] = set()
        for source, visibility, pairs in (
            ("revealed", "public", revealed),
            ("dealt", "private", private),
            ("dealt", "public", public),
            ("viewed", "private", viewed),
        ):
            for pair in pairs:
                if pair in seen:
                    continue
                seen.add(pair)
                slots.append(
                    MarketInformationSlotV1(
                        visibility=visibility,
                        source=source,
                        card=self._information_card(session.rule_set, pair),
                    )
                )
        while len(slots) < session.rule_set.company_count:
            slots.append(
                MarketInformationSlotV1(
                    visibility="hidden", source="unknown", card=HiddenCardV1()
                )
            )
        return slots[: session.rule_set.company_count]

    def _legal_action(
        self,
        session: GameSession,
        state: platform.GameState,
        action: platform.LegalAction,
    ) -> LegalActionV1:
        namespace = action.action_type
        ordinal = int(action.payload.get("ordinal", 0))
        stage = state.stage
        control = "generic"
        target_id: str | None = None
        amount = action.amount
        placement: str | None = None
        preview: SalePreviewV1 | None = None
        if stage == "supply_card":
            control, target_id = "card", f"hand:{ordinal}"
        elif stage in {"supply_up_pile", "supply_down_pile"}:
            control, target_id = "stockpile", f"stockpile:{ordinal}"
            placement = "face_up" if stage == "supply_up_pile" else "face_down"
        elif stage == "demand_pile":
            control, target_id = "stockpile", f"stockpile:{ordinal}"
        elif stage == "demand_bid":
            control = "bid"
        elif stage in {"action_direction", "action_cramer_direction"}:
            control = "action_card"
            target_id = f"action:{'boom' if ordinal == 0 else 'bust'}"
        elif stage in {"action_company", "action_cramer_company"}:
            control, target_id = "company", f"company:{ordinal}"
        elif stage == "selling":
            control = "sell"
            preview = self._sale_preview(session.rule_set, state, action)
        elif stage == "dividend_claim":
            control = "dividend"
        elif namespace == "done":
            control = "continue"
        elif namespace == "company":
            control, target_id = "company", f"company:{ordinal}"
        return LegalActionV1(
            action_id=action.action_id,
            control=control,
            label=action.display_label,
            target_id=target_id,
            amount=amount,
            placement_visibility=placement,
            sale_preview=preview,
        )

    def _sale_preview(
        self,
        rule_set: platform.RuleSet,
        state: platform.GameState,
        action: platform.LegalAction,
    ) -> SalePreviewV1:
        authoritative = getattr(platform, "preview_sale_action", None)
        if authoritative is not None:
            preview = authoritative(
                rule_set, state, state.current_player(), action.action_id
            )
            return SalePreviewV1(
                company_id=preview.company_id,
                company=preview.company_name,
                quantity=preview.quantity_sold,
                unit_price=preview.unit_price,
                gross_value=preview.gross_value,
                resulting_regular=preview.resulting_regular,
                resulting_split=preview.resulting_split,
                resulting_represented=preview.resulting_represented,
            )
        player_id = state.current_player()
        company_id = state._selling_company
        regular, split = state._selling_holdings(player_id)
        before_regular = int(regular[company_id])
        before_split = int(split[company_id])
        clone = _apply_player_action(rule_set, state, player_id, action.action_id)
        after_regular, after_split = clone._selling_holdings(player_id)
        resulting_regular = int(after_regular[company_id])
        resulting_split = int(after_split[company_id])
        before = before_regular + 2 * before_split
        after = resulting_regular + 2 * resulting_split
        quantity = max(0, before - after)
        price = state._company_price(company_id)
        name = rule_set.company_names[company_id]
        return SalePreviewV1(
            company_id=company_id,
            company=name,
            quantity=quantity,
            unit_price=price,
            gross_value=quantity * price,
            resulting_regular=resulting_regular,
            resulting_split=resulting_split,
            resulting_represented=after,
        )

    @staticmethod
    def _pending_decision(
        session: GameSession,
        viewer_id: int,
        state: platform.GameState,
        presentation: platform.PresentationState,
        legal_actions: Sequence[LegalActionV1],
        sealed: bool,
        plan: SealedPlayerPlan | None,
    ) -> PendingDecisionV1:
        if session.state.is_terminal():
            return PendingDecisionV1(kind="terminal", prompt="Game complete")
        if sealed:
            if plan is None or plan.complete:
                return PendingDecisionV1(
                    kind="private_selling", prompt="Private selling in progress"
                )
            return PendingDecisionV1(
                kind="sell",
                prompt="Choose what to sell",
                company_id=presentation.selling_company,
                private_progress=presentation.selling_company,
                private_total=session.rule_set.company_count,
            )
        if session.state.current_player() != viewer_id or not legal_actions:
            return PendingDecisionV1(kind="waiting", prompt="Waiting for another player")
        stage = state.stage
        mapping = {
            "supply_card": ("supply_card", "Choose a Market Card"),
            "supply_up_pile": ("supply_face_up_pile", "Choose its face-up Stockpile"),
            "supply_down_pile": ("supply_face_down_pile", "Choose the face-down Stockpile"),
            "demand_pile": ("bid_pile", "Choose a Stockpile"),
            "demand_bid": ("bid_amount", "Choose a legal bid"),
            "action_direction": ("action_card", "Choose a Market Impact card"),
            "action_company": ("action_company", "Choose a company"),
            "selling": ("sell", "Choose what to sell"),
            "dividend_claim": ("dividend_claim", "Claim or waive the dividend"),
        }
        kind, prompt = mapping.get(stage, ("generic", "Choose an action"))
        return PendingDecisionV1(
            kind=kind,
            prompt=prompt,
            selected_card_index=(
                int(presentation.supply_choice)
                if presentation.supply_choice is not None
                else None
            ),
            selected_stockpile_id=(
                int(presentation.supply_up_pile)
                if presentation.supply_up_pile is not None
                else (
                    int(presentation.demand_pile)
                    if presentation.demand_pile is not None
                    else None
                )
            ),
            selected_action_effect=presentation.selected_direction,
            company_id=(presentation.selling_company if stage == "selling" else None),
        )

    @staticmethod
    def _public_history(
        records: Sequence[Mapping[str, Any]],
    ) -> list[PublicHistoryEntryV1]:
        result: list[PublicHistoryEntryV1] = []
        for index, record in enumerate(records[-MAX_PUBLIC_ITEMS:]):
            sequence = int(record.get("sequence", index + 1))
            actor_raw = int(record.get("player", -1))
            summary = str(record.get("label") or "Placed market cards")
            sale_totals: dict[str, dict[str, int]] | None = None
            raw_sales = record.get("sales")
            if isinstance(raw_sales, Mapping):
                sale_totals = {
                    str(player): {
                        str(company): int(quantity)
                        for company, quantity in company_sales.items()
                    }
                    for player, company_sales in raw_sales.items()
                    if isinstance(company_sales, Mapping)
                }
            result.append(
                PublicHistoryEntryV1(
                    sequence=sequence,
                    phase=str(record.get("phase", "")),
                    actor_id=actor_raw if actor_raw >= 0 else None,
                    summary=summary,
                    sale_totals=sale_totals,
                )
            )
        return result

    @staticmethod
    def _presentation_events(state: platform.GameState) -> list[PublicEventV1]:
        getter = getattr(platform, "get_presentation_events", None)
        if getter is None:
            return []
        try:
            raw_events = getter(state.rule_set, state, since_sequence=0)
        except TypeError:
            raw_events = getter(state.rule_set, state)
        events: list[PublicEventV1] = []
        for index, raw in enumerate(raw_events[-MAX_PUBLIC_ITEMS:]):
            data = asdict(raw) if is_dataclass(raw) else dict(raw)
            event_id = int(
                data.get("presentation_sequence", data.get("sequence", index + 1))
            )
            events.append(
                PublicEventV1(
                    event_id=event_id,
                    event_type=str(data.get("event_type", data.get("cause", "market"))),
                    cause=_optional_str(data.get("cause")),
                    round=int(data.get("round", state.round)),
                    description=str(data.get("description", "Market moved")),
                    company_id=_optional_int(data.get("company_id")),
                    company=_optional_str(
                        data.get("company", data.get("company_name"))
                    ),
                    prior_price=_optional_int(data.get("prior_price")),
                    requested_delta=_optional_int(data.get("requested_delta")),
                    actual_delta=_optional_int(data.get("actual_delta")),
                    resulting_price=_optional_int(data.get("resulting_price")),
                    forecast=_optional_forecast(data.get("forecast")),
                    effect=_optional_str(data.get("effect")),
                    actor_id=_optional_int(data.get("actor_id")),
                )
            )
        return events

    def _terminal_results(self, session: GameSession) -> TerminalResultsV1:
        result = platform.score_game(session.rule_set, session.state)
        details_getter = getattr(platform, "terminal_liquidation_details", None)
        details = (
            details_getter(session.rule_set, session.state)
            if details_getter is not None
            else ()
        )
        detail_by_player = {detail.player_id: detail for detail in details}
        distinct = sorted(set(result.final_cash_by_player.values()), reverse=True)
        ranks = {value: distinct.index(value) + 1 for value in distinct}
        players: list[TerminalPlayerV1] = []
        for player_id in range(session.rule_set.player_count):
            detail = detail_by_player.get(player_id)
            if detail is not None:
                lines = [
                    LiquidationLineV1(
                        company_id=line.company_id,
                        company=line.company_name,
                        represented_shares=line.represented_shares,
                        unit_price=line.unit_price,
                        value=line.value,
                    )
                    for line in detail.companies
                ]
                liquidation_value = int(detail.liquidation_value)
                final_cash = int(detail.final_cash)
                rank = int(detail.rank)
                winner = bool(detail.winner)
            else:
                lines = []
                for company_id, name in enumerate(session.rule_set.company_names):
                    represented = session.state._represented_shares(
                        player_id, company_id
                    )
                    price = session.state._company_price(company_id)
                    lines.append(
                        LiquidationLineV1(
                            company_id=company_id,
                            company=name,
                            represented_shares=represented,
                            unit_price=price,
                            value=represented * price,
                        )
                    )
                liquidation_value = int(result.liquidation_values[player_id])
                final_cash = int(result.final_cash_by_player[player_id])
                rank = ranks[final_cash]
                winner = player_id in result.winner_ids
            players.append(
                TerminalPlayerV1(
                    player_id=player_id,
                    player_name=session.player_names[player_id],
                    cash_before_liquidation=int(session.state.players[player_id].cash),
                    liquidation_value=liquidation_value,
                    final_cash=final_cash,
                    rank=rank,
                    winner=winner,
                    liquidation=lines,
                )
            )
        return TerminalResultsV1(
            players=players, winner_ids=[int(value) for value in result.winner_ids]
        )

    @staticmethod
    def _visible_card(
        rule_set: platform.RuleSet, card: platform.Card | Mapping[str, Any]
    ) -> StockCardV1 | TradingFeeCardV1 | ActionCardV1:
        if isinstance(card, Mapping):
            kind = str(card.get("card_type", ""))
            company_id = card.get("company_id")
            value = card.get("value")
            effect = card.get("effect")
        else:
            kind = card.card_type
            company_id = card.company_id
            value = card.value
            effect = card.effect
        if kind == platform.CardType.STOCK.value:
            assert company_id is not None
            return StockCardV1(
                company_id=int(company_id),
                company=rule_set.company_names[int(company_id)],
                quantity=int(value or 1),
            )
        if kind == platform.CardType.TRADING_FEE.value:
            return TradingFeeCardV1(amount=abs(int(value or 0)))
        return ActionCardV1(effect=str(effect or value or "Market Impact"))

    @staticmethod
    def _information_card(
        rule_set: platform.RuleSet, pair: tuple[int, int | str]
    ) -> InformationCardV1:
        company_id, forecast = pair
        normalized_forecast: int | str = (
            "DIVIDEND" if str(forecast).upper() == "DIVIDEND" else int(forecast)
        )
        return InformationCardV1(
            company_id=int(company_id),
            company=rule_set.company_names[int(company_id)],
            forecast=normalized_forecast,
        )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_forecast(value: Any) -> int | str | None:
    if value is None:
        return None
    return "DIVIDEND" if str(value).upper() == "DIVIDEND" else int(value)
