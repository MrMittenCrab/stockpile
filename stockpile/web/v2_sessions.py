"""Human-versus-computer browser sessions without changing the game tree."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, is_dataclass
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
from .policy import ComputerPolicy, RandomComputerPolicy
from .sessions import COMPANY_PRESENTATION, SessionError
from .v2_schemas import (
    ActionCardV2,
    BidMarkerV2,
    CompanyV2,
    ComputerPublicPlayerV2,
    ConfigurationV2,
    CreateGameRequestV2,
    CreateGameResponseV2,
    DecisionBatchV2,
    DemandDecisionBatchV2,
    DemandDecisionPlanV2,
    GameViewV2,
    HiddenCardV2,
    HoldingV2,
    HumanPublicPlayerV2,
    InformationCardV2,
    LegalActionV2,
    LiquidationLineV2,
    LiteOptionsV2,
    MarketInformationSlotV2,
    MarketImpactDecisionBatchV2,
    MarketImpactDecisionPlanV2,
    PendingDecisionV2,
    PileCardV2,
    PresentationCheckpointV2,
    PublicEventV2,
    RememberedCardV2,
    SalePreviewV2,
    StockCardV2,
    StockpileBidV2,
    StockpileV2,
    SupplyBatchV2,
    SupplyCardV2,
    SupplyPlacementV2,
    SupplyPlanV2,
    TerminalPlayerV2,
    TerminalResultsV2,
    TradingFeeCardV2,
    ViewerPrivateV2,
    ViewerV2,
    VisibleCardV2,
)


HUMAN_ID = 0
COMPUTER_ID = 1
PLAYER_NAMES = ("YOU", "COMPUTER")
CHANCE_SEED_MASK = 0x4348414E43455F32
POLICY_SEED_MASK = 0x504F4C4943595F32
MAX_AUTOMATIC_STEPS = 20_000
MAX_PUBLIC_EVENTS = 80


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


@dataclass(frozen=True, slots=True)
class OrderedPileCard:
    card_id: int
    face_up: bool


@dataclass(frozen=True, slots=True)
class CapturedPileCard:
    card: platform.Card
    face_up: bool


@dataclass(frozen=True, slots=True)
class SupplyPlanRecord:
    plan_id: str
    action_ids: tuple[int, int, int]
    placements: tuple[SupplyPlacementV2, SupplyPlacementV2]


@dataclass(frozen=True, slots=True)
class DecisionPlanRecord:
    plan_id: str
    kind: str
    action_ids: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ResolvedAuctionSnapshot:
    round: int
    stockpiles: tuple[StockpileV2, ...]
    markers: tuple[BidMarkerV2, ...]


@dataclass(slots=True)
class SealedPlayerPlanV2:
    state: platform.GameState
    action_ids: list[int] = field(default_factory=list)
    complete: bool = False


@dataclass(slots=True)
class SealedSaleBufferV2:
    round: int
    plans: dict[int, SealedPlayerPlanV2]


@dataclass(slots=True)
class V2GameSession:
    game_id: str
    configuration: interface.GameConfig
    seed: int
    state: platform.GameState
    token_digest: str
    chance_rng: random.Random
    policy_rng: random.Random
    plan_secret: bytes
    lock: threading.RLock = field(default_factory=threading.RLock)
    revision: int = 0
    ordered_piles: dict[int, list[OrderedPileCard]] = field(default_factory=dict)
    supply_plans: dict[str, SupplyPlanRecord] = field(default_factory=dict)
    decision_plans: dict[str, DecisionPlanRecord] = field(default_factory=dict)
    used_bid_markers: set[tuple[int, int]] = field(default_factory=set)
    outbid_markers: set[tuple[int, int]] = field(default_factory=set)
    bid_tracking_round: int = 1
    sealed_sale: SealedSaleBufferV2 | None = None
    checkpoint: PresentationCheckpointV2 | None = None
    demand_cash_before: tuple[int, int] | None = None
    cash_deltas: tuple[int, int] | None = None
    last_end_position: int = 0
    position_delta: int | None = None
    resolved_auction: ResolvedAuctionSnapshot | None = None
    closed: bool = False

    @property
    def rule_set(self) -> platform.RuleSet:
        return self.configuration.rule_set


class V2SessionStore:
    """Process-local fixed-product sessions, isolated from the compatible V1 store."""

    def __init__(self, policy: ComputerPolicy | None = None) -> None:
        self._sessions: dict[str, V2GameSession] = {}
        self._lock = threading.RLock()
        self.policy = policy or RandomComputerPolicy()

    def create(
        self, request: CreateGameRequestV2
    ) -> tuple[V2GameSession, CreateGameResponseV2]:
        options = request.options
        configuration = interface.resolve_configuration(
            interface.ConfigurationMode.LITE,
            player_count=2,
            round_count=6,
            hand=False,
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
        state = platform.GameState(configuration.game)
        game_id = uuid.uuid4().hex
        token = secrets.token_urlsafe(32)
        session = V2GameSession(
            game_id=game_id,
            configuration=configuration,
            seed=seed,
            state=state,
            token_digest=_token_digest(token),
            chance_rng=random.Random(seed ^ CHANCE_SEED_MASK),
            policy_rng=random.Random(seed ^ POLICY_SEED_MASK),
            plan_secret=secrets.token_bytes(32),
            ordered_piles={index: [] for index in range(configuration.rule_set.stockpile_count)},
            bid_tracking_round=state.round,
        )
        session.last_end_position = self._position_value(state, HUMAN_ID)
        with session.lock:
            self._advance_automatic(session)
        with self._lock:
            self._sessions[game_id] = session
        return session, CreateGameResponseV2(
            game_id=game_id,
            game_url=f"/game/{game_id}#seat={token}",
        )

    def get(self, game_id: str) -> V2GameSession:
        with self._lock:
            session = self._sessions.get(game_id)
        if session is None:
            raise SessionError(404, "game_not_found", "Game not found")
        return session

    def authenticate(self, game_id: str, token: str | None) -> V2GameSession:
        if not token:
            raise SessionError(401, "invalid_seat_token", "A valid seat token is required")
        session = self.get(game_id)
        if not hmac.compare_digest(_token_digest(token), session.token_digest):
            raise SessionError(401, "invalid_seat_token", "A valid seat token is required")
        return session

    def view(self, session: V2GameSession) -> GameViewV2:
        with session.lock:
            self._validate_open(session)
            return self._build_view(session)

    def act(
        self,
        session: V2GameSession,
        *,
        action_id: int,
        expected_revision: int,
    ) -> GameViewV2:
        with session.lock:
            self._validate_open(session)
            self._validate_revision(session, expected_revision)
            self._validate_action_window(session)
            if self._is_buffered_selling(session):
                self._act_in_sealed_plan(session, action_id)
            else:
                if session.state.current_player() != HUMAN_ID:
                    raise SessionError(409, "turn_conflict", "COMPUTER must act first")
                if session.state.phase == platform.Phase.SUPPLY.value:
                    raise SessionError(
                        422, "supply_plan_required", "Submit the complete Supply plan"
                    )
                if session.state.stage in {
                    "demand_pile",
                    "demand_bid",
                    "action_direction",
                    "action_company",
                }:
                    raise SessionError(
                        422,
                        "decision_plan_required",
                        "Submit the complete decision plan",
                    )
                _information, observed_legal = platform.observe_game_state(
                    session.rule_set, session.state, HUMAN_ID
                )
                projected_legal = self._deduplicate_sell_actions(
                    session, session.state, observed_legal
                )
                legal_ids = {int(action.action_id) for action in projected_legal}
                if action_id not in legal_ids:
                    raise SessionError(422, "illegal_action", "Action is not legal now")
                self._track_bid_before_action(session, action_id)
                self._apply_and_observe(session, HUMAN_ID, action_id)
            if session.checkpoint is None:
                self._advance_automatic(session)
            session.revision += 1
            return self._build_view(session)

    def supply(
        self,
        session: V2GameSession,
        *,
        plan_id: str,
        expected_revision: int,
    ) -> GameViewV2:
        with session.lock:
            self._validate_open(session)
            self._validate_revision(session, expected_revision)
            self._validate_action_window(session)
            if self._is_buffered_selling(session):
                raise SessionError(422, "invalid_supply_plan", "Supply is not active")
            if (
                session.state.phase != platform.Phase.SUPPLY.value
                or session.state.stage != "supply_card"
                or session.state.current_player() != HUMAN_ID
            ):
                raise SessionError(409, "turn_conflict", "A Supply batch is not available")
            self._supply_batch(session)
            plan = session.supply_plans.get(plan_id)
            if plan is None:
                raise SessionError(422, "invalid_supply_plan", "Supply plan is not legal now")

            candidate = session.state
            for action_id in plan.action_ids:
                candidate = _apply_player_action(
                    session.rule_set, candidate, HUMAN_ID, action_id
                )
            session.state = candidate
            self._append_supply_placements(session, plan)
            session.supply_plans.clear()
            self._after_transition(
                session,
                before_phase=platform.Phase.SUPPLY.value,
                before_round=session.state.round,
            )
            if session.checkpoint is None:
                self._advance_automatic(session)
            session.revision += 1
            return self._build_view(session)

    def decide(
        self,
        session: V2GameSession,
        *,
        plan_id: str,
        expected_revision: int,
    ) -> GameViewV2:
        """Commit one browser decision while retaining its canonical engine steps."""

        with session.lock:
            self._validate_open(session)
            self._validate_revision(session, expected_revision)
            self._validate_action_window(session)
            if self._is_buffered_selling(session):
                raise SessionError(
                    422, "invalid_decision_plan", "A decision plan is not active"
                )
            batch = self._decision_batch(session)
            plan = session.decision_plans.get(plan_id)
            if batch is None or plan is None:
                raise SessionError(
                    422, "invalid_decision_plan", "Decision plan is not legal now"
                )

            candidate = session.state
            for action_id in plan.action_ids:
                candidate = _apply_player_action(
                    session.rule_set, candidate, HUMAN_ID, action_id
                )

            for action_id in plan.action_ids:
                self._track_bid_before_action(session, action_id)
                self._apply_and_observe(session, HUMAN_ID, action_id)
            session.decision_plans.clear()
            if session.checkpoint is None:
                self._advance_automatic(session)
            session.revision += 1
            return self._build_view(session)

    def acknowledge(
        self,
        session: V2GameSession,
        *,
        checkpoint_id: str,
        expected_revision: int,
    ) -> GameViewV2:
        with session.lock:
            self._validate_open(session)
            self._validate_revision(session, expected_revision)
            checkpoint = session.checkpoint
            if checkpoint is None or not hmac.compare_digest(
                checkpoint.checkpoint_id, checkpoint_id
            ):
                raise SessionError(
                    409, "stale_checkpoint", "The result checkpoint has changed"
                )
            checkpoint_kind = checkpoint.kind
            session.checkpoint = None
            session.cash_deltas = None
            session.position_delta = None
            if checkpoint_kind == "round_result":
                session.resolved_auction = None
            self._advance_automatic(session)
            session.revision += 1
            return self._build_view(session)

    def resign(
        self,
        session: V2GameSession,
        *,
        expected_revision: int,
    ) -> None:
        """Close and remove a session, excluding already queued operations."""

        with session.lock:
            self._validate_open(session)
            self._validate_revision(session, expected_revision)
            session.closed = True
            session.revision += 1
        with self._lock:
            if self._sessions.get(session.game_id) is session:
                del self._sessions[session.game_id]

    @staticmethod
    def _validate_open(session: V2GameSession) -> None:
        if session.closed:
            raise SessionError(409, "game_closed", "The game has been closed")

    @staticmethod
    def _validate_revision(session: V2GameSession, expected_revision: int) -> None:
        if expected_revision != session.revision:
            raise SessionError(
                409, "stale_revision", "The game view changed; refresh before acting"
            )

    @staticmethod
    def _validate_action_window(session: V2GameSession) -> None:
        if session.checkpoint is not None:
            raise SessionError(409, "checkpoint_pending", "A result must be acknowledged")
        if session.state.is_terminal():
            raise SessionError(409, "game_finished", "The game has ended")

    def _advance_automatic(self, session: V2GameSession) -> None:
        for _step in range(MAX_AUTOMATIC_STEPS):
            if session.checkpoint is not None or session.state.is_terminal():
                return
            self._record_demand_entry(session)
            if session.state.is_chance_node():
                before_phase = session.state.phase
                before_round = session.state.round
                self._apply_chance(session)
                self._after_transition(session, before_phase, before_round)
                continue
            if self._is_buffered_selling(session):
                buffer = self._ensure_sealed_buffer(session)
                self._complete_computer_sealed_plan(session, buffer)
                if buffer.plans[HUMAN_ID].complete and buffer.plans[COMPUTER_ID].complete:
                    before_phase = session.state.phase
                    before_round = session.state.round
                    self._commit_sealed_buffer(session)
                    self._after_transition(session, before_phase, before_round)
                    continue
                return
            actor = int(session.state.current_player())
            if actor == COMPUTER_ID:
                information, legal = platform.observe_game_state(
                    session.rule_set, session.state, COMPUTER_ID
                )
                if not legal:
                    raise RuntimeError("COMPUTER turn has no legal action")
                action_id = int(
                    self.policy.choose_action(information, tuple(legal), session.policy_rng)
                )
                if action_id not in {action.action_id for action in legal}:
                    raise RuntimeError("computer policy returned an illegal action")
                self._track_bid_before_action(session, action_id)
                self._apply_and_observe(session, COMPUTER_ID, action_id)
                continue
            if actor == HUMAN_ID:
                forced = self._forced_sale_action(session.state, HUMAN_ID)
                if forced is not None:
                    self._apply_and_observe(session, HUMAN_ID, forced)
                    continue
                return
            raise RuntimeError(f"unexpected non-player state {actor}")
        raise RuntimeError("automatic browser progression exceeded its safety limit")

    def _apply_chance(self, session: V2GameSession) -> None:
        outcomes = session.state.chance_outcomes()
        if not outcomes:
            raise RuntimeError("chance state has no outcomes")
        threshold = session.chance_rng.random()
        cumulative = 0.0
        selected = int(outcomes[-1][0])
        for action_id, probability in outcomes:
            cumulative += float(probability)
            if threshold <= cumulative:
                selected = int(action_id)
                break
        before_ids = {
            pile.stockpile_id: {
                card.card_id for card in pile.face_up_cards + pile.face_down_cards
            }
            for pile in session.state.stockpiles
        }
        session.state.apply_action(selected)
        for pile in session.state.stockpiles:
            prior = before_ids[pile.stockpile_id]
            for card in pile.face_up_cards + pile.face_down_cards:
                if card.card_id not in prior:
                    session.ordered_piles[pile.stockpile_id].append(
                        OrderedPileCard(card_id=card.card_id, face_up=bool(card.face_up))
                    )

    def _apply_and_observe(
        self, session: V2GameSession, player_id: int, action_id: int
    ) -> None:
        before = session.state
        before_phase = before.phase
        before_round = before.round
        captured_cards: dict[int, tuple[CapturedPileCard, ...]] | None = None
        if (
            before.phase == platform.Phase.DEMAND.value
            and before.stage == "demand_bid"
        ):
            captured_cards = self._capture_pile_cards(session, before)
        supply_record = self._pending_supply_record(session, before, action_id)
        session.state = _apply_player_action(
            session.rule_set, before, player_id, action_id
        )
        if supply_record is not None:
            for pile_id, card_id, face_up in supply_record:
                session.ordered_piles[pile_id].append(
                    OrderedPileCard(card_id=card_id, face_up=face_up)
                )
        self._after_transition(
            session,
            before_phase,
            before_round,
            captured_cards=captured_cards,
        )

    @staticmethod
    def _pending_supply_record(
        session: V2GameSession, state: platform.GameState, action_id: int
    ) -> tuple[tuple[int, int, bool], tuple[int, int, bool]] | None:
        if state.phase != platform.Phase.SUPPLY.value or state.stage != "supply_down_pile":
            return None
        namespace, down_pile = session.rule_set.action_codec.decode(action_id)
        if namespace != "pile" or state._supply_choice is None or state._supply_up_pile is None:
            return None
        hand = state._hands[state.current_player()][:2]
        if len(hand) != 2:
            return None
        up_card = hand[state._supply_choice]
        down_card = hand[1 - state._supply_choice]
        return (
            (int(state._supply_up_pile), up_card.card_id, not state._supply_both_down),
            (int(down_pile), down_card.card_id, False),
        )

    def _after_transition(
        self,
        session: V2GameSession,
        before_phase: str,
        before_round: int,
        *,
        captured_cards: Mapping[int, Sequence[CapturedPileCard]] | None = None,
    ) -> None:
        self._record_demand_entry(session)
        if before_phase == platform.Phase.DEMAND.value and session.state.phase != before_phase:
            before_cash = session.demand_cash_before or tuple(
                int(player.cash) for player in session.state.players
            )
            now = tuple(int(player.cash) for player in session.state.players)
            session.cash_deltas = (now[0] - before_cash[0], now[1] - before_cash[1])
            session.demand_cash_before = None
            if captured_cards is None:
                raise RuntimeError("Demand resolved without a safe Stockpile snapshot")
            resolved_information, _legal = platform.observe_game_state(
                session.rule_set, session.state, HUMAN_ID
            )
            known = {
                card.card_id: card for card in resolved_information.known_cards
            }
            markers = self._bid_markers(session)
            stockpiles: list[StockpileV2] = []
            for pile in session.state.stockpiles:
                marker = next(
                    (
                        item
                        for item in markers
                        if item.stockpile_id == pile.stockpile_id
                        and item.status in {"placed", "locked"}
                    ),
                    None,
                )
                bid = (
                    StockpileBidV2(
                        player_id=marker.player_id,
                        marker_index=marker.marker_index,
                        amount_thousands=int(marker.bid_thousands or 0),
                    )
                    if marker is not None
                    else None
                )
                stockpiles.append(
                    StockpileV2(
                        stockpile_id=pile.stockpile_id,
                        cards_bottom_to_top=[
                            self._captured_pile_card(
                                session.rule_set, item, known
                            )
                            for item in captured_cards.get(pile.stockpile_id, ())
                        ],
                        bid=bid,
                        locked=bool(pile.locked),
                        purchaser_id=pile.purchaser,
                        resolved=True,
                    )
                )
            session.resolved_auction = ResolvedAuctionSnapshot(
                round=before_round,
                stockpiles=tuple(stockpiles),
                markers=tuple(markers),
            )
            self._clear_ordered_piles(session)
            session.checkpoint = self._checkpoint("demand_result", before_round)
            return
        if session.state.round != before_round or (
            session.state.is_terminal() and before_phase != platform.Phase.TERMINAL.value
        ):
            current_position = self._position_value(session.state, HUMAN_ID)
            session.position_delta = current_position - session.last_end_position
            session.last_end_position = current_position
            session.cash_deltas = None
            self._clear_ordered_piles(session)
            session.checkpoint = self._checkpoint("round_result", before_round)
        if session.bid_tracking_round != session.state.round:
            session.bid_tracking_round = session.state.round
            session.used_bid_markers.clear()
            session.outbid_markers.clear()

    @staticmethod
    def _checkpoint(kind: str, round_number: int) -> PresentationCheckpointV2:
        return PresentationCheckpointV2(
            checkpoint_id=secrets.token_urlsafe(18),
            kind=kind,
            round=round_number,
        )

    @staticmethod
    def _record_demand_entry(session: V2GameSession) -> None:
        if (
            session.state.phase == platform.Phase.DEMAND.value
            and session.demand_cash_before is None
        ):
            session.demand_cash_before = tuple(
                int(player.cash) for player in session.state.players
            )  # type: ignore[assignment]

    @staticmethod
    def _clear_ordered_piles(session: V2GameSession) -> None:
        for cards in session.ordered_piles.values():
            cards.clear()

    @staticmethod
    def _position_value(state: platform.GameState, player_id: int) -> int:
        return sum(
            state._represented_shares(player_id, company_id)
            * state._company_price(company_id)
            for company_id in range(state.rule_set.company_count)
        )

    @staticmethod
    def _is_buffered_selling(session: V2GameSession) -> bool:
        return (
            session.state.phase == platform.Phase.SELLING.value
            and not session.rule_set.sequential_observable_selling
        )

    @staticmethod
    def _done_action(state: platform.GameState, player_id: int) -> int:
        for action_id in state.legal_actions(player_id):
            namespace, _ordinal = state.rule_set.action_codec.decode(int(action_id))
            if namespace == "done":
                return int(action_id)
        raise RuntimeError("selling state does not offer a hold action")

    def _ensure_sealed_buffer(
        self, session: V2GameSession
    ) -> SealedSaleBufferV2:
        current = session.sealed_sale
        if current is not None and current.round == session.state.round:
            return current
        plans: dict[int, SealedPlayerPlanV2] = {}
        for player_id in (HUMAN_ID, COMPUTER_ID):
            clone = session.state.clone()
            while (
                clone.phase == platform.Phase.SELLING.value
                and clone.current_player() != player_id
            ):
                actor = int(clone.current_player())
                if actor < 0:
                    break
                clone = _apply_player_action(
                    session.rule_set, clone, actor, self._done_action(clone, actor)
                )
            plan = SealedPlayerPlanV2(
                state=clone,
                complete=not (
                    clone.phase == platform.Phase.SELLING.value
                    and clone.current_player() == player_id
                ),
            )
            self._auto_forced_sealed_sales(session, plan, player_id)
            plans[player_id] = plan
        current = SealedSaleBufferV2(round=session.state.round, plans=plans)
        session.sealed_sale = current
        return current

    @staticmethod
    def _forced_sale_action(
        state: platform.GameState, player_id: int
    ) -> int | None:
        if (
            state.phase != platform.Phase.SELLING.value
            or state.stage != "selling"
            or state.current_player() != player_id
        ):
            return None
        actions = [int(action) for action in state.legal_actions(player_id)]
        if len(actions) != 1:
            return None
        namespace, _ordinal = state.rule_set.action_codec.decode(actions[0])
        return actions[0] if namespace == "done" else None

    def _auto_forced_sealed_sales(
        self,
        session: V2GameSession,
        plan: SealedPlayerPlanV2,
        player_id: int,
    ) -> None:
        while not plan.complete:
            action_id = self._forced_sale_action(plan.state, player_id)
            if action_id is None:
                break
            plan.action_ids.append(action_id)
            plan.state = _apply_player_action(
                session.rule_set, plan.state, player_id, action_id
            )
            plan.complete = not (
                plan.state.phase == platform.Phase.SELLING.value
                and plan.state.current_player() == player_id
            )

    def _complete_computer_sealed_plan(
        self, session: V2GameSession, buffer: SealedSaleBufferV2
    ) -> None:
        plan = buffer.plans[COMPUTER_ID]
        while not plan.complete:
            if (
                plan.state.phase != platform.Phase.SELLING.value
                or plan.state.current_player() != COMPUTER_ID
            ):
                plan.complete = True
                return
            information, legal = platform.observe_game_state(
                session.rule_set, plan.state, COMPUTER_ID
            )
            if not legal:
                raise RuntimeError("COMPUTER sealed-selling plan has no legal action")
            action_id = int(
                self.policy.choose_action(information, tuple(legal), session.policy_rng)
            )
            if action_id not in {action.action_id for action in legal}:
                raise RuntimeError("computer policy returned an illegal sealed-sale action")
            plan.action_ids.append(action_id)
            plan.state = _apply_player_action(
                session.rule_set, plan.state, COMPUTER_ID, action_id
            )
            plan.complete = not (
                plan.state.phase == platform.Phase.SELLING.value
                and plan.state.current_player() == COMPUTER_ID
            )

    def _act_in_sealed_plan(self, session: V2GameSession, action_id: int) -> None:
        buffer = self._ensure_sealed_buffer(session)
        plan = buffer.plans[HUMAN_ID]
        if plan.complete:
            raise SessionError(
                409, "turn_conflict", "Your private selling plan is complete"
            )
        if plan.state.current_player() != HUMAN_ID:
            raise SessionError(409, "turn_conflict", "YOU cannot act now")
        _information, observed_legal = platform.observe_game_state(
            session.rule_set, plan.state, HUMAN_ID
        )
        projected_legal = self._deduplicate_sell_actions(
            session, plan.state, observed_legal
        )
        legal_ids = {int(action.action_id) for action in projected_legal}
        if action_id not in legal_ids:
            raise SessionError(422, "illegal_action", "Action is not legal now")
        plan.action_ids.append(action_id)
        plan.state = _apply_player_action(
            session.rule_set, plan.state, HUMAN_ID, action_id
        )
        plan.complete = not (
            plan.state.phase == platform.Phase.SELLING.value
            and plan.state.current_player() == HUMAN_ID
        )
        self._auto_forced_sealed_sales(session, plan, HUMAN_ID)
        self._complete_computer_sealed_plan(session, buffer)
        if all(candidate.complete for candidate in buffer.plans.values()):
            before_phase = session.state.phase
            before_round = session.state.round
            self._commit_sealed_buffer(session)
            self._after_transition(session, before_phase, before_round)

    def _commit_sealed_buffer(self, session: V2GameSession) -> None:
        buffer = session.sealed_sale
        if buffer is None:
            return
        queues = {
            player: deque(plan.action_ids) for player, plan in buffer.plans.items()
        }
        state = session.state
        while state.phase == platform.Phase.SELLING.value:
            actor = int(state.current_player())
            if actor < 0 or not queues[actor]:
                raise RuntimeError("sealed selling plan is incomplete")
            state = _apply_player_action(
                session.rule_set, state, actor, queues[actor].popleft()
            )
        if any(queue for queue in queues.values()):
            raise RuntimeError("sealed selling plan contains unused actions")
        session.state = state
        session.sealed_sale = None

    @staticmethod
    def _track_bid_before_action(session: V2GameSession, action_id: int) -> None:
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

    def _supply_batch(self, session: V2GameSession) -> SupplyBatchV2 | None:
        state = session.state
        if (
            session.checkpoint is not None
            or state.phase != platform.Phase.SUPPLY.value
            or state.stage != "supply_card"
            or state.current_player() != HUMAN_ID
        ):
            session.supply_plans.clear()
            return None
        hand = state._hands[HUMAN_ID][:2]
        if len(hand) != 2:
            raise RuntimeError("Supply batch does not contain exactly two cards")
        refs = {card.card_id: self._card_ref(session, card.card_id) for card in hand}
        public_cards = [
            SupplyCardV2(
                card_ref=refs[card.card_id],
                card=self._visible_card(session.rule_set, card),
            )
            for card in hand
        ]
        generated: dict[str, SupplyPlanRecord] = {}
        public_plans: list[SupplyPlanV2] = []
        _info, card_actions = platform.observe_game_state(
            session.rule_set, state, HUMAN_ID
        )
        for card_action in card_actions:
            first = _apply_player_action(
                session.rule_set, state, HUMAN_ID, card_action.action_id
            )
            selected = int(first._supply_choice or 0)
            up_card = hand[selected]
            down_card = hand[1 - selected]
            _info, up_actions = platform.observe_game_state(
                session.rule_set, first, HUMAN_ID
            )
            for up_action in up_actions:
                second = _apply_player_action(
                    session.rule_set, first, HUMAN_ID, up_action.action_id
                )
                _namespace, up_pile = session.rule_set.action_codec.decode(
                    up_action.action_id
                )
                _info, down_actions = platform.observe_game_state(
                    session.rule_set, second, HUMAN_ID
                )
                for down_action in down_actions:
                    _namespace, down_pile = session.rule_set.action_codec.decode(
                        down_action.action_id
                    )
                    placements = (
                        SupplyPlacementV2(
                            card_ref=refs[up_card.card_id],
                            stockpile_id=int(up_pile),
                            visibility="face_up",
                        ),
                        SupplyPlacementV2(
                            card_ref=refs[down_card.card_id],
                            stockpile_id=int(down_pile),
                            visibility="face_down",
                        ),
                    )
                    action_ids = (
                        int(card_action.action_id),
                        int(up_action.action_id),
                        int(down_action.action_id),
                    )
                    plan_id = self._plan_id(session, action_ids, placements)
                    record = SupplyPlanRecord(plan_id, action_ids, placements)
                    generated[plan_id] = record
                    public_plans.append(
                        SupplyPlanV2(plan_id=plan_id, placements=list(placements))
                    )
        session.supply_plans = generated
        return SupplyBatchV2(cards=public_cards, plans=public_plans)

    def _decision_batch(self, session: V2GameSession) -> DecisionBatchV2 | None:
        state = session.state
        if (
            session.checkpoint is not None
            or state.is_terminal()
            or state.current_player() != HUMAN_ID
            or self._is_buffered_selling(session)
        ):
            session.decision_plans.clear()
            return None
        if state.stage not in {"demand_pile", "action_direction"}:
            session.decision_plans.clear()
            return None

        _information, first_actions = platform.observe_game_state(
            session.rule_set, state, HUMAN_ID
        )
        generated: dict[str, DecisionPlanRecord] = {}
        if state.stage == "demand_pile":
            if state._demand_token is None:
                raise RuntimeError("Demand decision has no active marker")
            marker_index = int(state._demand_token[1])
            public_plans: list[DemandDecisionPlanV2] = []
            for pile_action in first_actions:
                first = _apply_player_action(
                    session.rule_set, state, HUMAN_ID, pile_action.action_id
                )
                _namespace, pile_id = session.rule_set.action_codec.decode(
                    pile_action.action_id
                )
                _information, bid_actions = platform.observe_game_state(
                    session.rule_set, first, HUMAN_ID
                )
                for bid_action in bid_actions:
                    if bid_action.amount is None:
                        raise RuntimeError("Demand bid action has no amount")
                    action_ids = (
                        int(pile_action.action_id),
                        int(bid_action.action_id),
                    )
                    presentation = {
                        "stockpile_id": int(pile_id),
                        "amount_thousands": int(bid_action.amount),
                        "marker_index": marker_index,
                    }
                    plan_id = self._decision_plan_id(
                        session, "demand", action_ids, presentation
                    )
                    generated[plan_id] = DecisionPlanRecord(
                        plan_id=plan_id,
                        kind="demand",
                        action_ids=action_ids,
                    )
                    public_plans.append(
                        DemandDecisionPlanV2(plan_id=plan_id, **presentation)
                    )
            session.decision_plans = generated
            return DemandDecisionBatchV2(plans=public_plans)

        public_impact_plans: list[MarketImpactDecisionPlanV2] = []
        for direction_action in first_actions:
            first = _apply_player_action(
                session.rule_set, state, HUMAN_ID, direction_action.action_id
            )
            _namespace, direction_ordinal = session.rule_set.action_codec.decode(
                direction_action.action_id
            )
            direction = "up" if int(direction_ordinal) == 0 else "down"
            _information, company_actions = platform.observe_game_state(
                session.rule_set, first, HUMAN_ID
            )
            for company_action in company_actions:
                _namespace, company_id = session.rule_set.action_codec.decode(
                    company_action.action_id
                )
                action_ids = (
                    int(direction_action.action_id),
                    int(company_action.action_id),
                )
                presentation = {
                    "direction": direction,
                    "company_id": int(company_id),
                    "movement": 2 if direction == "up" else -2,
                }
                plan_id = self._decision_plan_id(
                    session, "market_impact", action_ids, presentation
                )
                generated[plan_id] = DecisionPlanRecord(
                    plan_id=plan_id,
                    kind="market_impact",
                    action_ids=action_ids,
                )
                public_impact_plans.append(
                    MarketImpactDecisionPlanV2(plan_id=plan_id, **presentation)
                )
        session.decision_plans = generated
        return MarketImpactDecisionBatchV2(plans=public_impact_plans)

    @staticmethod
    def _decision_plan_id(
        session: V2GameSession,
        kind: str,
        action_ids: tuple[int, int],
        presentation: Mapping[str, Any],
    ) -> str:
        payload = {
            "revision": session.revision,
            "kind": kind,
            "actions": action_ids,
            "presentation": dict(presentation),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hmac.new(
            session.plan_secret, encoded.encode("utf-8"), hashlib.sha256
        ).hexdigest()[:32]

    def _deduplicate_sell_actions(
        self,
        session: V2GameSession,
        state: platform.GameState,
        legal: Sequence[platform.LegalAction],
    ) -> Sequence[platform.LegalAction]:
        if state.phase != platform.Phase.SELLING.value or state.stage != "selling":
            return legal
        selected: dict[
            tuple[int, int, int, int], tuple[platform.LegalAction, bool]
        ] = {}
        for action in legal:
            preview = platform.preview_sale_action(
                session.rule_set,
                state,
                int(state.current_player()),
                int(action.action_id),
            )
            key = (
                int(preview.company_id),
                int(preview.quantity_sold),
                int(preview.gross_value),
                int(preview.resulting_represented),
            )
            advances = self._sale_advances_cursor(session, state, action.action_id)
            current = selected.get(key)
            if current is None or (advances and not current[1]):
                selected[key] = (action, advances)
        return tuple(item[0] for item in selected.values())

    @staticmethod
    def _sale_advances_cursor(
        session: V2GameSession,
        state: platform.GameState,
        action_id: int,
    ) -> bool:
        before = (
            state.phase,
            int(state.current_player()),
            int(state._selling_company),
        )
        after = _apply_player_action(
            session.rule_set, state, int(state.current_player()), int(action_id)
        )
        if after.phase != platform.Phase.SELLING.value:
            return True
        return (
            after.phase,
            int(after.current_player()),
            int(after._selling_company),
        ) != before

    @staticmethod
    def _card_ref(session: V2GameSession, card_id: int) -> str:
        return hmac.new(
            session.plan_secret,
            f"card:{card_id}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:24]

    @staticmethod
    def _plan_id(
        session: V2GameSession,
        action_ids: tuple[int, int, int],
        placements: tuple[SupplyPlacementV2, SupplyPlacementV2],
    ) -> str:
        payload = {
            "revision": session.revision,
            "actions": action_ids,
            "placements": [item.model_dump(mode="json") for item in placements],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hmac.new(
            session.plan_secret, encoded.encode("utf-8"), hashlib.sha256
        ).hexdigest()[:32]

    def _append_supply_placements(
        self, session: V2GameSession, plan: SupplyPlanRecord
    ) -> None:
        cards_by_ref = {
            self._card_ref(session, card.card_id): card
            for card in session.state.players[HUMAN_ID].known_cards.values()
        }
        # The authoritative commit relocates known cards, so player memory is
        # the safe source for resolving the opaque references after the swap.
        for placement in plan.placements:
            card = cards_by_ref.get(placement.card_ref)
            if card is None:
                raise RuntimeError("committed Supply card is absent from player memory")
            session.ordered_piles[placement.stockpile_id].append(
                OrderedPileCard(
                    card_id=card.card_id,
                    face_up=placement.visibility == "face_up",
                )
            )

    def _build_view(self, session: V2GameSession) -> GameViewV2:
        public_information, _unused = platform.observe_game_state(
            session.rule_set, session.state, None
        )
        checkpoint = session.checkpoint
        private_state = session.state
        legal: Sequence[platform.LegalAction] = ()
        # A presentation checkpoint is a hard information boundary.  In
        # particular, a Demand result may sit on the engine's first sealed-
        # Selling state; merely polling it must not create or expose a selling
        # validator plan before acknowledgement.
        sealed = checkpoint is None and self._is_buffered_selling(session)
        human_plan: SealedPlayerPlanV2 | None = None
        if sealed:
            buffer = self._ensure_sealed_buffer(session)
            human_plan = buffer.plans[HUMAN_ID]
            private_state = session.state if human_plan.complete else human_plan.state
        private_information, observed_legal = platform.observe_game_state(
            session.rule_set, private_state, HUMAN_ID
        )
        private_view_state = private_state
        private_view_information = private_information
        if human_plan is not None:
            # Keep canonical Portfolio holdings and POSITION on the same
            # authoritative, pre-settlement state.  The shadow clone exists
            # only to validate the human's sealed decisions and legal actions.
            private_view_state = session.state
            private_view_information, _settled_legal = platform.observe_game_state(
                session.rule_set, session.state, HUMAN_ID
            )
        if (
            session.checkpoint is None
            and not human_plan
            and session.state.current_player() == HUMAN_ID
        ):
            legal = observed_legal
        elif (
            session.checkpoint is None
            and human_plan is not None
            and not human_plan.complete
            and private_state.current_player() == HUMAN_ID
        ):
            legal = observed_legal

        public = public_information.public_state
        if checkpoint is not None:
            phase = checkpoint.kind.upper()
            phase_step = "acknowledge"
            round_number = checkpoint.round
            active_player_id: int | None = None
            legal = ()
        elif session.state.is_terminal():
            phase = platform.Phase.TERMINAL.value
            phase_step = "terminal"
            round_number = int(public["round"])
            active_player_id = None
        elif sealed:
            phase = str(public["phase"])
            phase_step = "private_selling"
            round_number = int(public["round"])
            active_player_id = HUMAN_ID if human_plan and not human_plan.complete else None
        else:
            presentation = platform.get_presentation_state(
                session.rule_set, session.state, HUMAN_ID
            )
            phase = str(public["phase"])
            round_number = int(public["round"])
            active_player_id = presentation.current_actor
            phase_step = (
                presentation.stage if active_player_id == HUMAN_ID else "waiting"
            )

        live_markers = self._bid_markers(session)
        if session.resolved_auction is not None:
            markers = [
                marker.model_copy(deep=True)
                for marker in session.resolved_auction.markers
            ]
            stockpiles = [
                stockpile.model_copy(deep=True)
                for stockpile in session.resolved_auction.stockpiles
            ]
        else:
            markers = live_markers
            stockpiles = self._stockpiles(
                session, private_information, markers
            )
        legal = self._deduplicate_sell_actions(session, private_state, legal)
        legal_actions = [
            self._legal_action(session, private_state, action) for action in legal
        ]
        supply_batch = self._supply_batch(session)
        decision_batch = self._decision_batch(session)
        if supply_batch is not None:
            legal_actions = []
            phase_step = "supply"
        if decision_batch is not None:
            legal_actions = []
        pending = self._pending_decision(
            session,
            private_state,
            legal_actions,
            supply_batch,
            decision_batch,
            human_plan,
        )
        options = LiteOptionsV2(
            market_impact=session.configuration.impact,
            trading_fees=session.configuration.fees,
            dividends=session.configuration.dividend,
            sell_order=session.configuration.sell_order,
        )
        terminal = (
            self._terminal_results(session)
            if session.state.is_terminal() and checkpoint is None
            else None
        )
        return GameViewV2(
            game_id=session.game_id,
            revision=session.revision,
            configuration=ConfigurationV2(options=options),
            round=round_number,
            phase=phase,
            phase_step=phase_step,
            viewer=ViewerV2(),
            active_player_id=active_player_id,
            companies=self._companies(session, public),
            stockpiles=stockpiles,
            players=self._players(
                session,
                markers,
                active_player_id,
                human_plan,
            ),
            private=self._private_view(
                session,
                private_view_state,
                private_view_information,
                suppress_action_cards=checkpoint is not None,
            ),
            pending_decision=pending,
            legal_actions=legal_actions,
            supply_batch=supply_batch,
            decision_batch=decision_batch,
            checkpoint=checkpoint,
            recent_events=self._presentation_events(session.state, round_number),
            terminal_results=terminal,
        )

    @staticmethod
    def _companies(
        session: V2GameSession, public: Mapping[str, Any]
    ) -> list[CompanyV2]:
        return [
            CompanyV2(
                company_id=index,
                symbol=name[:1].upper(),
                name=name,
                display_name=COMPANY_PRESENTATION[index][0],
                pattern=COMPANY_PRESENTATION[index][1],
                price_dollars_per_share=int(public["prices"][name]),
            )
            for index, name in enumerate(session.rule_set.company_names)
        ]

    def _bid_markers(self, session: V2GameSession) -> list[BidMarkerV2]:
        result: list[BidMarkerV2] = []
        presentation = platform.get_presentation_state(
            session.rule_set, session.state, None
        )
        active_token = presentation.demand_token
        occupied = {
            (marker.player_id, marker.marker_index): marker
            for marker in presentation.stockpile_markers
        }
        for player_id in (HUMAN_ID, COMPUTER_ID):
            for marker_index in range(session.rule_set.meeples_per_player):
                marker_key = (player_id, marker_index)
                leading = occupied.get(marker_key)
                if leading is not None:
                    status = "locked" if leading.status == "locked" else "placed"
                    stockpile_id = leading.stockpile_id
                    bid = leading.bid_value
                elif marker_key in session.outbid_markers:
                    status = "rebidding" if active_token == marker_key else "outbid"
                    stockpile_id = None
                    bid = None
                else:
                    status = "available"
                    stockpile_id = None
                    bid = None
                result.append(
                    BidMarkerV2(
                        player_id=player_id,
                        marker_index=marker_index,
                        status=status,
                        stockpile_id=stockpile_id,
                        bid_thousands=bid,
                    )
                )
        return result

    def _stockpiles(
        self,
        session: V2GameSession,
        information: platform.InformationState,
        markers: Sequence[BidMarkerV2],
    ) -> list[StockpileV2]:
        cards_by_pile = self._viewer_pile_cards(
            session, session.state, information
        )
        result: list[StockpileV2] = []
        for pile in session.state.stockpiles:
            marker = next(
                (
                    item
                    for item in markers
                    if item.stockpile_id == pile.stockpile_id
                    and item.status in {"placed", "locked"}
                ),
                None,
            )
            bid = (
                StockpileBidV2(
                    player_id=marker.player_id,
                    marker_index=marker.marker_index,
                    amount_thousands=int(marker.bid_thousands or 0),
                )
                if marker is not None
                else None
            )
            result.append(
                StockpileV2(
                    stockpile_id=pile.stockpile_id,
                    cards_bottom_to_top=list(cards_by_pile[pile.stockpile_id]),
                    bid=bid,
                    locked=bool(pile.locked),
                    purchaser_id=pile.purchaser,
                    resolved=False,
                )
            )
        return result

    def _viewer_pile_cards(
        self,
        session: V2GameSession,
        state: platform.GameState,
        information: platform.InformationState,
    ) -> dict[int, tuple[PileCardV2, ...]]:
        known = {card.card_id: card for card in information.known_cards}
        result: dict[int, tuple[PileCardV2, ...]] = {}
        for pile in state.stockpiles:
            actual = {
                card.card_id: card
                for card in pile.face_up_cards + pile.face_down_cards
            }
            ledger = session.ordered_piles[pile.stockpile_id]
            if set(actual) != {entry.card_id for entry in ledger}:
                raise RuntimeError("ordered Stockpile ledger diverged from engine state")
            cards = []
            for entry in ledger:
                card = actual[entry.card_id]
                if entry.face_up:
                    cards.append(self._visible_card(session.rule_set, card))
                elif entry.card_id in known:
                    cards.append(
                        RememberedCardV2(
                            card=self._visible_card(
                                session.rule_set, known[entry.card_id]
                            )
                        )
                    )
                else:
                    cards.append(HiddenCardV2())
            result[pile.stockpile_id] = tuple(cards)
        return result

    @staticmethod
    def _capture_pile_cards(
        session: V2GameSession,
        state: platform.GameState,
    ) -> dict[int, tuple[CapturedPileCard, ...]]:
        captured: dict[int, tuple[CapturedPileCard, ...]] = {}
        for pile in state.stockpiles:
            actual = {
                card.card_id: card
                for card in pile.face_up_cards + pile.face_down_cards
            }
            ledger = session.ordered_piles[pile.stockpile_id]
            if set(actual) != {entry.card_id for entry in ledger}:
                raise RuntimeError("ordered Stockpile ledger diverged from engine state")
            captured[pile.stockpile_id] = tuple(
                CapturedPileCard(card=actual[entry.card_id], face_up=entry.face_up)
                for entry in ledger
            )
        return captured

    @staticmethod
    def _captured_pile_card(
        rule_set: platform.RuleSet,
        captured: CapturedPileCard,
        known: Mapping[int, platform.Card],
    ) -> PileCardV2:
        if captured.face_up:
            return V2SessionStore._visible_card(rule_set, captured.card)
        remembered = known.get(captured.card.card_id)
        if remembered is not None:
            return RememberedCardV2(
                card=V2SessionStore._visible_card(rule_set, remembered)
            )
        return HiddenCardV2()

    def _players(
        self,
        session: V2GameSession,
        markers: Sequence[BidMarkerV2],
        active_player_id: int | None,
        human_plan: SealedPlayerPlanV2 | None,
    ) -> list[HumanPublicPlayerV2 | ComputerPublicPlayerV2]:
        checkpoint_kind = session.checkpoint.kind if session.checkpoint else None
        cash_deltas = session.cash_deltas if checkpoint_kind == "demand_result" else None
        human_status = self._player_status(
            HUMAN_ID, active_player_id, session.state, human_plan
        )
        computer_status = self._player_status(
            COMPUTER_ID, active_player_id, session.state, None
        )
        human_cash = int(session.state.players[HUMAN_ID].cash)
        computer_cash = int(session.state.players[COMPUTER_ID].cash)
        return [
            HumanPublicPlayerV2(
                player_id=HUMAN_ID,
                name=PLAYER_NAMES[HUMAN_ID],
                cash_thousands=human_cash,
                cash_delta_thousands=(
                    cash_deltas[HUMAN_ID]
                    if cash_deltas and cash_deltas[HUMAN_ID] != 0
                    else None
                ),
                position_value_thousands=self._position_value(
                    session.state, HUMAN_ID
                ),
                position_delta_thousands=(
                    session.position_delta
                    if checkpoint_kind == "round_result"
                    and session.position_delta != 0
                    else None
                ),
                active=active_player_id == HUMAN_ID,
                status=human_status,
                bid_markers=[item for item in markers if item.player_id == HUMAN_ID],
            ),
            ComputerPublicPlayerV2(
                player_id=COMPUTER_ID,
                name=PLAYER_NAMES[COMPUTER_ID],
                cash_thousands=computer_cash,
                cash_delta_thousands=(
                    cash_deltas[COMPUTER_ID]
                    if cash_deltas and cash_deltas[COMPUTER_ID] != 0
                    else None
                ),
                active=active_player_id == COMPUTER_ID,
                status=computer_status,
                bid_markers=[item for item in markers if item.player_id == COMPUTER_ID],
            ),
        ]

    @staticmethod
    def _player_status(
        player_id: int,
        active_player_id: int | None,
        state: platform.GameState,
        human_plan: SealedPlayerPlanV2 | None,
    ) -> str:
        if human_plan is not None and player_id == HUMAN_ID:
            return "WAIT" if human_plan.complete else "SELL"
        if active_player_id != player_id:
            return "WAIT"
        return {
            platform.Phase.SUPPLY.value: "SUPPLY",
            platform.Phase.DEMAND.value: "BID",
            platform.Phase.ACTION.value: "IMPACT",
            platform.Phase.SELLING.value: "SELL",
            platform.Phase.MOVEMENT.value: "MOVE",
        }.get(state.phase, "TURN")

    def _private_view(
        self,
        session: V2GameSession,
        state: platform.GameState,
        information: platform.InformationState,
        *,
        suppress_action_cards: bool = False,
    ) -> ViewerPrivateV2:
        holdings = []
        for company_id, name in enumerate(session.rule_set.company_names):
            regular = int(information.owned_stocks["regular"][name])
            split = int(information.owned_stocks["split"][name])
            represented = regular + 2 * split
            price = state._company_price(company_id)
            holdings.append(
                HoldingV2(
                    company_id=company_id,
                    company=name,
                    shares_thousands=represented,
                    price_dollars_per_share=price,
                    market_value_thousands=represented * price,
                )
            )
        return ViewerPrivateV2(
            market_information=self._market_information(session, state, information),
            holdings=holdings,
            available_action_cards=[
                self._action_card(effect) for effect in information.acquired_actions
            ] if not suppress_action_cards else [],
        )

    def _market_information(
        self,
        session: V2GameSession,
        state: platform.GameState,
        information: platform.InformationState,
    ) -> list[MarketInformationSlotV2]:
        slots: list[MarketInformationSlotV2] = []
        seen: set[tuple[int, int | str]] = set()
        for visibility, pairs in (
            ("public", state.revealed_information),
            ("private", information.private_information),
            ("public", state.public_information),
            ("private", information.viewed_information_pairs),
        ):
            for raw_pair in pairs:
                pair = tuple(raw_pair)
                if pair in seen:
                    continue
                seen.add(pair)
                slots.append(
                    MarketInformationSlotV2(
                        visibility=visibility,
                        card=self._information_card(session.rule_set, pair),
                    )
                )
        while len(slots) < session.rule_set.company_count:
            slots.append(
                MarketInformationSlotV2(
                    visibility="hidden", card=HiddenCardV2()
                )
            )
        return slots[: session.rule_set.company_count]

    def _legal_action(
        self,
        session: V2GameSession,
        state: platform.GameState,
        action: platform.LegalAction,
    ) -> LegalActionV2:
        namespace = action.action_type
        ordinal = int(action.payload.get("ordinal", 0))
        stage = state.stage
        control = "generic"
        target_id: str | None = None
        amount = action.amount
        direction: str | None = None
        preview: SalePreviewV2 | None = None
        if stage == "demand_pile":
            control, target_id = "stockpile", f"stockpile:{ordinal}"
        elif stage == "demand_bid":
            control = "bid"
        elif stage in {"action_direction", "action_cramer_direction"}:
            control = "action_card"
            direction = "up" if ordinal == 0 else "down"
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
        return LegalActionV2(
            action_id=action.action_id,
            control=control,
            label=action.display_label,
            target_id=target_id,
            amount_thousands=amount,
            direction=direction,
            sale_preview=preview,
        )

    @staticmethod
    def _sale_preview(
        rule_set: platform.RuleSet,
        state: platform.GameState,
        action: platform.LegalAction,
    ) -> SalePreviewV2:
        preview = platform.preview_sale_action(
            rule_set, state, int(state.current_player()), action.action_id
        )
        return SalePreviewV2(
            company_id=preview.company_id,
            company=preview.company_name,
            shares_thousands=preview.quantity_sold,
            price_dollars_per_share=preview.unit_price,
            gross_value_thousands=preview.gross_value,
            resulting_shares_thousands=preview.resulting_represented,
        )

    @staticmethod
    def _pending_decision(
        session: V2GameSession,
        state: platform.GameState,
        legal_actions: Sequence[LegalActionV2],
        supply_batch: SupplyBatchV2 | None,
        decision_batch: DecisionBatchV2 | None,
        human_plan: SealedPlayerPlanV2 | None,
    ) -> PendingDecisionV2:
        if session.checkpoint is not None:
            return PendingDecisionV2(kind="acknowledge", prompt="CONTINUE")
        if session.state.is_terminal():
            return PendingDecisionV2(kind="terminal", prompt="GAME END")
        if human_plan is not None:
            if human_plan.complete:
                return PendingDecisionV2(
                    kind="private_selling", prompt="COMPUTER"
                )
            return PendingDecisionV2(
                kind="sell",
                prompt="SELL",
                company_id=state._selling_company,
            )
        if supply_batch is not None:
            return PendingDecisionV2(kind="supply", prompt="PLACE")
        if decision_batch is not None:
            if decision_batch.kind == "demand":
                return PendingDecisionV2(kind="bid_pile", prompt="BID")
            return PendingDecisionV2(kind="action_card", prompt="IMPACT")
        if session.state.current_player() != HUMAN_ID or not legal_actions:
            return PendingDecisionV2(kind="waiting", prompt="COMPUTER")
        presentation = platform.get_presentation_state(
            session.rule_set, state, HUMAN_ID
        )
        kind, prompt = {
            "demand_pile": ("bid_pile", "BID"),
            "demand_bid": ("bid_amount", "BID"),
            "action_direction": ("action_card", "IMPACT"),
            "action_company": ("action_company", "IMPACT"),
            "selling": ("sell", "SELL"),
            "dividend_claim": ("dividend_claim", "DIVIDEND"),
        }.get(state.stage, ("generic", "CHOOSE"))
        return PendingDecisionV2(
            kind=kind,
            prompt=prompt,
            selected_stockpile_id=presentation.demand_pile,
            selected_action_effect=presentation.selected_direction,
            company_id=(
                presentation.selling_company if state.stage == "selling" else None
            ),
        )

    @staticmethod
    def _presentation_events(
        state: platform.GameState, display_round: int
    ) -> list[PublicEventV2]:
        events: list[PublicEventV2] = []
        raw_events = [
            raw
            for raw in platform.get_presentation_events(state.rule_set, state)
            if int(getattr(raw, "round", display_round)) == display_round
        ][-MAX_PUBLIC_EVENTS:]
        for index, raw in enumerate(raw_events):
            data = asdict(raw) if is_dataclass(raw) else dict(raw)
            actual_delta = data.get("actual_delta")
            requested_delta = data.get("requested_delta")
            delta = actual_delta if actual_delta is not None else requested_delta
            direction = None
            if delta is not None and int(delta) != 0:
                direction = "up" if int(delta) > 0 else "down"
            forecast = _normalise_forecast(data.get("forecast"))
            events.append(
                PublicEventV2(
                    event_id=int(
                        data.get(
                            "presentation_sequence", data.get("sequence", index + 1)
                        )
                    ),
                    event_type=str(data.get("event_type", data.get("cause", "market"))),
                    cause=(
                        None if data.get("cause") is None else str(data.get("cause"))
                    ),
                    round=int(data.get("round", state.round)),
                    company_id=(
                        None
                        if data.get("company_id") is None
                        else int(data.get("company_id"))
                    ),
                    company=(
                        None
                        if data.get("company", data.get("company_name")) is None
                        else str(data.get("company", data.get("company_name")))
                    ),
                    prior_price_dollars_per_share=(
                        None
                        if data.get("prior_price") is None
                        else int(data.get("prior_price"))
                    ),
                    price_delta=None if delta is None else int(delta),
                    resulting_price_dollars_per_share=(
                        None
                        if data.get("resulting_price") is None
                        else int(data.get("resulting_price"))
                    ),
                    forecast=forecast,
                    cash_effect_thousands=2 if forecast == "DIVIDEND" else None,
                    direction=direction,
                )
            )
        return events

    def _terminal_results(self, session: V2GameSession) -> TerminalResultsV2:
        result = platform.score_game(session.rule_set, session.state)
        details = platform.terminal_liquidation_details(
            session.rule_set, session.state
        )
        detail_by_player = {detail.player_id: detail for detail in details}
        distinct = sorted(set(result.final_cash_by_player.values()), reverse=True)
        ranks = {value: distinct.index(value) + 1 for value in distinct}
        players: list[TerminalPlayerV2] = []
        for player_id in (HUMAN_ID, COMPUTER_ID):
            detail = detail_by_player.get(player_id)
            if detail is not None:
                lines = [
                    LiquidationLineV2(
                        company_id=line.company_id,
                        company=line.company_name,
                        shares_thousands=line.represented_shares,
                        price_dollars_per_share=line.unit_price,
                        value_thousands=line.value,
                    )
                    for line in detail.companies
                ]
                liquidation_value = int(detail.liquidation_value)
                final_cash = int(detail.final_cash)
                rank = int(detail.rank)
                winner = bool(detail.winner)
            else:
                lines = [
                    LiquidationLineV2(
                        company_id=company_id,
                        company=name,
                        shares_thousands=session.state._represented_shares(
                            player_id, company_id
                        ),
                        price_dollars_per_share=session.state._company_price(company_id),
                        value_thousands=(
                            session.state._represented_shares(player_id, company_id)
                            * session.state._company_price(company_id)
                        ),
                    )
                    for company_id, name in enumerate(session.rule_set.company_names)
                ]
                liquidation_value = int(result.liquidation_values[player_id])
                final_cash = int(result.final_cash_by_player[player_id])
                rank = ranks[final_cash]
                winner = player_id in result.winner_ids
            players.append(
                TerminalPlayerV2(
                    player_id=player_id,
                    player_name=PLAYER_NAMES[player_id],
                    # Final fee debts settle against gross terminal cash.  This
                    # is therefore the net cash immediately before stock
                    # liquidation, so the three displayed amounts reconcile.
                    cash_before_liquidation_thousands=(
                        final_cash - liquidation_value
                    ),
                    liquidation_value_thousands=liquidation_value,
                    final_cash_thousands=final_cash,
                    rank=rank,
                    winner=winner,
                    liquidation=lines,
                )
            )
        return TerminalResultsV2(
            players=players,
            winner_ids=[int(value) for value in result.winner_ids],
        )

    @staticmethod
    def _visible_card(
        rule_set: platform.RuleSet,
        card: platform.Card | Mapping[str, Any],
    ) -> VisibleCardV2:
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
            if company_id is None:
                raise RuntimeError("stock card has no company")
            return StockCardV2(
                company_id=int(company_id),
                company=rule_set.company_names[int(company_id)],
                shares_thousands=int(value or 1),
            )
        if kind == platform.CardType.TRADING_FEE.value:
            return TradingFeeCardV2(cash_effect_thousands=-abs(int(value or 0)))
        return V2SessionStore._action_card(effect or value)

    @staticmethod
    def _action_card(effect: object) -> ActionCardV2:
        effect_text = str(effect)
        normalized = effect_text.strip().casefold()
        if normalized in {"boom", "stock boom"}:
            direction = "up"
        elif normalized in {"bust", "stock bust"}:
            direction = "down"
        else:
            raise ValueError(f"unsupported Market Impact effect {effect_text!r}")
        return ActionCardV2(
            effect=effect_text,
            direction=direction,
            movement=2 if direction == "up" else -2,
        )

    @staticmethod
    def _information_card(
        rule_set: platform.RuleSet,
        pair: tuple[int, int | str],
    ) -> InformationCardV2:
        company_id, forecast = pair
        normalized = (
            "DIVIDEND" if str(forecast).upper() == "DIVIDEND" else int(forecast)
        )
        return InformationCardV2(
            company_id=int(company_id),
            company=rule_set.company_names[int(company_id)],
            forecast=normalized,
            cash_effect_thousands=2 if normalized == "DIVIDEND" else None,
        )


def _normalise_forecast(value: Any) -> int | str | None:
    if value is None:
        return None
    return "DIVIDEND" if str(value).upper() == "DIVIDEND" else int(value)


__all__ = [
    "COMPUTER_ID",
    "HUMAN_ID",
    "SupplyPlanRecord",
    "V2GameSession",
    "V2SessionStore",
]
