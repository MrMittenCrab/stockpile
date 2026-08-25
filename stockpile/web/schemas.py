"""Versioned, allowlisted HTTP contracts for local Stockpile Lite play."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


SCHEMA_VERSION = "1.0"


class StrictModel(BaseModel):
    """Base model whose wire shape cannot silently grow through request extras."""

    model_config = ConfigDict(extra="forbid", strict=True)


class ErrorBody(StrictModel):
    code: str
    message: str


class ErrorResponse(StrictModel):
    error: ErrorBody


class IntegerLimits(StrictModel):
    minimum: int
    maximum: int


class SetupDefaults(StrictModel):
    player_count: int
    round_count: int


class OptionDescriptor(StrictModel):
    key: Literal[
        "market_impact",
        "starting_share",
        "trading_fees",
        "dividends",
        "sell_order",
    ]
    label: str
    description: str
    default: bool


class SetupResponse(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    mode: Literal["lite"] = "lite"
    defaults: SetupDefaults
    player_limits: IntegerLimits
    round_limits: IntegerLimits
    options: list[OptionDescriptor]


class LiteOptions(StrictModel):
    market_impact: bool = False
    starting_share: bool = False
    trading_fees: bool = False
    dividends: bool = False
    sell_order: bool = False


PlayerName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=32),
]


class CreateGameRequest(StrictModel):
    player_count: int = Field(ge=2, le=5)
    player_names: list[PlayerName] = Field(min_length=2, max_length=5)
    round_count: int = Field(default=6, ge=1, le=10)
    options: LiteOptions = Field(default_factory=LiteOptions)
    seed: int | None = Field(default=None, ge=0, le=(2**63 - 1))

    @model_validator(mode="after")
    def validate_players(self) -> "CreateGameRequest":
        if len(self.player_names) != self.player_count:
            raise ValueError("player_names must contain exactly player_count names")
        normalized = [name.casefold() for name in self.player_names]
        if len(set(normalized)) != len(normalized):
            raise ValueError("player names must be unique")
        return self


class SeatLink(StrictModel):
    player_id: int
    player_name: str
    url: str


class CreateGameResponse(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    game_id: str
    seats: list[SeatLink]


class ActionRequestV1(StrictModel):
    action_id: int = Field(ge=0)
    expected_revision: int = Field(ge=0)


ChatText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]


class ChatRequestV1(StrictModel):
    message: ChatText


class ChatMessageV1(StrictModel):
    message_id: int
    player_id: int
    player_name: str
    message: str
    created_at: str


class ChatResponseV1(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    chat_message: ChatMessageV1


class HiddenCardV1(StrictModel):
    visibility: Literal["hidden"] = "hidden"


class StockCardV1(StrictModel):
    visibility: Literal["visible"] = "visible"
    kind: Literal["stock"] = "stock"
    company_id: int
    company: str
    quantity: int


class TradingFeeCardV1(StrictModel):
    visibility: Literal["visible"] = "visible"
    kind: Literal["trading_fee"] = "trading_fee"
    amount: int


class ActionCardV1(StrictModel):
    visibility: Literal["visible"] = "visible"
    kind: Literal["action"] = "action"
    effect: str
    direction: Literal["up", "down"]
    movement: Literal[2] = 2


class InformationCardV1(StrictModel):
    visibility: Literal["visible"] = "visible"
    kind: Literal["company_forecast"] = "company_forecast"
    company_id: int
    company: str
    forecast: int | Literal["DIVIDEND"]


VisibleCardV1 = Annotated[
    StockCardV1 | TradingFeeCardV1 | ActionCardV1 | InformationCardV1,
    Field(discriminator="kind"),
]
CardV1 = HiddenCardV1 | VisibleCardV1


class CompanyV1(StrictModel):
    company_id: int
    symbol: str
    name: str
    display_name: str
    pattern: Literal[
        "matrix",
        "ledger",
        "molecular",
        "chevron",
        "crosshatch",
        "wave",
    ]
    price: int
    color: str


class BidMarkerV1(StrictModel):
    player_id: int
    marker_index: int
    status: Literal["available", "placed", "outbid", "rebidding", "locked"]
    stockpile_id: int | None = None
    bid: int | None = None


class StockpileV1(StrictModel):
    stockpile_id: int
    visible_cards: list[VisibleCardV1]
    hidden_cards: list[HiddenCardV1]
    marker: BidMarkerV1 | None = None
    bid: int | None = None
    locked: bool
    purchaser_id: int | None = None


class PublicPlayerV1(StrictModel):
    player_id: int
    name: str
    cash: int
    active: bool
    status: str
    fee_debts: list[int]
    bid_markers: list[BidMarkerV1]


class MarketInformationSlotV1(StrictModel):
    visibility: Literal["private", "public", "hidden"]
    source: Literal["dealt", "viewed", "revealed", "unknown"]
    card: InformationCardV1 | HiddenCardV1


class HoldingV1(StrictModel):
    company_id: int
    company: str
    regular: int
    split: int
    represented: int
    price: int


class KnownPileCardV1(StrictModel):
    stockpile_id: int
    card: VisibleCardV1


class ViewerPrivateV1(StrictModel):
    hand: list[VisibleCardV1]
    market_information: list[MarketInformationSlotV1]
    holdings: list[HoldingV1]
    known_pile_cards: list[KnownPileCardV1]
    available_action_cards: list[ActionCardV1]


class SalePreviewV1(StrictModel):
    company_id: int
    company: str
    quantity: int
    unit_price: int
    gross_value: int
    resulting_regular: int
    resulting_split: int
    resulting_represented: int


class LegalActionV1(StrictModel):
    action_id: int
    control: Literal[
        "card",
        "stockpile",
        "bid",
        "action_card",
        "company",
        "sell",
        "dividend",
        "continue",
        "generic",
    ]
    label: str
    target_id: str | None = None
    amount: int | None = None
    placement_visibility: Literal["face_up", "face_down"] | None = None
    sale_preview: SalePreviewV1 | None = None


class PendingDecisionV1(StrictModel):
    kind: Literal[
        "supply_card",
        "supply_face_up_pile",
        "supply_face_down_pile",
        "bid_pile",
        "bid_amount",
        "action_card",
        "action_company",
        "sell",
        "dividend_claim",
        "acknowledge",
        "waiting",
        "private_selling",
        "terminal",
        "generic",
    ]
    prompt: str
    selected_card_index: int | None = None
    selected_stockpile_id: int | None = None
    selected_action_effect: str | None = None
    company_id: int | None = None
    private_progress: int | None = None
    private_total: int | None = None


class PublicHistoryEntryV1(StrictModel):
    sequence: int
    phase: str
    actor_id: int | None
    summary: str
    sale_totals: dict[str, dict[str, int]] | None = None


class PublicEventV1(StrictModel):
    event_id: int
    event_type: str
    cause: str | None = None
    round: int
    description: str
    company_id: int | None = None
    company: str | None = None
    prior_price: int | None = None
    requested_delta: int | None = None
    actual_delta: int | None = None
    resulting_price: int | None = None
    forecast: int | Literal["DIVIDEND"] | None = None
    effect: str | None = None
    actor_id: int | None = None


class LiquidationLineV1(StrictModel):
    company_id: int
    company: str
    represented_shares: int
    unit_price: int
    value: int


class TerminalPlayerV1(StrictModel):
    player_id: int
    player_name: str
    cash_before_liquidation: int
    liquidation_value: int
    final_cash: int
    rank: int
    winner: bool
    liquidation: list[LiquidationLineV1]


class TerminalResultsV1(StrictModel):
    players: list[TerminalPlayerV1]
    winner_ids: list[int]


class ConfigurationV1(StrictModel):
    mode: Literal["lite"] = "lite"
    player_count: int
    round_count: int
    options: LiteOptions


class CapabilitiesV1(StrictModel):
    market_impact: bool
    starting_share: bool
    trading_fees: bool
    dividends: bool
    sequential_selling: bool
    stock_splits: Literal[False] = False
    majority_bonus: Literal[False] = False
    price_ceiling: None = None


class ViewerV1(StrictModel):
    player_id: int
    name: str


class GameViewV1(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    game_id: str
    revision: int
    configuration: ConfigurationV1
    capabilities: CapabilitiesV1
    round: int
    total_rounds: int
    phase: str
    phase_step: str
    viewer: ViewerV1
    active_player_id: int | None
    companies: list[CompanyV1]
    stockpiles: list[StockpileV1]
    players: list[PublicPlayerV1]
    private: ViewerPrivateV1
    pending_decision: PendingDecisionV1
    legal_actions: list[LegalActionV1]
    public_history: list[PublicHistoryEntryV1]
    recent_events: list[PublicEventV1]
    chat: list[ChatMessageV1]
    terminal_results: TerminalResultsV1 | None = None
