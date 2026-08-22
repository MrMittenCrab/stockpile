"""Strict browser-only contracts for the human-versus-computer Lite UI."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION_V2 = "2.0"


class StrictModelV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class LiteOptionsV2(StrictModelV2):
    market_impact: bool = False
    trading_fees: bool = False
    dividends: bool = False
    sell_order: bool = False


OptionKeyV2 = Literal[
    "market_impact", "trading_fees", "dividends", "sell_order"
]


class OptionDescriptorV2(StrictModelV2):
    key: OptionKeyV2
    label: str
    default: bool


class SetupResponseV2(StrictModelV2):
    schema_version: Literal["2.0"] = SCHEMA_VERSION_V2
    mode: Literal["lite"] = "lite"
    round_count: Literal[6] = 6
    options: list[OptionDescriptorV2]


class CreateGameRequestV2(StrictModelV2):
    options: LiteOptionsV2 = Field(default_factory=LiteOptionsV2)
    seed: int | None = Field(default=None, ge=0, le=(2**63 - 1))


class CreateGameResponseV2(StrictModelV2):
    schema_version: Literal["2.0"] = SCHEMA_VERSION_V2
    game_id: str
    game_url: str


class ActionRequestV2(StrictModelV2):
    action_id: int = Field(ge=0)
    expected_revision: int = Field(ge=0)


class SupplyRequestV2(StrictModelV2):
    plan_id: str = Field(min_length=1, max_length=128)
    expected_revision: int = Field(ge=0)


class AcknowledgementRequestV2(StrictModelV2):
    checkpoint_id: str = Field(min_length=1, max_length=128)
    expected_revision: int = Field(ge=0)


class HiddenCardV2(StrictModelV2):
    visibility: Literal["hidden"] = "hidden"


class StockCardV2(StrictModelV2):
    visibility: Literal["visible"] = "visible"
    kind: Literal["stock"] = "stock"
    company_id: int
    company: str
    shares_thousands: int


class TradingFeeCardV2(StrictModelV2):
    visibility: Literal["visible"] = "visible"
    kind: Literal["trading_fee"] = "trading_fee"
    cash_effect_thousands: int


class ActionCardV2(StrictModelV2):
    visibility: Literal["visible"] = "visible"
    kind: Literal["action"] = "action"
    effect: str
    direction: Literal["up", "down"]
    movement: int


VisibleCardV2 = StockCardV2 | TradingFeeCardV2 | ActionCardV2


class RememberedCardV2(StrictModelV2):
    visibility: Literal["remembered"] = "remembered"
    face_down: Literal[True] = True
    card: VisibleCardV2


PileCardV2 = HiddenCardV2 | VisibleCardV2 | RememberedCardV2


class InformationCardV2(StrictModelV2):
    visibility: Literal["visible"] = "visible"
    kind: Literal["company_forecast"] = "company_forecast"
    company_id: int
    company: str
    forecast: int | Literal["DIVIDEND"]
    cash_effect_thousands: int | None = None


class MarketInformationSlotV2(StrictModelV2):
    visibility: Literal["private", "public", "hidden"]
    card: InformationCardV2 | HiddenCardV2


class CompanyV2(StrictModelV2):
    company_id: int
    symbol: str
    name: str
    display_name: str
    pattern: Literal[
        "matrix", "ledger", "molecular", "chevron", "crosshatch", "wave"
    ]
    price_dollars_per_share: int


class BidMarkerV2(StrictModelV2):
    player_id: int
    marker_index: int
    status: Literal["available", "placed", "outbid", "rebidding", "locked"]
    stockpile_id: int | None = None
    bid_thousands: int | None = None


class StockpileBidV2(StrictModelV2):
    player_id: int
    marker_index: int
    amount_thousands: int


class StockpileV2(StrictModelV2):
    stockpile_id: int
    cards_bottom_to_top: list[PileCardV2]
    bid: StockpileBidV2 | None = None
    locked: bool
    purchaser_id: int | None = None


class HumanPublicPlayerV2(StrictModelV2):
    role: Literal["human"] = "human"
    player_id: int
    name: str
    cash_thousands: int
    cash_delta_thousands: int | None = None
    position_value_thousands: int
    position_delta_thousands: int | None = None
    active: bool
    status: str
    bid_markers: list[BidMarkerV2]


class ComputerPublicPlayerV2(StrictModelV2):
    role: Literal["computer"] = "computer"
    player_id: int
    name: str
    cash_thousands: int
    cash_delta_thousands: int | None = None
    active: bool
    status: str
    bid_markers: list[BidMarkerV2]


PublicPlayerV2 = HumanPublicPlayerV2 | ComputerPublicPlayerV2


class HoldingV2(StrictModelV2):
    company_id: int
    company: str
    shares_thousands: int
    price_dollars_per_share: int
    market_value_thousands: int


class ViewerPrivateV2(StrictModelV2):
    market_information: list[MarketInformationSlotV2]
    holdings: list[HoldingV2]
    available_action_cards: list[ActionCardV2]


class SalePreviewV2(StrictModelV2):
    company_id: int
    company: str
    shares_thousands: int
    price_dollars_per_share: int
    gross_value_thousands: int
    resulting_shares_thousands: int


class LegalActionV2(StrictModelV2):
    action_id: int
    control: Literal[
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
    amount_thousands: int | None = None
    direction: Literal["up", "down"] | None = None
    sale_preview: SalePreviewV2 | None = None


class SupplyCardV2(StrictModelV2):
    card_ref: str
    card: VisibleCardV2


class SupplyPlacementV2(StrictModelV2):
    card_ref: str
    stockpile_id: int
    visibility: Literal["face_up", "face_down"]


class SupplyPlanV2(StrictModelV2):
    plan_id: str
    placements: list[SupplyPlacementV2]


class SupplyBatchV2(StrictModelV2):
    cards: list[SupplyCardV2]
    plans: list[SupplyPlanV2]


class PendingDecisionV2(StrictModelV2):
    kind: Literal[
        "supply",
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
    selected_stockpile_id: int | None = None
    selected_action_effect: str | None = None
    company_id: int | None = None


class PresentationCheckpointV2(StrictModelV2):
    checkpoint_id: str
    kind: Literal["demand_result", "round_result"]
    round: int


class PublicEventV2(StrictModelV2):
    event_id: int
    event_type: str
    cause: str | None = None
    round: int
    company_id: int | None = None
    company: str | None = None
    prior_price_dollars_per_share: int | None = None
    price_delta: int | None = None
    resulting_price_dollars_per_share: int | None = None
    forecast: int | Literal["DIVIDEND"] | None = None
    cash_effect_thousands: int | None = None
    direction: Literal["up", "down"] | None = None


class LiquidationLineV2(StrictModelV2):
    company_id: int
    company: str
    shares_thousands: int
    price_dollars_per_share: int
    value_thousands: int


class TerminalPlayerV2(StrictModelV2):
    player_id: int
    player_name: str
    cash_before_liquidation_thousands: int
    liquidation_value_thousands: int
    final_cash_thousands: int
    rank: int
    winner: bool
    liquidation: list[LiquidationLineV2]


class TerminalResultsV2(StrictModelV2):
    players: list[TerminalPlayerV2]
    winner_ids: list[int]


class ConfigurationV2(StrictModelV2):
    mode: Literal["lite"] = "lite"
    player_count: Literal[2] = 2
    round_count: Literal[6] = 6
    options: LiteOptionsV2


class ViewerV2(StrictModelV2):
    player_id: Literal[0] = 0
    name: Literal["YOU"] = "YOU"


class GameViewV2(StrictModelV2):
    schema_version: Literal["2.0"] = SCHEMA_VERSION_V2
    game_id: str
    revision: int
    configuration: ConfigurationV2
    round: int
    total_rounds: Literal[6] = 6
    phase: str
    phase_step: str
    viewer: ViewerV2
    active_player_id: int | None
    companies: list[CompanyV2]
    stockpiles: list[StockpileV2]
    players: list[PublicPlayerV2]
    private: ViewerPrivateV2
    pending_decision: PendingDecisionV2
    legal_actions: list[LegalActionV2]
    supply_batch: SupplyBatchV2 | None = None
    checkpoint: PresentationCheckpointV2 | None = None
    recent_events: list[PublicEventV2]
    terminal_results: TerminalResultsV2 | None = None
