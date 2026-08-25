"""Configurable Stockpile rules engine with a native OpenSpiel interface.

Money is represented in thousands of dollars.  The module deliberately keeps
the mutable :class:`GameState` compact; Pydantic is used only at configuration
and validation boundaries.

The ``lite`` rules profile keeps the complete six-company game and can add the
Market Impact phase on request, while omitting split, majority-shareholder,
and advanced-track mechanics. ``classic`` implements the standard base game
without Investors. ``deluxe`` adds the complete ten-card Investor deck.
"""

from __future__ import annotations

import base64
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import json
import math
import random
import time
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
import pyspiel


MONEY_UNIT = 1_000
COMPANY_NAMES = (
    "Cosmic Computers",
    "Bottomline Bank",
    "Leading Laboratories",
    "American Automotive",
    "Stanford Steel",
    "Epic Electric",
)
CLASSIC_FORECASTS: tuple[int | str, ...] = (4, 2, 1, "DIVIDEND", -2, -3)
CLASSIC_BIDS = (0, 1, 3, 6, 10, 15, 20, 25)
INVESTOR_NAMES = (
    "bill",
    "broker_bernie",
    "crazy_cramer",
    "discount_donald",
    "dividend_deborah",
    "golden_graham",
    "mayknow_martha",
    "maverick_mark",
    "secretive_stuart",
    "wise_warren",
)
INVESTOR_CASH = {
    "bill": 32,
    "broker_bernie": 16,
    "crazy_cramer": 18,
    "discount_donald": 13,
    "dividend_deborah": 20,
    "golden_graham": 16,
    "mayknow_martha": 22,
    "maverick_mark": 25,
    "secretive_stuart": 22,
    "wise_warren": 20,
}
INVESTOR_CASH_TWO_PLAYER = {
    "bill": 35,
    "broker_bernie": 16,
    "crazy_cramer": 12,
    "discount_donald": 4,
    "dividend_deborah": 14,
    "golden_graham": 6,
    "mayknow_martha": 20,
    "maverick_mark": 22,
    "secretive_stuart": 16,
    "wise_warren": 23,
}

OBSERVATION_STAGES = (
    "setup",
    "chance",
    "investor_select",
    "supply_mode",
    "supply_card",
    "supply_up_pile",
    "supply_down_pile",
    "investor_offer",
    "investor_target",
    "demand_martha_offer",
    "demand_pile",
    "demand_bid",
    "action_cramer_offer",
    "action_direction",
    "action_company",
    "selling",
    "movement",
    "dividend_claim",
    "deborah_company",
    "terminal",
)

# The values are board spaces, not arithmetic tracks.  Repeated values on the
# Steel and Electric tracks are intentionally preserved.
ADVANCED_TRACKS: tuple[tuple[int, ...], ...] = (
    (1, 3, 5, 7, 9),
    (1, 2, 3, 4, 5, 6),
    tuple(range(1, 11)),
    (1, 3, 5, 7, 9, 11, 13),
    (1, 1, 2, 2, 3, 3, 4, 4, 5, 5),
    (1, 3, 4, 5, 5, 6, 6, 7, 8, 10),
)
ADVANCED_START_INDEX = (2, 4, 4, 2, 8, 4)
ADVANCED_SPLIT_INDEX = (2, 3, 5, 3, 4, 3)


class RulesProfile(str, Enum):
    LITE = "lite"
    CLASSIC = "classic"
    DELUXE = "deluxe"

    # Source-compatible enum attributes. Iteration and serialization use only
    # the three canonical values above; textual aliases are normalized below.
    CORE = "classic"
    FULL = "deluxe"
    MINIMAL_TRAINING = "lite"
    EXPANDED = "deluxe"
    EXPANDED_VARIANTS = "deluxe"


class LiteOptionalRule(str, Enum):
    """The user-selectable additions to the Lite rules profile."""

    STARTING_SHARE = "starting_share"
    TRADING_FEES = "trading_fees"
    DIVIDENDS = "dividends"
    MARKET_IMPACT = "market_impact"


PROFILE_ALIASES = {
    "minimal_training": RulesProfile.LITE.value,
    "core": RulesProfile.CLASSIC.value,
    "full": RulesProfile.DELUXE.value,
    "expanded": RulesProfile.DELUXE.value,
    "expanded_variants": RulesProfile.DELUXE.value,
}


class Phase(str, Enum):
    SETUP = "setup"
    INFORMATION = "information"
    SUPPLY = "supply"
    DEMAND = "demand"
    ACTION = "action"
    SELLING = "selling"
    MOVEMENT = "movement"
    TERMINAL = "terminal"


class CardType(str, Enum):
    STOCK = "stock"
    TRADING_FEE = "trading_fee"
    ACTION = "action"
    COMPANY = "company"
    FORECAST = "forecast"
    INVESTOR = "investor"


FIXED_STOCKPILE_KEYS = frozenset(
    {
        "insider_information",
        "stockpile_construction",
        "ascending_auction",
        "secret_portfolios",
        "selling_phase",
        "price_movement",
        "terminal_liquidation",
    }
)
OPTIONAL_FEATURE_KEYS = (
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


class GameParameters(BaseModel):
    """Validated user-facing configuration.

    OpenSpiel itself accepts scalar parameters only.  The convenience model
    accepts Python lists inside ``rule_overrides`` and the game flattens them
    to comma-delimited strings for registration/serialization.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    player_count: int = Field(default=2, ge=2, le=5)
    rules_profile: RulesProfile | str = RulesProfile.LITE.value
    round_count: int = Field(default=6, ge=1, le=10)
    # Retained for loading old OpenSpiel strings. Canonical Deluxe presets
    # always enable all ten Investors; this scalar can no longer disable them.
    deluxe_investors: bool = False
    # Retained for loading old OpenSpiel strings. New callers should use the
    # profile and grouped rule overrides instead.
    board_side: Literal["standard", "advanced"] = "standard"
    investor_mode: Literal["none", "standard", "all"] = "none"
    rule_overrides: dict[str, Any] = Field(default_factory=dict)
    action_space_mode: Literal["compact", "shared"] = "compact"

    @field_validator("rules_profile")
    @classmethod
    def _known_profile(cls, value: RulesProfile | str) -> str:
        value = value.value if isinstance(value, RulesProfile) else str(value).strip().lower()
        value = PROFILE_ALIASES.get(value, value)
        if value not in {profile.value for profile in RulesProfile}:
            raise ValueError(f"unknown rules profile: {value}")
        return value

    @field_validator("rule_overrides")
    @classmethod
    def _locked_core(cls, value: dict[str, Any]) -> dict[str, Any]:
        unknown_locked = [key for key in FIXED_STOCKPILE_KEYS if key in value]
        if unknown_locked:
            raise ValueError(
                "Stockpile's fixed gameplay mechanics cannot be overridden: "
                + ", ".join(sorted(unknown_locked))
            )
        return dict(value)


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    legal: bool = True
    reachable: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    normalized_fields: dict[str, Any] = Field(default_factory=dict)


class InformationSetComplexity(BaseModel):
    """Result of a bounded reachable information-set traversal."""

    model_config = ConfigDict(extra="forbid")

    parameters: GameParameters
    exact: bool
    count_kind: Literal["exact", "lower_bound"]
    information_sets: int
    information_set_actions: int
    max_actions_per_information_set: int
    per_player_information_sets: dict[int, int]
    per_player_information_set_actions: dict[int, int]
    states_visited: int
    terminal_states: int
    chance_nodes: int
    elapsed_seconds: float
    max_states: int
    max_seconds: float
    truncation_reason: str | None = None


class InformationSetEnumerationLimit(RuntimeError):
    """Raised when an exact count exceeds its configured traversal budget."""

    def __init__(self, result: InformationSetComplexity):
        self.result = result
        super().__init__(
            "information-set enumeration did not finish within the "
            f"{result.truncation_reason or 'configured'} limit; "
            f"observed {result.information_sets} information sets"
        )


@dataclass(frozen=True, slots=True)
class Card:
    card_id: int
    card_type: str
    company_id: int | None = None
    value: int | str | None = None
    effect: str | None = None
    trigger: str | None = None
    face_up: bool = False
    owner: int | None = None
    location: str = "deck"


@dataclass(slots=True)
class PlayerState:
    player_id: int
    cash: int
    regular_portfolio: list[int]
    split_portfolio: list[int]
    fees: list[int] = field(default_factory=list)
    meeples: list[int | None] = field(default_factory=lambda: [None])
    bids: list[int] = field(default_factory=lambda: [0])
    private_information: list[tuple[int, int | str]] = field(default_factory=list)
    viewed_information: list[tuple[int, int | str]] = field(default_factory=list)
    investors: list[str] = field(default_factory=list)
    investor_offer: list[str] = field(default_factory=list)
    investor_candidates: list[str] = field(default_factory=list)
    acquired_actions: list[str] = field(default_factory=list)
    viewed_cards: set[int] = field(default_factory=set)
    known_cards: dict[int, Card] = field(default_factory=dict)
    phase_complete: bool = False
    revealed_information: list[tuple[int, int | str]] = field(default_factory=list)


@dataclass(slots=True)
class Stockpile:
    stockpile_id: int
    face_up_cards: list[Card] = field(default_factory=list)
    face_down_cards: list[Card] = field(default_factory=list)
    bid_level: int | None = None
    occupying_player: int | None = None
    occupying_token: int | None = None
    locked: bool = False
    purchaser: int | None = None


@dataclass(frozen=True, slots=True)
class LegalAction:
    action_id: int
    phase: str
    actor_ids: tuple[int, ...]
    action_type: str
    source_ids: tuple[int, ...] = ()
    target_ids: tuple[int, ...] = ()
    amount: int | None = None
    card_ids: tuple[int, ...] = ()
    reveal_count: int = 0
    payload: Mapping[str, Any] = field(default_factory=dict)
    display_label: str = ""


@dataclass(frozen=True, slots=True)
class InformationState:
    player_id: int | None
    public_state: Mapping[str, Any]
    owned_stocks: Mapping[str, Any]
    private_information: tuple[tuple[int, int | str], ...]
    legally_viewed_cards: tuple[int, ...]
    public_cash_and_bids: Mapping[str, Any]
    observable_history: tuple[Mapping[str, Any], ...]
    information_state_id: str
    tensor: tuple[float, ...]
    legal_action_ids: tuple[int, ...]
    private_hand: tuple[Card, ...] = ()
    known_cards: tuple[Card, ...] = ()
    private_investor_offer: tuple[str, ...] = ()
    public_investors: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    acquired_actions: tuple[str, ...] = ()
    viewed_information_pairs: tuple[tuple[int, int | str], ...] = ()

    @property
    def private_information_pairs(self) -> tuple[tuple[int, int | str], ...]:
        return self.private_information


@dataclass(frozen=True, slots=True)
class ActionRequest:
    player_id: int
    action_id: int
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActionRecord:
    player_id: int
    action_id: int
    phase: str
    action_type: str
    description: str
    sequence: int


@dataclass(frozen=True, slots=True)
class AutomaticEventRecord:
    event_type: str
    description: str
    sequence: int
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PresentationEventRecord:
    """Public, display-only event excluded from game information semantics."""

    presentation_sequence: int
    round: int
    event_type: str
    cause: str
    company_id: int
    company_name: str
    prior_price: int
    requested_delta: int | None
    actual_delta: int
    resulting_price: int
    forecast: int | str | None = None
    effect: str | None = None
    actor_id: int | None = None
    description: str = ""


@dataclass(frozen=True, slots=True)
class BidMarkerPresentation:
    """One public bidding position currently occupying a Stockpile."""

    stockpile_id: int
    player_id: int
    marker_index: int
    bid_value: int
    status: Literal["leading", "locked"]


@dataclass(frozen=True, slots=True)
class PresentationState:
    """Viewer-safe staged context that is not part of an information state."""

    phase: str
    stage: str
    current_actor: int | None
    demand_token: tuple[int, int] | None
    demand_pile: int | None
    supply_choice: int | None
    supply_up_pile: int | None
    selected_direction: str | None
    selling_company: int | None
    stockpile_markers: tuple[BidMarkerPresentation, ...]


@dataclass(frozen=True, slots=True)
class SalePreview:
    """Authoritative, non-mutating proceeds and holdings preview for a sale."""

    action_id: int
    action_type: str
    label: str
    company_id: int
    company_name: str
    quantity_sold: int
    unit_price: int
    gross_value: int
    resulting_regular: int
    resulting_split: int
    resulting_represented: int


@dataclass(frozen=True, slots=True)
class TerminalCompanyLiquidation:
    company_id: int
    company_name: str
    represented_shares: int
    unit_price: int
    value: int


@dataclass(frozen=True, slots=True)
class TerminalPlayerLiquidation:
    player_id: int
    companies: tuple[TerminalCompanyLiquidation, ...]
    liquidation_value: int
    final_cash: int
    rank: int
    winner: bool


@dataclass(frozen=True, slots=True)
class GameResult:
    final_cash_by_player: Mapping[int, int]
    majority_shareholders: Mapping[int, tuple[int, ...]]
    bonuses: Mapping[int, int]
    liquidation_values: Mapping[int, int]
    tie_break: str | None
    winner_ids: tuple[int, ...]
    utilities: tuple[float, ...]


@dataclass(slots=True)
class Playthrough:
    initial_state: str
    state_snapshots: list[str]
    action_records: list[ActionRecord]
    automatic_event_records: list[AutomaticEventRecord]
    terminal_state: str | None
    game_result: GameResult | None
    random_seed: int | None
    completion_status: Literal["complete", "awaiting_action", "invalid"]


@dataclass(frozen=True, slots=True)
class InitialInput:
    first_player: int
    market_deck_order: tuple[Card, ...]
    starting_shares: tuple[int, ...]
    round_information: tuple[tuple[tuple[int, int | str], ...], ...]
    investor_deals: tuple[tuple[str, ...], ...] = ()
    starting_prices: tuple[int, ...] = ()
    starting_stockpiles: tuple[Any, ...] = ()
    company_forecast_deals: tuple[Any, ...] = ()
    chance_outcomes: tuple[int, ...] = ()
    random_seed: int | None = None

    @property
    def deck_order(self) -> tuple[Card, ...]:
        return self.market_deck_order


@dataclass(frozen=True, slots=True)
class ActionCodec:
    """A stable, typed action catalog for one configured game."""

    ranges: Mapping[str, range]
    num_distinct_actions: int
    shared_action_head: int

    def ids(self, namespace: str) -> tuple[int, ...]:
        return tuple(self.ranges.get(namespace, ()))

    def offset(self, namespace: str) -> int:
        values = self.ranges.get(namespace)
        if values is None:
            raise KeyError(namespace)
        return values.start

    def decode(self, action: int) -> tuple[str, int]:
        for name, values in self.ranges.items():
            if action in values:
                return name, action - values.start
        raise ValueError(f"unknown action id {action}")


@dataclass(frozen=True, slots=True)
class RuleSet:
    profile: Literal["lite", "classic", "deluxe"]
    player_count: int
    company_count: int
    round_count: int
    company_names: tuple[str, ...]
    starting_shares_per_player: int
    starting_cash: int
    starting_price: int
    standard_price_ceiling: int | None
    bid_values: tuple[int, ...]
    forecast_values: tuple[int | str, ...]
    stockpile_count: int
    meeples_per_player: int
    supply_batches: int
    private_pairs_per_player: int
    two_player_topology: Literal["simple", "standard", "official"]
    trading_fees: bool
    market_action_cards: bool
    stock_boom_cards: bool
    stock_bust_cards: bool
    forecast_dividends: bool
    dividend_reveal_choice: bool
    stock_splits: bool
    repeat_split_bonus: bool
    bankruptcy: bool
    majority_bonus: bool
    blind_information_pairs: bool
    partial_sales: bool
    sequential_observable_selling: bool
    advanced_price_tracks: bool
    advanced_track_dividends: bool
    investors: bool
    enabled_investors: tuple[str, ...]
    action_space_mode: Literal["compact", "shared"]
    action_codec: ActionCodec
    max_legal_actions: int
    max_chance_outcomes: int
    max_game_length: int

    @property
    def phase_order(self) -> tuple[str, ...]:
        phases = [Phase.INFORMATION.value, Phase.SUPPLY.value, Phase.DEMAND.value]
        if self.market_action_cards or self.investors:
            phases.append(Phase.ACTION.value)
        phases.extend((Phase.SELLING.value, Phase.MOVEMENT.value))
        return tuple(phases)


@dataclass(frozen=True, slots=True)
class ConfiguredGame:
    parameters: GameParameters
    rule_set: RuleSet
    game: "StockpileGame"
    parameter_schema: Mapping[str, Any]
    state_schema: Mapping[str, Any]


def _profile_defaults(profile: str, players: int) -> dict[str, Any]:
    """Return the fixed board/rule contract for a canonical profile."""

    values: dict[str, Any] = {
        "company_count": 6,
        "starting_shares_per_player": 1,
        "starting_cash": 30 if players == 2 else 20,
        "starting_price": 5,
        "standard_price_ceiling": (
            None if profile == RulesProfile.LITE.value else 10
        ),
        "bid_values": list(CLASSIC_BIDS),
        "forecast_values": list(CLASSIC_FORECASTS),
        "two_player_topology": "official" if players == 2 else "standard",
        "trading_fees": True,
        "market_action_cards": True,
        "stock_boom_cards": True,
        "stock_bust_cards": True,
        "forecast_dividends": True,
        "dividend_reveal_choice": True,
        "stock_splits": True,
        "repeat_split_bonus": True,
        "bankruptcy": True,
        "majority_bonus": True,
        "blind_information_pairs": True,
        "partial_sales": True,
        "sequential_observable_selling": True,
        "advanced_price_tracks": False,
        "advanced_track_dividends": False,
        "investors": profile == RulesProfile.DELUXE.value,
        "enabled_investors": (
            list(INVESTOR_NAMES) if profile == RulesProfile.DELUXE.value else []
        ),
    }
    if profile == RulesProfile.LITE.value:
        values.update(
            {
                "trading_fees": False,
                "starting_shares_per_player": 0,
                "market_action_cards": False,
                "stock_boom_cards": False,
                "stock_bust_cards": False,
                "forecast_dividends": False,
                "dividend_reveal_choice": False,
                "stock_splits": False,
                "repeat_split_bonus": False,
                "majority_bonus": False,
                "sequential_observable_selling": False,
                "advanced_price_tracks": False,
                "advanced_track_dividends": False,
            }
        )
    return values


def _normalise_forecast(value: Any) -> int | str:
    if isinstance(value, str):
        stripped = value.strip().upper()
        if stripped in {"$$", "DIVIDEND"}:
            return "DIVIDEND"
        return int(stripped)
    return int(value)


def _make_action_codec(
    profile: str,
    players: int,
    companies: int,
    piles: int,
    bids: int,
    enabled: Mapping[str, bool],
    action_space_mode: str,
) -> ActionCodec:
    del profile  # The catalog follows effective mechanics, not a profile name.
    investor_actions = bool(enabled.get("investors"))
    company_actions = bool(enabled.get("market_action_cards")) or investor_actions
    direction_actions = bool(enabled.get("market_action_cards")) or investor_actions
    card_namespace = "card_ordinal" if investor_actions else "card_slot"
    card_choices = (6 if players == 2 else players + 1) if investor_actions else 2
    sizes: list[tuple[str, int]] = [
        (card_namespace, card_choices),
        ("pile", piles),
        ("bid_level", bids),
    ]
    if company_actions:
        sizes.append(("company", companies))
    sizes.append(("done", 1))
    if direction_actions:
        sizes.append(("direction", 2))
    if enabled.get("partial_sales"):
        sizes.append(("sale_mode", 3))
    else:
        sizes.append(("sell_all", 1))
    if enabled.get("dividend_reveal_choice"):
        sizes.append(("dividend_claim", 2))
    if investor_actions:
        investor_slots = 2 if players == 2 else 0
        sizes.extend(
            [
                ("use_ability", 1),
                ("investor_slot", investor_slots),
                ("card_face", 2),
                ("info_target", 4),
            ]
        )

    compact_total = sum(size for _, size in sizes)
    # Every non-Investor configuration can share the Classic policy head.
    # The larger head is reserved exclusively for the Investor expansion;
    # advanced company tracks do not introduce player actions by themselves.
    shared = max(42 if investor_actions else 29, compact_total)

    cursor = 0
    ranges: dict[str, range] = {}
    for name, size in sizes:
        ranges[name] = range(cursor, cursor + size)
        cursor += size
    total = shared if action_space_mode == "shared" else compact_total
    return ActionCodec(ranges=ranges, num_distinct_actions=total, shared_action_head=shared)


def _normalise_rule_set(parameters: GameParameters) -> RuleSet:
    profile = (
        parameters.rules_profile.value
        if isinstance(parameters.rules_profile, RulesProfile)
        else str(parameters.rules_profile)
    )
    profile = PROFILE_ALIASES.get(profile, profile)
    overrides = dict(parameters.rule_overrides)
    values = _profile_defaults(profile, parameters.player_count)

    # ``round_count`` in an old encoded override remains authoritative so old
    # OpenSpiel strings continue to replay. New callers use the typed field (or
    # the OpenSpiel ``rounds`` scalar) and should not duplicate it in overrides.
    rounds = int(overrides.pop("round_count", parameters.round_count))

    # These switches remain decodable for old game strings. The new preset
    # factory constrains them to the appropriate profile, while the lower-level
    # model keeps enough flexibility to replay legacy Flex configurations.
    if parameters.board_side == "advanced":
        values["advanced_price_tracks"] = True
    if parameters.investor_mode != "none":
        values["investors"] = True
        values["enabled_investors"] = list(INVESTOR_NAMES)
    if parameters.deluxe_investors:
        if profile != RulesProfile.DELUXE.value:
            raise ValueError("deluxe_investors is available only in Deluxe")
        values["investors"] = True
        values["enabled_investors"] = list(INVESTOR_NAMES)

    # Friendly grouped options are the canonical customization layer for every
    # profile. Low-level names remain decodable for historical game strings.
    friendly_aliases = {
        "hand": ("starting_shares_per_player",),
        "fees": ("trading_fees",),
        "dividend": ("forecast_dividends", "dividend_reveal_choice"),
        "dividends": ("forecast_dividends", "dividend_reveal_choice"),
        "impact": (
            "market_action_cards",
            "stock_boom_cards",
            "stock_bust_cards",
        ),
        "split": ("stock_splits", "repeat_split_bonus"),
        "majority": ("majority_bonus",),
        "stock_tracks": ("advanced_price_tracks",),
        "tracks": ("advanced_price_tracks",),
        "sell_order": ("sequential_observable_selling",),
        "starting_share": ("starting_shares_per_player",),
    }
    for friendly, targets in friendly_aliases.items():
        if friendly not in overrides:
            continue
        enabled_value = _decode_bool_scalar(overrides.pop(friendly), name=friendly)
        for target in targets:
            overrides.setdefault(
                target,
                int(enabled_value)
                if target == "starting_shares_per_player"
                else enabled_value,
            )
    if (
        profile != RulesProfile.LITE.value
        and "impact" in parameters.rule_overrides
    ):
        raise ValueError("impact is configurable only in Lite")
    if "stock_splits" in overrides and "repeat_split_bonus" not in overrides:
        overrides["repeat_split_bonus"] = overrides["stock_splits"]

    # Explicit low-level overrides intentionally take final precedence. This
    # path is not presented by the terminal UI, but it is necessary for old
    # serialized custom games and for programmer-level experiments.
    values.update(overrides)
    # OpenSpiel scalar parameters and legacy serialized overrides may carry
    # booleans as strings. Canonicalize every feature switch before any
    # derived-rule or profile validation so ``"false"`` cannot become truthy
    # through a later generic ``bool(...)`` conversion.
    for key in OPTIONAL_FEATURE_KEYS:
        values[key] = _decode_bool_scalar(values[key], name=key)
    expected_price_ceiling = None if profile == RulesProfile.LITE.value else 10
    if (
        "standard_price_ceiling" in overrides
        and values["standard_price_ceiling"] != expected_price_ceiling
    ):
        raise ValueError(
            "standard_price_ceiling is fixed by the selected rules profile"
        )
    values["standard_price_ceiling"] = expected_price_ceiling
    if "advanced_track_dividends" not in overrides:
        values["advanced_track_dividends"] = bool(
            values.get("advanced_price_tracks")
            and values.get("forecast_dividends")
        )
    if values.get("investors") and not values.get("enabled_investors"):
        values["enabled_investors"] = list(INVESTOR_NAMES)
    if not values.get("investors") and "enabled_investors" not in overrides:
        values["enabled_investors"] = []

    # Umbrella action-card control governs both card types unless individual
    # card switches explicitly turn the action phase back on.
    if not bool(values.get("market_action_cards", False)):
        if "stock_boom_cards" not in overrides:
            values["stock_boom_cards"] = False
        if "stock_bust_cards" not in overrides:
            values["stock_bust_cards"] = False
    values["market_action_cards"] = bool(
        values.get("stock_boom_cards", False) or values.get("stock_bust_cards", False)
    )

    if profile == RulesProfile.LITE.value:
        unsupported_lite_rules = {
            "stock_splits": "stock splits",
            "repeat_split_bonus": "repeat-split bonuses",
            "majority_bonus": "majority-shareholder bonuses",
            "advanced_price_tracks": "advanced price tracks",
            "advanced_track_dividends": "advanced-track dividends",
        }
        enabled_unsupported = [
            label
            for key, label in unsupported_lite_rules.items()
            if _decode_bool_scalar(values.get(key, False), name=key)
        ]
        if enabled_unsupported:
            raise ValueError(
                "Lite does not support " + ", ".join(enabled_unsupported)
            )

    companies = int(values["company_count"])
    starting_shares_per_player = int(values.get("starting_shares_per_player", 1))
    if not 2 <= parameters.player_count <= 5:
        raise ValueError("Stockpile requires 2-5 players")
    if not parameters.player_count + 1 <= companies <= 6:
        raise ValueError(
            "company_count must be between player_count + 1 and 6 so each "
            "player has private information and a public pair"
        )
    if not 1 <= rounds <= 10:
        raise ValueError("round_count must be between 1 and 10")
    if starting_shares_per_player not in {0, 1}:
        raise ValueError("starting_shares_per_player must be 0 or 1")

    bids = tuple(int(value) for value in values["bid_values"])
    if len(bids) < 2 or tuple(sorted(set(bids))) != bids or bids[0] != 0:
        raise ValueError("bid_values must be a strictly increasing sequence starting at 0")
    forecasts = tuple(_normalise_forecast(value) for value in values["forecast_values"])
    if len(forecasts) != companies:
        raise ValueError("forecast_values must contain exactly one value per company")

    topology = str(values["two_player_topology"])
    if topology == "official" and (parameters.player_count != 2 or companies != 6):
        raise ValueError("official two-player topology requires two players and six companies")
    if topology not in {"simple", "standard", "official"}:
        raise ValueError("two_player_topology must be simple, standard, or official")
    if values.get("advanced_price_tracks") and companies != 6:
        raise ValueError("advanced price tracks require all six companies")

    enabled_investors = tuple(str(name).lower() for name in values.get("enabled_investors", ()))
    if len(set(enabled_investors)) != len(enabled_investors):
        raise ValueError("enabled_investors must not contain duplicates")
    unknown_investors = sorted(set(enabled_investors) - set(INVESTOR_NAMES))
    if unknown_investors:
        raise ValueError("unknown Investors: " + ", ".join(unknown_investors))
    if values.get("repeat_split_bonus") and not values.get("stock_splits"):
        raise ValueError("repeat_split_bonus requires stock_splits")
    if values.get("dividend_reveal_choice") and not values.get("forecast_dividends"):
        raise ValueError("dividend_reveal_choice requires forecast_dividends")
    if values.get("advanced_track_dividends") and not (
        values.get("advanced_price_tracks") and values.get("forecast_dividends")
    ):
        raise ValueError(
            "advanced_track_dividends requires advanced_price_tracks and forecast_dividends"
        )
    if values.get("investors"):
        deal_size = 4 if parameters.player_count == 2 else 2
        required_investors = deal_size * parameters.player_count
        if len(enabled_investors) < required_investors:
            raise ValueError(
                "investor setup requires at least "
                f"{required_investors} unique enabled Investors"
            )
    elif enabled_investors:
        raise ValueError("enabled_investors requires investors=True")

    if not values.get("forecast_dividends") and "DIVIDEND" in forecasts:
        forecasts = tuple(0 if value == "DIVIDEND" else value for value in forecasts)
    if values.get("forecast_dividends") and "DIVIDEND" not in forecasts:
        raise ValueError("forecast_dividends requires a DIVIDEND/$$ forecast value")

    official_two = parameters.player_count == 2 and topology == "official"
    piles = 4 if official_two else parameters.player_count
    meeples = 2 if official_two else 1
    batches = 2 if official_two else 1
    private_pairs = 2 if official_two else 1
    enabled = {name: bool(values.get(name, False)) for name in OPTIONAL_FEATURE_KEYS}
    codec = _make_action_codec(
        profile,
        parameters.player_count,
        companies,
        piles,
        len(bids),
        enabled,
        parameters.action_space_mode,
    )
    investor_offer = 4 if enabled["investors"] and parameters.player_count == 2 else 2
    company_choice = (
        companies
        if enabled["market_action_cards"] or "crazy_cramer" in enabled_investors
        else 0
    )
    max_legal = max(
        2,
        piles,
        len(bids),
        company_choice,
        4 if enabled["partial_sales"] else 2,
        investor_offer if enabled["investors"] else 0,
        4 if "mayknow_martha" in enabled_investors else 0,
        companies + 1 if "dividend_deborah" in enabled_investors else 0,
    )
    market_types = companies
    if enabled["trading_fees"]:
        market_types += 3
    if enabled["stock_boom_cards"]:
        market_types += 1
    if enabled["stock_bust_cards"]:
        market_types += 1
    max_chance = max(companies, len(forecasts), market_types, len(enabled_investors))

    # The bound is intentionally conservative and finite. Player choices only;
    # chance nodes do not count in OpenSpiel's max_game_length. Sealed selling
    # visits every player/company pair even when the portfolio is empty, and a
    # player can make one decision per represented stock card before holding.
    cards_per_round = piles + 2 * parameters.player_count * batches
    selling_actions_per_round = (
        companies * parameters.player_count
        + 2 * cards_per_round * rounds
    )
    max_game_length = rounds * (
        3 * parameters.player_count * batches
        + 3 * piles * len(bids)
        + 3 * cards_per_round
        + selling_actions_per_round
        + 40
    )

    return RuleSet(
        profile=cast(Literal["lite", "classic", "deluxe"], profile),
        player_count=parameters.player_count,
        company_count=companies,
        round_count=rounds,
        company_names=tuple(COMPANY_NAMES[:companies]),
        starting_shares_per_player=starting_shares_per_player,
        starting_cash=int(values["starting_cash"]),
        starting_price=int(values.get("starting_price", 5)),
        standard_price_ceiling=cast(int | None, values["standard_price_ceiling"]),
        bid_values=bids,
        forecast_values=forecasts,
        stockpile_count=piles,
        meeples_per_player=meeples,
        supply_batches=batches,
        private_pairs_per_player=private_pairs,
        two_player_topology=cast(
            Literal["simple", "standard", "official"], topology
        ),
        trading_fees=enabled["trading_fees"],
        market_action_cards=bool(values["market_action_cards"]),
        stock_boom_cards=enabled["stock_boom_cards"],
        stock_bust_cards=enabled["stock_bust_cards"],
        forecast_dividends=enabled["forecast_dividends"],
        dividend_reveal_choice=enabled["dividend_reveal_choice"],
        stock_splits=enabled["stock_splits"],
        repeat_split_bonus=enabled["repeat_split_bonus"],
        bankruptcy=enabled["bankruptcy"],
        majority_bonus=enabled["majority_bonus"],
        blind_information_pairs=enabled["blind_information_pairs"],
        partial_sales=enabled["partial_sales"],
        sequential_observable_selling=_decode_bool_scalar(
            values.get("sequential_observable_selling", True),
            name="sequential_observable_selling",
        ),
        advanced_price_tracks=enabled["advanced_price_tracks"],
        advanced_track_dividends=enabled["advanced_track_dividends"],
        investors=enabled["investors"],
        enabled_investors=enabled_investors,
        action_space_mode=cast(
            Literal["compact", "shared"], parameters.action_space_mode
        ),
        action_codec=codec,
        max_legal_actions=max_legal,
        max_chance_outcomes=max_chance,
        max_game_length=max_game_length,
    )


def _market_templates(rule_set: RuleSet) -> list[tuple[str, int | None, int | str | None]]:
    templates: list[tuple[str, int | None, int | str | None]] = []
    for company in range(rule_set.company_count):
        templates.extend((CardType.STOCK.value, company, 1) for _ in range(10))
    if rule_set.trading_fees:
        for fee in (-1, -2, -3):
            templates.extend((CardType.TRADING_FEE.value, None, fee) for _ in range(4))
    if rule_set.stock_boom_cards:
        templates.extend((CardType.ACTION.value, None, "boom") for _ in range(4))
    if rule_set.stock_bust_cards:
        templates.extend((CardType.ACTION.value, None, "bust") for _ in range(4))
    return templates


def _build_market_deck(rule_set: RuleSet) -> list[Card]:
    required = rule_set.round_count * (
        rule_set.stockpile_count
        + 2 * rule_set.player_count * rule_set.supply_batches
    )
    starting_share_cards = (
        rule_set.player_count * rule_set.starting_shares_per_player
    )
    templates = _market_templates(rule_set)
    if not templates:
        raise ValueError("the Market deck cannot be empty")
    copies = max(1, math.ceil((required + starting_share_cards) / len(templates)))
    # Always add complete deck copies. The template is grouped by company and
    # card type, so truncating a partial repeated copy would systematically
    # overrepresent its early companies even after shuffling.
    expanded = templates * copies
    return [
        Card(
            card_id=index,
            card_type=kind,
            company_id=company,
            value=value,
            effect=str(value) if kind == CardType.ACTION.value else None,
        )
        for index, (kind, company, value) in enumerate(expanded)
    ]


def _card_key(card: Card) -> tuple[str, int | None, int | str | None]:
    return card.card_type, card.company_id, card.value


def _relocate_card(
    card: Card,
    *,
    location: str,
    face_up: bool | None = None,
    owner: int | None = None,
) -> Card:
    return replace(
        card,
        location=location,
        face_up=card.face_up if face_up is None else face_up,
        owner=owner,
    )


def _investor_deals(rule_set: RuleSet, rng: random.Random) -> tuple[tuple[str, ...], ...]:
    if not rule_set.investors:
        return ()
    deal_size = 4 if rule_set.player_count == 2 else 2
    required = deal_size * rule_set.player_count
    deck = list(rule_set.enabled_investors)
    if len(deck) < required:
        raise ValueError(
            f"investor setup requires {required} unique enabled Investors"
        )
    rng.shuffle(deck)
    return tuple(
        tuple(deck[player * deal_size : (player + 1) * deal_size])
        for player in range(rule_set.player_count)
    )


def randomize_initial_input(rule_set: RuleSet, random_seed: int | None = None) -> InitialInput:
    """Samples a deterministic standalone setup for replay and unit tests."""

    rng = random.Random(random_seed)
    starting_shares = list(range(rule_set.company_count))
    rng.shuffle(starting_shares)
    starting_shares = starting_shares[
        : rule_set.player_count * rule_set.starting_shares_per_player
    ]

    deck = _build_market_deck(rule_set)
    # Starting shares are removed from one copy of each selected company.
    for company in starting_shares:
        index = next(
            index
            for index, card in enumerate(deck)
            if card.card_type == CardType.STOCK.value and card.company_id == company
        )
        deck.pop(index)
    rng.shuffle(deck)

    round_information: list[tuple[tuple[int, int | str], ...]] = []
    for _ in range(rule_set.round_count):
        companies = list(range(rule_set.company_count))
        forecasts = list(rule_set.forecast_values)
        rng.shuffle(companies)
        rng.shuffle(forecasts)
        round_information.append(tuple(zip(companies, forecasts, strict=True)))

    prices = (
        tuple(
            ADVANCED_TRACKS[company][ADVANCED_START_INDEX[company]]
            for company in range(rule_set.company_count)
        )
        if rule_set.advanced_price_tracks
        else tuple(rule_set.starting_price for _ in range(rule_set.company_count))
    )
    return InitialInput(
        first_player=rng.randrange(rule_set.player_count),
        market_deck_order=tuple(deck),
        starting_shares=tuple(starting_shares),
        round_information=tuple(round_information),
        investor_deals=_investor_deals(rule_set, rng),
        starting_prices=prices,
        company_forecast_deals=tuple(round_information),
        random_seed=random_seed,
    )


_PARAMETER_SPECIFICATION = {
    "players": 2,
    "rules_profile": RulesProfile.LITE.value,
    "rounds": 6,
    "deluxe_investors": False,
    "board_side": "standard",
    "investor_mode": "none",
    "action_space_mode": "compact",
    "rule_overrides_json": "",
}


def _encode_rule_overrides(overrides: Mapping[str, Any]) -> str:
    """Encode override JSON into an OpenSpiel game-string-safe scalar."""

    if not overrides:
        return ""
    payload = json.dumps(overrides, sort_keys=True, separators=(",", ":")).encode()
    return "b64_" + base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_rule_overrides(value: str) -> dict[str, Any]:
    """Decode current base64 parameters while accepting legacy raw JSON."""

    if not value:
        return {}
    if not value.startswith("b64_"):
        loaded = json.loads(value)
    else:
        encoded = value.removeprefix("b64_")
        encoded += "=" * (-len(encoded) % 4)
        loaded = json.loads(base64.urlsafe_b64decode(encoded).decode())
    if not isinstance(loaded, dict):
        raise ValueError("rule_overrides_json must encode a JSON object")
    return loaded


def _decode_bool_scalar(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    raise ValueError(f"{name} must be boolean")


_GAME_TYPE = pyspiel.GameType(
    short_name="python_stockpile",
    long_name="Python Stockpile",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=5,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True,
    default_loadable=True,
    parameter_specification=_PARAMETER_SPECIFICATION,
)


class StockpileGame(pyspiel.Game):
    """OpenSpiel game definition for one normalized :class:`RuleSet`."""

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        parameters: GameParameters | None = None,
        rule_set: RuleSet | None = None,
    ) -> None:
        raw = dict(params or {})
        if parameters is None:
            override_text = str(raw.get("rule_overrides_json", "") or "")
            overrides = _decode_rule_overrides(override_text)
            parameters = GameParameters(
                player_count=int(raw.get("players", 2)),
                rules_profile=str(raw.get("rules_profile", "lite")),
                round_count=int(raw.get("rounds", 6)),
                deluxe_investors=_decode_bool_scalar(
                    raw.get("deluxe_investors", False),
                    name="deluxe_investors",
                ),
                board_side=cast(
                    Literal["standard", "advanced"],
                    str(raw.get("board_side", "standard")),
                ),
                investor_mode=cast(
                    Literal["none", "standard", "all"],
                    str(raw.get("investor_mode", "none")),
                ),
                action_space_mode=cast(
                    Literal["compact", "shared"],
                    str(raw.get("action_space_mode", "compact")),
                ),
                rule_overrides=overrides,
            )
        self.parameters_model = parameters
        self.rule_set = rule_set or _normalise_rule_set(parameters)
        open_spiel_params = {
            "players": self.rule_set.player_count,
            "rules_profile": self.rule_set.profile,
            "rounds": self.rule_set.round_count,
            # Keep the new explicit toggle separate from a legacy Investor mode
            # or a low-level override. Otherwise serializing an old custom game
            # could reload it as a different Deluxe preset.
            "deluxe_investors": bool(
                parameters.deluxe_investors
                or (
                    self.rule_set.profile == RulesProfile.DELUXE.value
                    and self.rule_set.investors
                )
            ),
            "board_side": "advanced" if self.rule_set.advanced_price_tracks else "standard",
            "investor_mode": parameters.investor_mode,
            "action_space_mode": self.rule_set.action_space_mode,
            "rule_overrides_json": _encode_rule_overrides(parameters.rule_overrides),
        }
        game_info = pyspiel.GameInfo(
            num_distinct_actions=self.rule_set.action_codec.num_distinct_actions,
            max_chance_outcomes=self.rule_set.max_chance_outcomes,
            num_players=self.rule_set.player_count,
            min_utility=-1.0,
            max_utility=1.0,
            utility_sum=0.0,
            max_game_length=self.rule_set.max_game_length,
        )
        super().__init__(_GAME_TYPE, game_info, open_spiel_params)

    def new_initial_state(self) -> "GameState":
        return GameState(self)

    def make_py_observer(
        self,
        iig_obs_type: pyspiel.IIGObservationType | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> "StockpileObserver":
        return StockpileObserver(
            self,
            iig_obs_type
            or pyspiel.IIGObservationType(
                perfect_recall=False,
                public_info=True,
                private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER,
            ),
            params,
        )


class GameState(pyspiel.State):
    """Mutable, explicit-chance OpenSpiel state.

    Callers that need persistent transitions should use :func:`advance_game` or
    ``state.child(action)``.  Native ``apply_action`` follows OpenSpiel and
    mutates this object in place.
    """

    def __init__(self, game: StockpileGame, initial_input: InitialInput | None = None):
        super().__init__(game)
        self.rule_set = game.rule_set
        self.round = 1
        self.first_player = 0
        self.phase = Phase.SETUP.value
        self.stage = "starting_share"
        self.current_actor = pyspiel.PlayerId.CHANCE
        self.players = [
            PlayerState(
                player_id=player,
                cash=self.rule_set.starting_cash,
                regular_portfolio=[0] * self.rule_set.company_count,
                split_portfolio=[0] * self.rule_set.company_count,
                meeples=[None] * self.rule_set.meeples_per_player,
                bids=[0] * self.rule_set.meeples_per_player,
            )
            for player in range(self.rule_set.player_count)
        ]
        self.prices = {
            name: self.rule_set.starting_price for name in self.rule_set.company_names
        }
        self.price_indices = [
            ADVANCED_START_INDEX[index] if self.rule_set.advanced_price_tracks else 4
            for index in range(self.rule_set.company_count)
        ]
        if self.rule_set.advanced_price_tracks:
            self.prices = {
                self.rule_set.company_names[index]: ADVANCED_TRACKS[index][self.price_indices[index]]
                for index in range(self.rule_set.company_count)
            }
        self.stockpiles = [Stockpile(index) for index in range(self.rule_set.stockpile_count)]
        self.public_information: list[tuple[int, int | str]] = []
        self.blind_information: list[tuple[int, int | str]] = []
        self.revealed_information: list[tuple[int, int | str]] = []
        self.discards: list[Card] = []
        self.history_records: list[dict[str, Any]] = []
        self.pending_events: list[AutomaticEventRecord] = []
        # Presentation events deliberately use an independent sequence and are
        # excluded from information states, tensors, action histories, and
        # state serialization. Replaying explicit OpenSpiel actions rebuilds
        # the journal deterministically in :meth:`clone`.
        self._presentation_events: list[PresentationEventRecord] = []
        self._presentation_sequence = 0
        self.terminal_status = False
        self._sequence = 0
        self._preset = initial_input
        self._preset_deck: deque[Card] = deque()
        self._market_counts: Counter[tuple[str, int | None, int | str | None]] = Counter()
        self._next_card_id = 100_000
        self._starting_remaining = list(range(self.rule_set.company_count))
        # This is a deal ordinal; with the standard one-share setup it is also
        # the receiving player id.
        self._starting_player = 0
        self._info_remaining_companies: list[int] = []
        self._info_remaining_forecasts: list[int | str] = []
        self._info_pairs: list[tuple[int, int | str]] = []
        self._pending_info_company: int | None = None
        self._chance_kind = "starting_share"
        self._market_targets: deque[tuple[str, int, int]] = deque()
        self._market_resume_batch: int | None = None
        self._current_supply_batch = 0
        self._hands: list[list[Card]] = [[] for _ in self.players]
        self._supply_order: deque[tuple[int, int]] = deque()
        self._supply_choice: int | None = None
        self._supply_up_pile: int | None = None
        self._supply_both_down = False
        self._last_supply_public: dict[str, Any] | None = None
        self._demand_queue: deque[tuple[int, int]] = deque()
        self._demand_token: tuple[int, int] | None = None
        self._demand_pile: int | None = None
        self._demand_pending_actor: int | None = None
        self._action_players: deque[int] = deque()
        self._selected_direction: str | None = None
        self._cramer_deferred: set[tuple[int, int]] = set()
        self._selling_players: deque[int] = deque()
        self._selling_company = 0
        self._selling_shadow_regular: list[list[int]] = []
        self._selling_shadow_split: list[list[int]] = []
        self._private_sale_history: list[list[dict[str, Any]]] = [
            [] for _ in self.players
        ]
        self._movement_pairs: deque[tuple[int, int | str]] = deque()
        self._dividend_players: deque[tuple[int, int]] = deque()
        self._dividend_company: int | None = None
        self.public_dividend_claims: list[dict[str, int]] = []
        self._investor_used: set[tuple[int, str, int]] = set()
        self._investor_remaining: list[str] = []
        self._investor_deal_order: deque[int] = deque()
        self._investor_selection_players: deque[int] = deque()
        self._investors_revealed = not self.rule_set.investors
        self._current_investor_ability: str | None = None
        self._investor_source = 0
        self._investor_face = 0
        self._investor_scan_index = 0
        self._investor_card_ordinal = 0

        if initial_input is not None:
            self._load_preset(initial_input)
        else:
            for card in _build_market_deck(self.rule_set):
                self._market_counts[_card_key(card)] += 1
            if self.rule_set.investors:
                self._investor_remaining = list(self.rule_set.enabled_investors)
            if self.rule_set.starting_shares_per_player == 0:
                self.stage = "first_player"
                self._chance_kind = "first_player"

    @property
    def acting_players(self) -> tuple[int, ...]:
        return () if self.current_actor < 0 else (int(self.current_actor),)

    @property
    def bids(self) -> tuple[tuple[int | None, int | None], ...]:
        return tuple((pile.occupying_player, pile.bid_level) for pile in self.stockpiles)

    @property
    def public_cash(self) -> tuple[int, ...]:
        return tuple(player.cash for player in self.players)

    def _load_preset(self, initial_input: InitialInput) -> None:
        self._chance_kind = ""
        self.first_player = int(initial_input.first_player) % self.rule_set.player_count
        self._preset_deck = deque(initial_input.market_deck_order)
        expected_starting_shares = (
            self.rule_set.player_count * self.rule_set.starting_shares_per_player
        )
        if len(initial_input.starting_shares) != expected_starting_shares:
            raise ValueError(
                "initial input must contain exactly "
                f"{expected_starting_shares} starting shares"
            )
        if len(set(initial_input.starting_shares)) != len(initial_input.starting_shares):
            raise ValueError("starting shares must use distinct companies")
        if any(
            not 0 <= company < self.rule_set.company_count
            for company in initial_input.starting_shares
        ):
            raise ValueError("starting share company is out of range")
        for deal_index, company in enumerate(initial_input.starting_shares):
            player = deal_index // max(1, self.rule_set.starting_shares_per_player)
            self.players[player].regular_portfolio[company] += 1
        if initial_input.starting_prices:
            if len(initial_input.starting_prices) != self.rule_set.company_count:
                raise ValueError(
                    "initial input must contain one starting price per company"
                )
            if self.rule_set.advanced_price_tracks:
                marked_prices = tuple(
                    ADVANCED_TRACKS[company][self.price_indices[company]]
                    for company in range(self.rule_set.company_count)
                )
                if tuple(initial_input.starting_prices) != marked_prices:
                    raise ValueError(
                        "advanced starting prices must match their marked track spaces"
                    )
            self.prices = {
                self.rule_set.company_names[index]: int(value)
                for index, value in enumerate(initial_input.starting_prices)
            }
        if self.rule_set.investors:
            self._install_investor_offers(initial_input.investor_deals)
            self._begin_investor_selection()
        else:
            self._prepare_preset_round()

    def _install_investor_offers(self, deals: Sequence[Sequence[str]]) -> None:
        if not self.rule_set.investors:
            return
        deal_size = 4 if self.rule_set.player_count == 2 else 2
        if len(deals) != self.rule_set.player_count:
            raise ValueError("initial input must contain one Investor deal per player")
        flattened = [str(name) for deal in deals for name in deal]
        if len(flattened) != deal_size * self.rule_set.player_count:
            raise ValueError(f"each player must receive {deal_size} Investors")
        if len(set(flattened)) != len(flattened):
            raise ValueError("Investor deals must contain unique cards")
        if not set(flattened).issubset(self.rule_set.enabled_investors):
            raise ValueError("Investor deal contains a disabled Investor")
        for player, deal in enumerate(deals):
            self.players[player].investor_offer = list(deal)
            self.players[player].investor_candidates = list(deal)

    def _begin_investor_deal(self) -> None:
        deal_size = 4 if self.rule_set.player_count == 2 else 2
        self.phase = Phase.SETUP.value
        self.stage = "chance_investor"
        self.current_actor = pyspiel.PlayerId.CHANCE
        self._investor_deal_order = deque(
            player
            for _ in range(deal_size)
            for player in self._turn_order()
        )
        self._chance_kind = "investor"

    def _begin_investor_selection(self) -> None:
        self.phase = Phase.SETUP.value
        self.stage = "investor_select"
        self._chance_kind = ""
        self._investor_selection_players = deque(self._turn_order())
        self.current_actor = self._investor_selection_players[0]

    def _finish_investor_selection_player(self) -> None:
        self._investor_selection_players.popleft()
        if self._investor_selection_players:
            self.current_actor = self._investor_selection_players[0]
            self.stage = "investor_select"
            return
        self._investors_revealed = True
        cash_table = (
            INVESTOR_CASH_TWO_PLAYER
            if self.rule_set.player_count == 2
            else INVESTOR_CASH
        )
        for player in self.players:
            player.cash = sum(cash_table[name] for name in player.investors)
            player.investor_candidates.clear()
        if self._preset is not None:
            self._prepare_preset_round()
        else:
            self._prepare_chance_round()

    def _prepare_preset_round(self) -> None:
        assert self._preset is not None
        pairs = list(self._preset.round_information[self.round - 1])
        self._install_information(pairs)
        for player in self.players:
            player.viewed_information.clear()
            player.revealed_information.clear()
            player.acquired_actions.clear()
            player.meeples = [None] * self.rule_set.meeples_per_player
            player.bids = [0] * self.rule_set.meeples_per_player
        for pile in self.stockpiles:
            pile.face_up_cards.clear()
            pile.face_down_cards.clear()
            pile.bid_level = None
            pile.occupying_player = None
            pile.occupying_token = None
            pile.locked = False
            pile.purchaser = None
            card = _relocate_card(
                self._preset_deck.popleft(),
                location=f"stockpile:{pile.stockpile_id}",
                face_up=True,
            )
            pile.face_up_cards.append(card)
            self._remember_card_for_all(card)
        self._hands = [[] for _ in self.players]
        self._deal_preset_supply_batch(0)
        self._begin_supply(0)

    def _deal_preset_supply_batch(self, batch: int) -> None:
        del batch
        for player in self._turn_order():
            for _ in range(2):
                card = _relocate_card(
                    self._preset_deck.popleft(),
                    location=f"hand:{player}",
                    owner=player,
                )
                self._hands[player].append(card)
                self._remember_card(player, card)

    def _turn_order(self) -> list[int]:
        return [
            (self.first_player + offset) % self.rule_set.player_count
            for offset in range(self.rule_set.player_count)
        ]

    def _remember_card(self, player: int, card: Card) -> None:
        self.players[player].viewed_cards.add(card.card_id)
        self.players[player].known_cards[card.card_id] = card

    def _remember_card_for_all(self, card: Card) -> None:
        for player in range(self.rule_set.player_count):
            self._remember_card(player, card)

    def _update_known_card(self, card: Card) -> None:
        for player in self.players:
            if card.card_id in player.known_cards:
                player.known_cards[card.card_id] = card

    def _new_card_from_key(
        self, key: tuple[str, int | None, int | str | None]
    ) -> Card:
        kind, company, value = key
        card = Card(
            card_id=self._next_card_id,
            card_type=kind,
            company_id=company,
            value=value,
            effect=str(value) if kind == CardType.ACTION.value else None,
        )
        self._next_card_id += 1
        return card

    def current_player(self) -> int:
        if self.terminal_status:
            return pyspiel.PlayerId.TERMINAL
        if self._chance_kind:
            return pyspiel.PlayerId.CHANCE
        return int(self.current_actor)

    def is_terminal(self) -> bool:
        return self.terminal_status

    def chance_outcomes(self) -> list[tuple[int, float]]:
        outcomes = self._chance_outcomes_unchecked()
        if len(outcomes) > self.rule_set.max_chance_outcomes:
            raise RuntimeError(
                "chance branching exceeds RuleSet.max_chance_outcomes: "
                f"kind={self._chance_kind}, observed={len(outcomes)}, "
                f"declared={self.rule_set.max_chance_outcomes}"
            )
        return outcomes

    def _chance_outcomes_unchecked(self) -> list[tuple[int, float]]:
        if not self.is_chance_node():
            return []
        if self._chance_kind == "starting_share":
            probability = 1.0 / len(self._starting_remaining)
            return [(index, probability) for index in range(len(self._starting_remaining))]
        if self._chance_kind == "information_company":
            probability = 1.0 / len(self._info_remaining_companies)
            return [(index, probability) for index in range(len(self._info_remaining_companies))]
        if self._chance_kind == "information_forecast":
            counts = Counter(self._info_remaining_forecasts)
            values = sorted(counts, key=lambda value: str(value))
            total = len(self._info_remaining_forecasts)
            return [(index, counts[value] / total) for index, value in enumerate(values)]
        if self._chance_kind == "market":
            available = sorted(
                (key for key, count in self._market_counts.items() if count),
                key=lambda card_key: (
                    card_key[0],
                    -1 if card_key[1] is None else card_key[1],
                    str(card_key[2]),
                ),
            )
            total = sum(self._market_counts[key] for key in available)
            return [
                (index, self._market_counts[key] / total)
                for index, key in enumerate(available)
            ]
        if self._chance_kind == "first_player":
            probability = 1.0 / self.rule_set.player_count
            return [(player, probability) for player in range(self.rule_set.player_count)]
        if self._chance_kind == "investor":
            probability = 1.0 / len(self._investor_remaining)
            return [
                (index, probability)
                for index in range(len(self._investor_remaining))
            ]
        raise RuntimeError(f"unknown chance kind {self._chance_kind}")

    def _apply_chance(self, action: int) -> None:
        if self._chance_kind == "starting_share":
            if not 0 <= action < len(self._starting_remaining):
                raise ValueError("illegal starting-share chance action")
            company = self._starting_remaining.pop(action)
            shares_per_player = self.rule_set.starting_shares_per_player
            if shares_per_player <= 0:
                raise RuntimeError("starting-share chance stage is disabled")
            receiving_player = self._starting_player // shares_per_player
            self.players[receiving_player].regular_portfolio[company] += 1
            key = (CardType.STOCK.value, company, 1)
            if self._market_counts[key] <= 0:
                raise RuntimeError("starting share is missing from Market deck")
            self._market_counts[key] -= 1
            self._starting_player += 1
            if self._starting_player == self.rule_set.player_count * shares_per_player:
                self.stage = "first_player"
                self._chance_kind = "first_player"
            return
        if self._chance_kind == "first_player":
            if not 0 <= action < self.rule_set.player_count:
                raise ValueError("illegal first-player chance action")
            self.first_player = action
            if self.rule_set.investors:
                self._begin_investor_deal()
            else:
                self._prepare_chance_round()
            return
        if self._chance_kind == "investor":
            if not 0 <= action < len(self._investor_remaining):
                raise ValueError("illegal Investor chance action")
            if not self._investor_deal_order:
                raise RuntimeError("Investor deal has no pending recipient")
            name = self._investor_remaining.pop(action)
            player = self._investor_deal_order.popleft()
            self.players[player].investor_offer.append(name)
            self.players[player].investor_candidates.append(name)
            if not self._investor_deal_order:
                self._begin_investor_selection()
            return
        if self._chance_kind == "information_company":
            if not 0 <= action < len(self._info_remaining_companies):
                raise ValueError("illegal company chance action")
            self._pending_info_company = self._info_remaining_companies.pop(action)
            self._chance_kind = "information_forecast"
            return
        if self._chance_kind == "information_forecast":
            counts = Counter(self._info_remaining_forecasts)
            values = sorted(counts, key=lambda value: str(value))
            if not 0 <= action < len(values):
                raise ValueError("illegal forecast chance action")
            forecast = values[action]
            self._info_remaining_forecasts.remove(forecast)
            assert self._pending_info_company is not None
            self._info_pairs.append((self._pending_info_company, forecast))
            self._pending_info_company = None
            if self._info_remaining_companies:
                self._chance_kind = "information_company"
            else:
                self._install_information(self._info_pairs)
                self._chance_kind = "market"
            return
        if self._chance_kind == "market":
            available = sorted(
                (key for key, count in self._market_counts.items() if count),
                key=lambda card_key: (
                    card_key[0],
                    -1 if card_key[1] is None else card_key[1],
                    str(card_key[2]),
                ),
            )
            if not 0 <= action < len(available):
                raise ValueError("illegal Market chance action")
            key = available[action]
            self._market_counts[key] -= 1
            card = self._new_card_from_key(key)
            kind, owner, ordinal = self._market_targets.popleft()
            if kind == "seed":
                located = _relocate_card(
                    card, location=f"stockpile:{owner}", face_up=True
                )
                self.stockpiles[owner].face_up_cards.append(located)
                self._remember_card_for_all(located)
            else:
                located = _relocate_card(
                    card, location=f"hand:{owner}", owner=owner
                )
                self._hands[owner].append(located)
                self._remember_card(owner, located)
            if not self._market_targets:
                self._chance_kind = ""
                batch = self._market_resume_batch
                self._market_resume_batch = None
                if batch is None:
                    raise RuntimeError("Market draw has no Supply batch to resume")
                self._begin_supply(batch)
            return
        raise RuntimeError(f"unknown chance kind {self._chance_kind}")

    def _prepare_chance_round(self) -> None:
        self.phase = Phase.INFORMATION.value
        self.stage = "chance_information"
        self.current_actor = pyspiel.PlayerId.CHANCE
        self.public_information.clear()
        self.blind_information.clear()
        self.revealed_information.clear()
        for player in self.players:
            player.private_information.clear()
            player.viewed_information.clear()
            player.revealed_information.clear()
            player.acquired_actions.clear()
            player.phase_complete = False
            player.meeples = [None] * self.rule_set.meeples_per_player
            player.bids = [0] * self.rule_set.meeples_per_player
        for pile in self.stockpiles:
            pile.face_up_cards.clear()
            pile.face_down_cards.clear()
            pile.bid_level = None
            pile.occupying_player = None
            pile.occupying_token = None
            pile.locked = False
            pile.purchaser = None
        self._hands = [[] for _ in self.players]
        self._info_remaining_companies = list(range(self.rule_set.company_count))
        self._info_remaining_forecasts = list(self.rule_set.forecast_values)
        self._info_pairs = []
        self._pending_info_company = None
        self._market_targets = deque(
            [("seed", pile, 0) for pile in range(self.rule_set.stockpile_count)]
        )
        for player in self._turn_order():
            self._market_targets.extend(
                [("hand", player, 0), ("hand", player, 1)]
            )
        self._market_resume_batch = 0
        self._chance_kind = "information_company"

    def _install_information(self, pairs: Sequence[tuple[int, int | str]]) -> None:
        cursor = 0
        for player in self._turn_order():
            count = self.rule_set.private_pairs_per_player
            self.players[player].private_information = list(pairs[cursor : cursor + count])
            cursor += count
        remaining = list(pairs[cursor:])
        if self.rule_set.two_player_topology != "official" and remaining:
            self.public_information = [remaining.pop(0)]
        else:
            self.public_information = []
        if self.rule_set.blind_information_pairs:
            self.blind_information = remaining
        else:
            self.public_information.extend(remaining)
            self.blind_information = []

    def _begin_supply(self, batch: int = 0) -> None:
        self.phase = Phase.SUPPLY.value
        self.stage = "supply_card"
        self._current_supply_batch = batch
        self._supply_order = deque((player, batch) for player in self._turn_order())
        self._supply_choice = None
        self._supply_up_pile = None
        self._supply_both_down = False
        self._set_supply_actor()

    def _begin_next_supply_batch(self) -> None:
        batch = self._current_supply_batch + 1
        if batch >= self.rule_set.supply_batches:
            if self.rule_set.investors:
                self._begin_investor_pre_demand()
            else:
                self._begin_demand()
            return
        if self._preset is not None:
            self._deal_preset_supply_batch(batch)
            self._begin_supply(batch)
            return
        self.phase = Phase.SUPPLY.value
        self.stage = "chance_supply_deal"
        self.current_actor = pyspiel.PlayerId.CHANCE
        self._market_targets = deque()
        for player in self._turn_order():
            self._market_targets.extend(
                [("hand", player, 2 * batch), ("hand", player, 2 * batch + 1)]
            )
        self._market_resume_batch = batch
        self._chance_kind = "market"

    def _set_supply_actor(self) -> None:
        if not self._supply_order:
            self._begin_next_supply_batch()
            return
        self.current_actor = self._supply_order[0][0]
        self.stage = (
            "supply_mode"
            if "secretive_stuart" in self.players[self.current_actor].investors
            else "supply_card"
        )

    def _begin_demand(self) -> None:
        self.phase = Phase.DEMAND.value
        self.stage = "demand_pile"
        tokens: list[tuple[int, int]] = []
        if self.rule_set.meeples_per_player == 2:
            for token in range(2):
                tokens.extend((player, token) for player in self._turn_order())
        else:
            tokens.extend((player, 0) for player in self._turn_order())
        self._demand_queue = deque(tokens)
        self._demand_token = None
        self._demand_pile = None
        self._demand_pending_actor = None
        self._set_demand_actor()

    def _set_demand_actor(self) -> None:
        if not self._demand_queue:
            if all(pile.occupying_player is not None for pile in self.stockpiles):
                self._resolve_auction()
                return
            raise RuntimeError("auction queue ended before every stockpile was occupied")
        self._demand_token = self._demand_queue[0]
        assert self._demand_token is not None
        self._demand_pile = None
        self._demand_pending_actor = self._demand_token[0]
        martha = next(
            (
                player.player_id
                for player in self.players
                if "mayknow_martha" in player.investors
                and (player.player_id, "mayknow_martha", self.round)
                not in self._investor_used
                and self._martha_targets(player.player_id)
            ),
            None,
        )
        if martha is not None:
            self.current_actor = martha
            self.stage = "demand_martha_offer"
        else:
            self._resume_demand_actor()

    def _resume_demand_actor(self) -> None:
        if self._demand_pending_actor is None:
            raise RuntimeError("Demand has no pending bidder")
        self.current_actor = self._demand_pending_actor
        self.stage = "demand_pile"

    def _card_namespace(self) -> str:
        return "card_ordinal" if "card_ordinal" in self.rule_set.action_codec.ranges else "card_slot"

    def _legal_actions(self, player: int) -> list[int]:
        actions = self._legal_actions_unchecked(player)
        distinct_actions = self.rule_set.action_codec.num_distinct_actions
        invalid = [action for action in actions if not 0 <= action < distinct_actions]
        if invalid:
            raise RuntimeError(
                "legal action IDs exceed the declared OpenSpiel action catalog: "
                f"stage={self.stage}, invalid={invalid}, "
                f"num_distinct_actions={distinct_actions}"
            )
        if len(actions) > self.rule_set.max_legal_actions:
            raise RuntimeError(
                "legal branching exceeds RuleSet.max_legal_actions: "
                f"stage={self.stage}, observed={len(actions)}, "
                f"declared={self.rule_set.max_legal_actions}"
            )
        return actions

    def _legal_actions_unchecked(self, player: int) -> list[int]:
        if self.terminal_status or self._chance_kind:
            return []
        if player != self.current_actor:
            return []
        codec = self.rule_set.action_codec
        if self.stage == "investor_select":
            offer = self.players[player].investor_candidates
            return [
                codec.offset("card_ordinal") + index
                for index in range(len(offer))
            ]
        if self.stage == "supply_mode":
            return sorted([codec.offset("done"), codec.offset("use_ability")])
        if self.stage == "supply_card":
            values = codec.ids(self._card_namespace())
            return list(values[:2])
        if self.stage in {"supply_up_pile", "supply_down_pile"}:
            return list(codec.ids("pile"))
        if self.stage == "demand_pile":
            return [
                codec.offset("pile") + pile
                for pile in range(self.rule_set.stockpile_count)
                if self._legal_bid_levels(player, pile)
            ]
        if self.stage == "demand_bid":
            assert self._demand_pile is not None
            return [
                codec.offset("bid_level") + level
                for level in self._legal_bid_levels(player, self._demand_pile)
            ]
        if self.stage == "demand_martha_offer":
            return sorted([codec.offset("done"), codec.offset("use_ability")])
        if self.stage == "investor_offer":
            actions = [codec.offset("done")]
            slot_ids = codec.ids("investor_slot")
            available = self._available_pre_demand_abilities(player)
            if slot_ids:
                slot_base = codec.offset("investor_slot")
                actions.extend(slot_base + slot for slot, _ability in available)
            elif available:
                actions.append(codec.offset("use_ability"))
            return sorted(actions)
        if self.stage in {"investor_warren_pile", "investor_mark_source", "investor_mark_destination"}:
            candidates = range(self.rule_set.stockpile_count)
            if self.stage == "investor_mark_source":
                candidates = [
                    index
                    for index, pile in enumerate(self.stockpiles)
                    if pile.face_up_cards or pile.face_down_cards
                ]
            elif self.stage == "investor_mark_destination":
                candidates = [index for index in candidates if index != self._investor_source]
            return [codec.offset("pile") + index for index in candidates]
        if self.stage == "investor_mark_face":
            options: list[int] = []
            pile = self.stockpiles[self._investor_source]
            base = codec.offset("card_face")
            if pile.face_up_cards:
                options.append(base)
            if pile.face_down_cards:
                options.append(base + 1)
            return options
        if self.stage == "investor_mark_scan":
            pile = self.stockpiles[self._investor_source]
            cards = pile.face_up_cards if self._investor_face == 0 else pile.face_down_cards
            actions = [codec.offset("use_ability")]
            if self._investor_scan_index + 1 < len(cards):
                actions.append(codec.offset("done"))
            return sorted(actions)
        if self.stage == "investor_martha_target":
            targets = self._martha_targets(player)
            return [codec.offset("info_target") + index for index in range(len(targets))]
        if self.stage in {"action_cramer_before", "action_cramer_after"}:
            return sorted([codec.offset("done"), codec.offset("use_ability")])
        if self.stage == "action_cramer_direction":
            return list(codec.ids("direction"))
        if self.stage == "action_direction":
            directions: list[int] = []
            base = codec.offset("direction")
            actor = self.players[player]
            if "boom" in actor.acquired_actions:
                directions.append(base)
            if "bust" in actor.acquired_actions:
                directions.append(base + 1)
            return sorted(set(directions))
        if self.stage in {"action_company", "action_cramer_company"}:
            return list(codec.ids("company"))
        if self.stage == "deborah_company":
            return sorted([*codec.ids("company"), codec.offset("done")])
        if self.stage == "selling":
            company = self._selling_company
            regular, split = self._selling_holdings(player)
            if not self.rule_set.partial_sales:
                actions = [codec.offset("done")]
                if regular[company] or split[company]:
                    actions.append(codec.offset("sell_all"))
                return sorted(actions)
            actions = [codec.offset("done")]
            sale = codec.offset("sale_mode")
            if regular[company] > 0:
                actions.append(sale)
            if split[company] > 0:
                actions.append(sale + 1)
            if regular[company] + split[company] > 0:
                actions.append(sale + 2)
            return sorted(set(actions))
        if self.stage == "dividend_claim":
            return list(codec.ids("dividend_claim"))
        return []

    def _legal_bid_levels(self, player: int, pile_index: int) -> list[int]:
        pile = self.stockpiles[pile_index]
        assert self._demand_token is not None
        token_player, token = self._demand_token
        if token_player != player:
            return []
        # A two-player token may not share a track with its owner's other token.
        for other_token, occupied in enumerate(self.players[player].meeples):
            if other_token != token and occupied == pile_index:
                return []
        minimum = 0 if pile.bid_level is None else pile.bid_level + 1
        committed_elsewhere = sum(
            bid for index, bid in enumerate(self.players[player].bids) if index != token
        )
        legal: list[int] = []
        for level in range(minimum, len(self.rule_set.bid_values)):
            cost = self._bid_cost(player, level)
            if committed_elsewhere + cost <= self.players[player].cash:
                legal.append(level)
        return legal

    def _bid_cost(self, player: int, level: int) -> int:
        value = self.rule_set.bid_values[level]
        if "discount_donald" in self.players[player].investors and level > 0:
            return self.rule_set.bid_values[level - 1]
        return value

    def _apply_action(self, action: int) -> None:
        if self.is_chance_node():
            self._apply_chance(int(action))
            return
        legal = self._legal_actions(int(self.current_actor))
        if int(action) not in legal:
            raise ValueError(
                f"illegal action {action} for player {self.current_actor}; legal={legal}"
            )
        namespace, ordinal = self.rule_set.action_codec.decode(int(action))
        actor = int(self.current_actor)
        phase_before = self.phase
        stage_before = self.stage
        sealed_sale_action = (
            stage_before == "selling"
            and not self.rule_set.sequential_observable_selling
        )

        if self.stage == "investor_select":
            offer = self.players[actor].investor_candidates
            self.players[actor].investors.append(offer.pop(ordinal))
            keep = 2 if self.rule_set.player_count == 2 else 1
            if len(self.players[actor].investors) >= keep:
                self._finish_investor_selection_player()
        elif self.stage == "supply_mode":
            self._supply_both_down = namespace == "use_ability"
            self.stage = "supply_card"
        elif self.stage == "supply_card":
            self._supply_choice = ordinal
            self.stage = "supply_up_pile"
        elif self.stage == "supply_up_pile":
            self._supply_up_pile = ordinal
            self.stage = "supply_down_pile"
        elif self.stage == "supply_down_pile":
            self._commit_supply(ordinal)
        elif self.stage == "demand_pile":
            self._demand_pile = ordinal
            self.stage = "demand_bid"
        elif self.stage == "demand_bid":
            self._commit_bid(ordinal)
        elif self.stage == "demand_martha_offer":
            if namespace == "done":
                self._resume_demand_actor()
            else:
                self.stage = "investor_martha_target"
        elif self.stage == "investor_offer":
            if namespace == "done":
                self._skip_remaining_pre_demand_abilities(actor)
            else:
                self._start_investor_ability(actor, ordinal)
        elif self.stage == "investor_warren_pile":
            for card in self.stockpiles[ordinal].face_down_cards:
                self._remember_card(actor, card)
            self._finish_investor_ability(actor)
        elif self.stage == "investor_martha_target":
            target = self._martha_targets(actor)[ordinal]
            if target[0] == "player":
                pair = self.players[target[1]].private_information[target[2]]
            else:
                pair = self.blind_information[target[1]]
            if pair not in self.players[actor].viewed_information:
                self.players[actor].viewed_information.append(pair)
            self._investor_used.add((actor, "mayknow_martha", self.round))
            self._resume_demand_actor()
        elif self.stage == "investor_mark_source":
            self._investor_source = ordinal
            self.stage = "investor_mark_face"
        elif self.stage == "investor_mark_face":
            self._investor_face = ordinal
            self._investor_scan_index = 0
            self.stage = "investor_mark_scan"
        elif self.stage == "investor_mark_scan":
            if namespace == "done":
                self._investor_scan_index += 1
            else:
                self._investor_card_ordinal = self._investor_scan_index
                self.stage = "investor_mark_destination"
        elif self.stage == "investor_mark_destination":
            source = self.stockpiles[self._investor_source]
            cards = source.face_up_cards if self._investor_face == 0 else source.face_down_cards
            card = cards.pop(self._investor_card_ordinal)
            target = self.stockpiles[ordinal]
            target_cards = target.face_up_cards if self._investor_face == 0 else target.face_down_cards
            located = _relocate_card(
                card,
                location=f"stockpile:{ordinal}",
                face_up=self._investor_face == 0,
            )
            target_cards.append(located)
            self._update_known_card(located)
            if self._investor_face == 0:
                self._remember_card_for_all(located)
            self._finish_investor_ability(actor)
        elif self.stage in {"action_cramer_before", "action_cramer_after"}:
            if namespace == "done":
                if self.stage == "action_cramer_before":
                    self._cramer_deferred.add((actor, self.round))
                    self.stage = "action_direction"
                else:
                    self._investor_used.add((actor, "crazy_cramer", self.round))
                    self._finish_action_actor()
            else:
                self.stage = "action_cramer_direction"
        elif self.stage == "action_cramer_direction":
            self._selected_direction = "boom" if ordinal == 0 else "bust"
            self.stage = "action_cramer_company"
        elif self.stage == "action_cramer_company":
            self._commit_cramer_action(actor, ordinal)
        elif self.stage == "action_direction":
            self._selected_direction = "boom" if ordinal == 0 else "bust"
            self.stage = "action_company"
        elif self.stage == "action_company":
            self._commit_market_action(actor, ordinal)
        elif self.stage == "selling":
            if sealed_sale_action:
                self._record_private_sale_action(
                    actor,
                    int(action),
                    namespace,
                    ordinal,
                )
            self._commit_sale(actor, namespace, ordinal)
        elif self.stage == "dividend_claim":
            self._commit_dividend(actor, ordinal)
        elif self.stage == "deborah_company":
            if namespace != "done":
                self._pay_dividend(ordinal, amount_per_share=1)
            self._finish_deborah_actor()
        else:
            raise RuntimeError(f"unhandled stage {self.stage}")

        if sealed_sale_action:
            return
        self._sequence += 1
        history_record: dict[str, Any] = {
            "sequence": self._sequence,
            "player": actor,
            "phase": phase_before,
            "stage": stage_before,
            "action": int(action),
            "label": self._action_label(int(action), stage_before),
        }
        if stage_before == "supply_down_pile" and self._last_supply_public is not None:
            history_record["public_supply_commit"] = self._last_supply_public
            self._last_supply_public = None
        self.history_records.append(history_record)

    def _commit_supply(self, down_pile: int) -> None:
        player, _batch = self._supply_order.popleft()
        assert self._supply_choice is not None and self._supply_up_pile is not None
        hand = self._hands[player]
        batch_cards = hand[:2]
        if len(batch_cards) != 2:
            raise RuntimeError("Supply hand does not contain two cards")
        up_card = batch_cards[self._supply_choice]
        down_card = batch_cards[1 - self._supply_choice]
        del hand[:2]
        up_target = self.stockpiles[self._supply_up_pile]
        if self._supply_both_down:
            located_up = _relocate_card(
                up_card,
                location=f"stockpile:{self._supply_up_pile}",
                face_up=False,
                owner=None,
            )
            up_target.face_down_cards.append(located_up)
        else:
            located_up = _relocate_card(
                up_card,
                location=f"stockpile:{self._supply_up_pile}",
                face_up=True,
                owner=None,
            )
            up_target.face_up_cards.append(located_up)
            self._remember_card_for_all(located_up)
        located_down = _relocate_card(
            down_card,
            location=f"stockpile:{down_pile}",
            face_up=False,
            owner=None,
        )
        self.stockpiles[down_pile].face_down_cards.append(located_down)
        self._update_known_card(located_up)
        self._update_known_card(located_down)
        self._last_supply_public = {
            "player": player,
            "face_up_pile": self._supply_up_pile,
            "face_down_pile": down_pile,
            "face_up_card": None if self._supply_both_down else asdict(located_up),
            "both_face_down": self._supply_both_down,
        }
        self._supply_choice = None
        self._supply_up_pile = None
        self._supply_both_down = False
        self._set_supply_actor()

    def _commit_bid(self, level: int) -> None:
        assert self._demand_token is not None and self._demand_pile is not None
        player, token = self._demand_token
        pile = self.stockpiles[self._demand_pile]
        displaced = (
            (pile.occupying_player, pile.occupying_token)
            if pile.occupying_player is not None and pile.occupying_token is not None
            else None
        )
        if displaced:
            displaced_player, displaced_token = displaced
            self.players[displaced_player].meeples[displaced_token] = None
            self.players[displaced_player].bids[displaced_token] = 0
        pile.occupying_player = player
        pile.occupying_token = token
        pile.bid_level = level
        self.players[player].meeples[token] = self._demand_pile
        self.players[player].bids[token] = self._bid_cost(player, level)
        self._demand_queue.popleft()
        if displaced and displaced not in self._demand_queue:
            self._demand_queue.append(displaced)
        self._demand_token = None
        self._demand_pile = None
        self._set_demand_actor()

    def _resolve_auction(self) -> None:
        # Reserve and pay every winning bid before resolving any pile cards.
        # This matters in the official two-player game: processing a fee from a
        # player's first pile must not consume cash already committed to their
        # second winning bid.
        for pile in self.stockpiles:
            assert pile.occupying_player is not None and pile.occupying_token is not None
            player = self.players[pile.occupying_player]
            bid = player.bids[pile.occupying_token]
            player.cash -= bid
            pile.purchaser = player.player_id
            pile.locked = True
        for pile in self.stockpiles:
            assert pile.purchaser is not None
            player = self.players[pile.purchaser]
            cards = pile.face_up_cards + pile.face_down_cards
            for card in cards:
                self._remember_card(player.player_id, card)
                if card.card_type == CardType.STOCK.value:
                    assert card.company_id is not None
                    player.regular_portfolio[card.company_id] += 1
                    located = _relocate_card(
                        card,
                        location=f"portfolio:{player.player_id}",
                        face_up=False,
                        owner=player.player_id,
                    )
                    self._update_known_card(located)
                elif card.card_type == CardType.TRADING_FEE.value:
                    fee = abs(int(card.value or 0))
                    if "broker_bernie" in player.investors:
                        player.cash += fee
                    elif player.cash >= fee:
                        player.cash -= fee
                    else:
                        player.fees.append(fee)
                    located = _relocate_card(card, location="discard", face_up=True)
                    self.discards.append(located)
                    self._remember_card_for_all(located)
                elif card.card_type == CardType.ACTION.value:
                    player.acquired_actions.append(str(card.value))
                    self._remember_card(
                        player.player_id,
                        _relocate_card(
                            card,
                            location=f"actions:{player.player_id}",
                            face_up=False,
                            owner=player.player_id,
                        ),
                    )
                else:
                    located = _relocate_card(card, location="discard", face_up=True)
                    self.discards.append(located)
                    self._remember_card_for_all(located)
            pile.face_up_cards.clear()
            pile.face_down_cards.clear()
        if self.rule_set.market_action_cards or self.rule_set.investors:
            self._begin_action_phase()
        else:
            self._begin_selling()

    def _begin_action_phase(self) -> None:
        self.phase = Phase.ACTION.value
        self._action_players = deque(
            player
            for player in self._turn_order()
            if self.players[player].acquired_actions
            or "crazy_cramer" in self.players[player].investors
        )
        self._set_action_actor()

    def _set_action_actor(self) -> None:
        while self._action_players:
            player = self._action_players[0]
            actor = self.players[player]
            cramer_available = (
                "crazy_cramer" in actor.investors
                and (player, "crazy_cramer", self.round) not in self._investor_used
            )
            self.current_actor = player
            self._selected_direction = None
            if (
                cramer_available
                and actor.acquired_actions
                and (player, self.round) not in self._cramer_deferred
            ):
                self.stage = "action_cramer_before"
                return
            if actor.acquired_actions:
                self.stage = "action_direction"
                return
            if cramer_available:
                self.stage = "action_cramer_after"
                return
            self._action_players.popleft()
        self._begin_selling()

    def _commit_market_action(self, player: int, company: int) -> None:
        assert self._selected_direction is not None
        actor = self.players[player]
        direction = self._selected_direction
        if direction not in actor.acquired_actions:
            raise RuntimeError("selected Market action is not available")
        actor.acquired_actions.remove(direction)
        self.discards.append(
            Card(
                card_id=self._next_card_id,
                card_type=CardType.ACTION.value,
                value=direction,
                effect=direction,
                face_up=True,
                location="discard",
            )
        )
        self._next_card_id += 1
        steps = 2 if direction == "boom" else -2
        self._move_price(
            company,
            steps,
            cause="market_impact",
            actor_id=player,
            effect=direction,
        )
        self._selected_direction = None
        self._set_action_actor()

    def _commit_cramer_action(self, player: int, company: int) -> None:
        assert self._selected_direction is not None
        steps = 1 if self._selected_direction == "boom" else -1
        self._move_price(
            company,
            steps,
            cause="investor_action",
            actor_id=player,
            effect=self._selected_direction,
        )
        self._selected_direction = None
        self._investor_used.add((player, "crazy_cramer", self.round))
        self._set_action_actor()

    def _finish_action_actor(self) -> None:
        self._action_players.popleft()
        self._set_action_actor()

    def _begin_selling(self) -> None:
        self.phase = Phase.SELLING.value
        self.stage = "selling"
        self._selling_players = deque(self._turn_order())
        self._selling_company = 0
        if not self.rule_set.sequential_observable_selling:
            self._selling_shadow_regular = [
                list(player.regular_portfolio) for player in self.players
            ]
            self._selling_shadow_split = [
                list(player.split_portfolio) for player in self.players
            ]
        self._advance_selling_cursor()

    def _advance_selling_cursor(self) -> None:
        if not self.rule_set.sequential_observable_selling:
            while self._selling_players:
                if self._selling_company < self.rule_set.company_count:
                    self.current_actor = self._selling_players[0]
                    self.stage = "selling"
                    return
                self._selling_players.popleft()
                self._selling_company = 0
            self._settle_sealed_sales()
            self._begin_movement()
            return

        while self._selling_players:
            player = self._selling_players[0]
            while self._selling_company < self.rule_set.company_count:
                if (
                    self.players[player].regular_portfolio[self._selling_company]
                    or self.players[player].split_portfolio[self._selling_company]
                ):
                    self.current_actor = player
                    self.stage = "selling"
                    return
                self._selling_company += 1
            self._selling_players.popleft()
            self._selling_company = 0
        self._begin_movement()

    def _selling_holdings(self, player: int) -> tuple[list[int], list[int]]:
        if (
            not self.rule_set.sequential_observable_selling
            and self._selling_shadow_regular
        ):
            return (
                self._selling_shadow_regular[player],
                self._selling_shadow_split[player],
            )
        actor = self.players[player]
        return actor.regular_portfolio, actor.split_portfolio

    def _record_private_sale_action(
        self,
        player: int,
        action: int,
        namespace: str,
        ordinal: int,
    ) -> None:
        self._private_sale_history[player].append(
            {
                "private_sequence": len(self._private_sale_history[player]) + 1,
                "after_public_sequence": self._sequence,
                "round": self.round,
                "player": player,
                "phase": Phase.SELLING.value,
                "stage": "selling_commitment",
                "company": self._selling_company,
                "action": int(action),
                "action_type": namespace,
                "ordinal": ordinal,
            }
        )

    def _commit_sale(self, player: int, namespace: str, mode: int) -> None:
        actor = self.players[player]
        company = self._selling_company
        regular, split = self._selling_holdings(player)
        settle_immediately = self.rule_set.sequential_observable_selling
        if not self.rule_set.partial_sales:
            if namespace == "sell_all":
                represented = regular[company] + 2 * split[company]
                if settle_immediately:
                    self._sell_shares(actor, company, represented)
                regular[company] = 0
                split[company] = 0
            self._selling_company += 1
            self._advance_selling_cursor()
            return

        if namespace == "done":
            self._selling_company += 1
            self._advance_selling_cursor()
            return
        if mode == 0 and regular[company] > 0:
            regular[company] -= 1
            if settle_immediately:
                self._sell_shares(actor, company, 1)
        elif mode == 1 and split[company] > 0:
            split[company] -= 1
            regular[company] += 1
            if settle_immediately:
                self._sell_shares(actor, company, 1)
        elif mode == 2:
            represented = regular[company] + 2 * split[company]
            regular[company] = 0
            split[company] = 0
            if settle_immediately:
                self._sell_shares(actor, company, represented)
            self._selling_company += 1
        self._advance_selling_cursor()

    def _settle_sealed_sales(self) -> None:
        sales: dict[int, dict[int, int]] = {}
        for player_id, player in enumerate(self.players):
            shadow_regular = self._selling_shadow_regular[player_id]
            shadow_split = self._selling_shadow_split[player_id]
            player_sales: dict[int, int] = {}
            for company in range(self.rule_set.company_count):
                before = (
                    player.regular_portfolio[company]
                    + 2 * player.split_portfolio[company]
                )
                after = shadow_regular[company] + 2 * shadow_split[company]
                represented = before - after
                if represented < 0:
                    raise RuntimeError("sealed sale plan cannot create shares")
                if represented:
                    player_sales[company] = represented
                    self._sell_shares(player, company, represented)
            player.regular_portfolio = list(shadow_regular)
            player.split_portfolio = list(shadow_split)
            sales[player_id] = player_sales

        self._sequence += 1
        self.history_records.append(
            {
                "sequence": self._sequence,
                "player": -1,
                "phase": Phase.SELLING.value,
                "stage": "selling_batch",
                "action": -1,
                "label": "Reveal and settle sealed sales",
                "sales": sales,
            }
        )
        self._selling_shadow_regular = []
        self._selling_shadow_split = []

    def _sell_shares(self, player: PlayerState, company: int, represented: int) -> None:
        price = self._company_price(company)
        bonus = represented if "golden_graham" in player.investors else 0
        player.cash += represented * price + bonus
        self._pay_fee_debts(player)

    def _pay_fee_debts(self, player: PlayerState) -> None:
        while player.fees and player.cash >= player.fees[0]:
            player.cash -= player.fees.pop(0)

    def _begin_movement(self) -> None:
        self.phase = Phase.MOVEMENT.value
        self.stage = "movement"
        pairs: list[tuple[int, int | str]] = []
        for player in self._turn_order():
            pairs.extend(self.players[player].private_information[: self.rule_set.private_pairs_per_player])
        pairs.extend(self.public_information)
        pairs.extend(self.blind_information)
        self._movement_pairs = deque(pairs)
        self._continue_movement()

    def _continue_movement(self) -> None:
        while self._movement_pairs:
            company, forecast = self._movement_pairs.popleft()
            pair = (company, forecast)
            self.revealed_information.append(pair)
            for player in self.players:
                if pair in player.private_information and pair not in player.revealed_information:
                    player.revealed_information.append(pair)
            if forecast == "DIVIDEND":
                self._record_presentation_market_event(
                    event_type="market_reveal",
                    cause="market_forecast",
                    company=company,
                    prior_price=self._company_price(company),
                    requested_delta=None,
                    resulting_price=self._company_price(company),
                    forecast=forecast,
                    description=(
                        f"{self.rule_set.company_names[company]} revealed a dividend"
                    ),
                )
                holders = deque(
                    (player_id, represented)
                    for player_id in self._turn_order()
                    for represented in (
                        [1] * self.players[player_id].regular_portfolio[company]
                        + [2] * self.players[player_id].split_portfolio[company]
                    )
                )
                if self.rule_set.dividend_reveal_choice and holders:
                    self._dividend_company = company
                    self._dividend_players = holders
                    self.current_actor = holders[0][0]
                    self.stage = "dividend_claim"
                    return
                self._pay_dividend(company, amount_per_share=2)
            else:
                self._move_price(
                    company,
                    int(forecast),
                    cause="market_forecast",
                    forecast=forecast,
                )
        self._begin_deborah_or_finish_round()

    def _commit_dividend(self, player: int, ordinal: int) -> None:
        assert self._dividend_company is not None
        holding_player, represented = self._dividend_players[0]
        if holding_player != player:
            raise RuntimeError("dividend decision actor does not own the holding")
        if ordinal == 1:
            self.players[player].cash += 2 * represented
            self._pay_fee_debts(self.players[player])
            self.public_dividend_claims.append(
                {
                    "round": self.round,
                    "player": player,
                    "company": self._dividend_company,
                    "shares": represented,
                }
            )
        self._dividend_players.popleft()
        if self._dividend_players:
            self.current_actor = self._dividend_players[0][0]
            return
        self._dividend_company = None
        self.stage = "movement"
        self._continue_movement()

    def _pay_dividend(self, company: int, amount_per_share: int) -> None:
        for player in self.players:
            represented = self._represented_shares(player.player_id, company)
            player.cash += amount_per_share * represented
            self._pay_fee_debts(player)

    def _begin_deborah_or_finish_round(self) -> None:
        self._deborah_players = deque(
            player
            for player in self._turn_order()
            if "dividend_deborah" in self.players[player].investors
        )
        self._set_deborah_actor()

    def _set_deborah_actor(self) -> None:
        if self._deborah_players:
            self.current_actor = self._deborah_players[0]
            self.stage = "deborah_company"
            return
        self._finish_round()

    def _finish_deborah_actor(self) -> None:
        self._deborah_players.popleft()
        self._set_deborah_actor()

    def _finish_round(self) -> None:
        if self.round >= self.rule_set.round_count:
            self.phase = Phase.TERMINAL.value
            self.stage = "terminal"
            self.current_actor = pyspiel.PlayerId.TERMINAL
            self.terminal_status = True
            self._chance_kind = ""
            return
        self.round += 1
        self.first_player = (self.first_player + 1) % self.rule_set.player_count
        if self._preset is not None:
            self._prepare_preset_round()
        else:
            self._prepare_chance_round()

    def _company_price(self, company: int) -> int:
        return int(self.prices[self.rule_set.company_names[company]])

    def _set_company_price(self, company: int, value: int) -> None:
        self.prices[self.rule_set.company_names[company]] = int(value)

    def _represented_shares(self, player: int, company: int) -> int:
        state = self.players[player]
        return state.regular_portfolio[company] + 2 * state.split_portfolio[company]

    def _move_price(
        self,
        company: int,
        movement: int,
        *,
        cause: str = "rule",
        actor_id: int | None = None,
        forecast: int | str | None = None,
        effect: str | None = None,
    ) -> None:
        prior_price = self._company_price(company)
        if movement == 0:
            self._record_presentation_market_event(
                event_type="market_movement",
                cause=cause,
                company=company,
                prior_price=prior_price,
                requested_delta=0,
                resulting_price=prior_price,
                forecast=forecast,
                effect=effect,
                actor_id=actor_id,
            )
            return
        direction = 1 if movement > 0 else -1
        for _ in range(abs(movement)):
            if self.rule_set.advanced_price_tracks:
                track = ADVANCED_TRACKS[company]
                index = self.price_indices[company]
                next_index = index + direction
                if direction > 0 and next_index >= len(track):
                    if not self.rule_set.stock_splits:
                        self.price_indices[company] = len(track) - 1
                        self._set_company_price(company, track[-1])
                        continue
                    self._trigger_split(company)
                    self.price_indices[company] = ADVANCED_SPLIT_INDEX[company]
                elif direction < 0 and next_index < 0:
                    if not self.rule_set.bankruptcy:
                        self.price_indices[company] = 0
                        self._set_company_price(company, track[0])
                        break
                    self._trigger_bankruptcy(company)
                    self.price_indices[company] = ADVANCED_START_INDEX[company]
                    self._set_company_price(company, track[self.price_indices[company]])
                    break
                else:
                    self.price_indices[company] = next_index
                    self._set_company_price(company, track[next_index])
                    if (
                        company == 4
                        and direction > 0
                        and self.rule_set.advanced_track_dividends
                        and next_index in {1, 3, 5, 7, 9}
                    ):
                        self._pay_dividend(company, amount_per_share=1)
                continue

            price = self._company_price(company)
            ceiling = self.rule_set.standard_price_ceiling
            if direction > 0 and ceiling is not None and price >= ceiling:
                if not self.rule_set.stock_splits:
                    self._set_company_price(company, ceiling)
                    continue
                self._trigger_split(company)
                self._set_company_price(company, 6)
            elif direction < 0 and price <= 1:
                if not self.rule_set.bankruptcy:
                    self._set_company_price(company, 1)
                    break
                self._trigger_bankruptcy(company)
                self._set_company_price(company, 5)
                break
            else:
                self._set_company_price(company, price + direction)

        self._record_presentation_market_event(
            event_type="market_movement",
            cause=cause,
            company=company,
            prior_price=prior_price,
            requested_delta=movement,
            resulting_price=self._company_price(company),
            forecast=forecast,
            effect=effect,
            actor_id=actor_id,
        )

    def _record_presentation_market_event(
        self,
        *,
        event_type: str,
        cause: str,
        company: int,
        prior_price: int,
        requested_delta: int | None,
        resulting_price: int,
        forecast: int | str | None = None,
        effect: str | None = None,
        actor_id: int | None = None,
        description: str | None = None,
    ) -> None:
        """Append a public UI event without touching gameplay sequencing."""

        actual_delta = resulting_price - prior_price
        company_name = self.rule_set.company_names[company]
        if description is None:
            sign = "+" if actual_delta > 0 else ""
            description = (
                f"{company_name} moved {sign}{actual_delta} "
                f"to ${resulting_price}K"
            )
        self._presentation_sequence += 1
        self._presentation_events.append(
            PresentationEventRecord(
                presentation_sequence=self._presentation_sequence,
                round=self.round,
                event_type=event_type,
                cause=cause,
                company_id=company,
                company_name=company_name,
                prior_price=prior_price,
                requested_delta=requested_delta,
                actual_delta=actual_delta,
                resulting_price=resulting_price,
                forecast=forecast,
                effect=effect,
                actor_id=actor_id,
                description=description,
            )
        )

    def _trigger_split(self, company: int) -> None:
        for player in self.players:
            if self.rule_set.repeat_split_bonus and player.split_portfolio[company]:
                # Each split card represents two shares and pays $10K per
                # represented share on a later split.
                player.cash += 20 * player.split_portfolio[company]
                self._pay_fee_debts(player)
            player.split_portfolio[company] += player.regular_portfolio[company]
            player.regular_portfolio[company] = 0
        self._record_event("split", f"{self.rule_set.company_names[company]} split", {"company": company})

    def _trigger_bankruptcy(self, company: int) -> None:
        for player in self.players:
            player.regular_portfolio[company] = 0
            player.split_portfolio[company] = 0
        self._record_event(
            "bankruptcy",
            f"{self.rule_set.company_names[company]} went bankrupt",
            {"company": company},
        )

    def _record_event(self, event_type: str, description: str, payload: Mapping[str, Any]) -> None:
        self._sequence += 1
        record = AutomaticEventRecord(event_type, description, self._sequence, dict(payload))
        self.pending_events.append(record)

    def _begin_investor_pre_demand(self) -> None:
        self._investor_players = deque(
            player
            for player in self._turn_order()
            if set(self.players[player].investors)
            & {"maverick_mark", "wise_warren"}
        )
        self._set_investor_actor()

    def _available_pre_demand_abilities(self, player: int) -> list[tuple[int, str]]:
        return [
            (slot, investor)
            for slot, investor in enumerate(self.players[player].investors)
            if investor in {"maverick_mark", "wise_warren"}
            and (player, investor, self.round) not in self._investor_used
        ]

    def _set_investor_actor(self) -> None:
        if not self._investor_players:
            self._begin_demand()
            return
        player = self._investor_players[0]
        if not self._available_pre_demand_abilities(player):
            self._investor_players.popleft()
            self._set_investor_actor()
            return
        self.current_actor = player
        self.stage = "investor_offer"

    def _start_investor_ability(self, player: int, slot: int) -> None:
        if not 0 <= slot < len(self.players[player].investors):
            raise ValueError("Investor slot is out of range")
        investor = self.players[player].investors[slot]
        if (slot, investor) not in self._available_pre_demand_abilities(player):
            raise ValueError("Investor ability is not available")
        self._current_investor_ability = investor
        if investor == "maverick_mark":
            self.stage = "investor_mark_source"
        elif investor == "wise_warren":
            self.stage = "investor_warren_pile"
        else:
            raise RuntimeError(f"unsupported pre-Demand Investor: {investor}")

    def _finish_investor_ability(self, player: int) -> None:
        if self._current_investor_ability is None:
            raise RuntimeError("no Investor ability is active")
        self._investor_used.add((player, self._current_investor_ability, self.round))
        self._current_investor_ability = None
        self._set_investor_actor()

    def _skip_remaining_pre_demand_abilities(self, player: int) -> None:
        for _slot, investor in self._available_pre_demand_abilities(player):
            self._investor_used.add((player, investor, self.round))
        self._investor_players.popleft()
        self._set_investor_actor()

    def _martha_targets(self, player: int) -> list[tuple[str, int, int]]:
        targets: list[tuple[str, int, int]] = [
            ("player", other, pair_index)
            for other in range(self.rule_set.player_count)
            if other != player
            for pair_index in range(
                min(
                    self.rule_set.private_pairs_per_player,
                    len(self.players[other].private_information),
                )
            )
        ]
        targets.extend(
            [("blind", index, 0) for index in range(len(self.blind_information))]
        )
        return targets[:4]

    def _action_label(self, action: int, stage: str | None = None) -> str:
        namespace, ordinal = self.rule_set.action_codec.decode(action)
        if namespace == "pile":
            return f"pile {ordinal}"
        if namespace == "bid_level":
            return f"bid ${self.rule_set.bid_values[ordinal]}K"
        if namespace == "company":
            return self.rule_set.company_names[ordinal]
        if namespace == "direction":
            return "Stock Boom" if ordinal == 0 else "Stock Bust"
        if namespace == "done":
            return "Hold" if stage == "selling" else "Done"
        if namespace == "sell_all":
            return "Sell all"
        if namespace == "dividend_claim":
            return "Claim dividend" if ordinal else "Waive dividend"
        return f"{namespace} {ordinal}"

    def _action_to_string(self, player: int, action: int) -> str:
        if player == pyspiel.PlayerId.CHANCE:
            return f"Chance:{self._chance_kind}:{action}"
        return self._action_label(int(action))

    def clone(self) -> "GameState":
        # ``copy.deepcopy`` of a pybind11 State subclass can leave the C++
        # trampoline attached to the source Python object. Replaying the action
        # history is deterministic (all chance actions are explicit) and also
        # preserves OpenSpiel's serialization contract.
        cloned = GameState(self.get_game(), self._preset)
        for action in self.history():
            cloned.apply_action(action)
        return cloned

    def returns(self) -> list[float]:
        if not self.terminal_status:
            return [0.0] * self.rule_set.player_count
        return list(score_game(self.rule_set, self).utilities)

    def information_state_string(self, player: int) -> str:
        if not 0 <= player < self.rule_set.player_count:
            raise ValueError("player_id out of range")
        information, _ = observe_game_state(self.rule_set, self, player)
        return information.information_state_id

    def observation_string(self, player: int) -> str:
        information, _ = observe_game_state(self.rule_set, self, player)
        return json.dumps(
            {
                "public": information.public_state,
                "private": information.private_information,
                "owned": information.owned_stocks,
                "private_hand": [asdict(card) for card in information.private_hand],
                "known_cards": [asdict(card) for card in information.known_cards],
                "private_investor_offer": information.private_investor_offer,
                "acquired_actions": information.acquired_actions,
                "viewed_information": information.viewed_information_pairs,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def observation_tensor(self, player: int) -> np.ndarray:
        observer = StockpileObserver(
            self.get_game(),
            pyspiel.IIGObservationType(
                perfect_recall=False,
                public_info=True,
                private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER,
            ),
            None,
        )
        observer.set_from(self, player)
        return observer.tensor.copy()

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "first_player": self.first_player,
            "phase": self.phase,
            "stage": self.stage,
            "current_player": int(self.current_player()),
            "cash": [player.cash for player in self.players],
            "regular": [list(player.regular_portfolio) for player in self.players],
            "split": [list(player.split_portfolio) for player in self.players],
            "prices": dict(self.prices),
            "stockpiles": [
                {
                    "face_up": [asdict(card) for card in pile.face_up_cards],
                    "face_down_count": len(pile.face_down_cards),
                    "bid_level": pile.bid_level,
                    "bidder": pile.occupying_player,
                }
                for pile in self.stockpiles
            ],
            "public_information": list(self.public_information),
            "revealed_information": list(self.revealed_information),
            "terminal": self.terminal_status,
            "history": list(self.history_records),
        }


class StockpileObserver:
    """Fixed-size current observation for policy-based OpenSpiel algorithms."""

    TENSOR_SIZE = 256

    def __init__(
        self,
        game: StockpileGame,
        iig_obs_type: pyspiel.IIGObservationType,
        params: Mapping[str, Any] | None,
    ) -> None:
        if params:
            raise ValueError(f"observer parameters are not supported: {params}")
        self.game = game
        self.iig_obs_type = iig_obs_type
        self.tensor = np.zeros(self.TENSOR_SIZE, np.float32)
        self.dict = {
            "observation": self.tensor,
        }

    def set_from(self, state: GameState, player: int) -> None:
        if not 0 <= player < state.rule_set.player_count:
            raise ValueError("player_id out of range")
        vector = self.tensor
        vector.fill(0)
        cursor = 0

        vector[cursor + player] = 1
        cursor += 5
        profile_index = {
            "lite": 0,
            "classic": 1,
            "deluxe": 2,
        }.get(state.rule_set.profile, 3)
        vector[cursor + profile_index] = 1
        cursor += 4
        for enabled in (
            state.rule_set.trading_fees,
            state.rule_set.market_action_cards,
            state.rule_set.forecast_dividends,
            state.rule_set.stock_splits,
            state.rule_set.bankruptcy,
            state.rule_set.majority_bonus,
            state.rule_set.partial_sales,
            state.rule_set.sequential_observable_selling,
            state.rule_set.advanced_price_tracks,
            state.rule_set.investors,
        ):
            vector[cursor] = float(enabled)
            cursor += 1
        phase_index = list(Phase).index(Phase(state.phase))
        vector[cursor + phase_index] = 1
        cursor += len(Phase)
        vector[cursor] = state.round / max(1, state.rule_set.round_count)
        cursor += 1
        vector[cursor] = state.first_player / 4.0
        cursor += 1

        for index in range(5):
            vector[cursor + index] = (
                state.players[index].cash / 50.0
                if index < state.rule_set.player_count
                else 0.0
            )
        cursor += 5
        for company in range(6):
            vector[cursor + company] = (
                state._company_price(company) / 15.0
                if company < state.rule_set.company_count
                else 0.0
            )
        cursor += 6

        # Public pile summaries: total cards, face-up stock counts, bidder, bid.
        for pile_index in range(5):
            if pile_index >= len(state.stockpiles):
                cursor += 14
                continue
            pile = state.stockpiles[pile_index]
            vector[cursor] = (len(pile.face_up_cards) + len(pile.face_down_cards)) / 12.0
            cursor += 1
            for company in range(6):
                vector[cursor + company] = sum(
                    card.card_type == CardType.STOCK.value and card.company_id == company
                    for card in pile.face_up_cards
                ) / 12.0
            cursor += 6
            for bidder in range(5):
                vector[cursor + bidder] = float(pile.occupying_player == bidder)
            cursor += 5
            vector[cursor] = (
                0.0
                if pile.bid_level is None
                else (pile.bid_level + 1) / max(1, len(state.rule_set.bid_values))
            )
            cursor += 1
            vector[cursor] = len(pile.face_down_cards) / 12.0
            cursor += 1

        owner = state.players[player]
        observed_regular, observed_split = state._selling_holdings(player)
        for company in range(6):
            vector[cursor + company] = (
                observed_regular[company] / 10.0
                if company < state.rule_set.company_count
                else 0.0
            )
        cursor += 6
        for company in range(6):
            vector[cursor + company] = (
                observed_split[company] / 10.0
                if company < state.rule_set.company_count
                else 0.0
            )
        cursor += 6
        for company, forecast in owner.private_information[:6]:
            if cursor + company < len(vector):
                vector[cursor + company] = (
                    0.5 if forecast == "DIVIDEND" else (float(forecast) + 4.0) / 8.0
                )
        cursor += 6

        # The currently dealt Supply hand is private and never includes a
        # future official-two-player batch.
        for slot in range(2):
            if slot < len(state._hands[player]):
                card = state._hands[player][slot]
                kind = {
                    CardType.STOCK.value: 1.0,
                    CardType.TRADING_FEE.value: 2.0,
                    CardType.ACTION.value: 3.0,
                }.get(card.card_type, 0.0)
                vector[cursor] = kind / 3.0
                vector[cursor + 1] = (
                    0.0 if card.company_id is None else (card.company_id + 1) / 6.0
                )
                if card.card_type == CardType.TRADING_FEE.value:
                    vector[cursor + 2] = abs(float(card.value or 0)) / 3.0
                elif card.card_type == CardType.ACTION.value:
                    vector[cursor + 2] = 1.0 if card.value == "boom" else -1.0
                else:
                    vector[cursor + 2] = float(card.value or 0)
                vector[cursor + 3] = 1.0
            cursor += 4

        # Retained Investors are public after selection; during selection only
        # the observing player sees their own partial choice.
        investor_keep = 2 if state.rule_set.player_count == 2 else 1
        investors_revealed = state._investors_revealed or (
            state.rule_set.investors
            and all(
                len(investor_player.investors) >= investor_keep
                for investor_player in state.players
            )
        )
        for investor_owner in range(5):
            visible = (
                investor_owner < state.rule_set.player_count
                and (investors_revealed or investor_owner == player)
            )
            names = state.players[investor_owner].investors if visible else ()
            for name in names:
                vector[cursor + INVESTOR_NAMES.index(name)] = 1.0
            cursor += len(INVESTOR_NAMES)

        for name in owner.investor_offer:
            vector[cursor + INVESTOR_NAMES.index(name)] = 1.0
        cursor += len(INVESTOR_NAMES)

        # Aggregate only cards whose identity this player legally knows and
        # that are currently face-down in a pile. Unknown cards remain zero.
        for pile in state.stockpiles:
            for card in pile.face_down_cards:
                if card.card_id not in owner.known_cards:
                    continue
                if card.card_type == CardType.STOCK.value and card.company_id is not None:
                    card_index = card.company_id
                elif card.card_type == CardType.TRADING_FEE.value:
                    card_index = 6
                elif card.card_type == CardType.ACTION.value and card.value == "boom":
                    card_index = 7
                elif card.card_type == CardType.ACTION.value:
                    card_index = 8
                else:
                    continue
                vector[cursor + pile.stockpile_id * 9 + card_index] += 1.0 / 12.0
        cursor += 5 * 9

        vector[cursor] = owner.acquired_actions.count("boom") / 4.0
        vector[cursor + 1] = owner.acquired_actions.count("bust") / 4.0
        cursor += 2

        stage_group = self._stage_group(state.stage)
        vector[cursor] = (OBSERVATION_STAGES.index(stage_group) + 1) / len(
            OBSERVATION_STAGES
        )
        cursor += 1
        vector[cursor] = (
            0.0 if state._supply_choice is None else (state._supply_choice + 1) / 2.0
        )
        vector[cursor + 1] = (
            0.0
            if state._supply_up_pile is None
            else (state._supply_up_pile + 1) / 5.0
        )
        vector[cursor + 2] = {
            None: 0.0,
            "boom": 1.0,
            "bust": -1.0,
        }[state._selected_direction]
        vector[cursor + 3] = (
            (state._selling_company + 1) / 6.0
            if (
                state.rule_set.sequential_observable_selling
                or (
                    state.phase == Phase.SELLING.value
                    and state.current_player() == player
                )
            )
            else 0.0
        )
        vector[cursor + 4] = state._investor_source / 5.0
        vector[cursor + 5] = state._investor_face
        vector[cursor + 6] = state._investor_scan_index / 12.0

    @staticmethod
    def _stage_group(stage: str) -> str:
        if stage.startswith("chance_") or stage.startswith("starting_"):
            return "chance"
        if (
            stage.startswith("investor_mark")
            or stage.startswith("investor_warren")
            or stage.startswith("investor_martha")
        ):
            return "investor_target"
        if stage.startswith("action_cramer"):
            return "action_cramer_offer"
        if stage in OBSERVATION_STAGES:
            return stage
        if stage == "action_company":
            return "action_company"
        return "setup"

    def string_from(self, state: GameState, player: int) -> str:
        return state.observation_string(player)


def get_parameter_preset(
    name: RulesProfile | str,
    *,
    player_count: int = 2,
    round_count: int = 6,
    deluxe_investors: bool = False,
    rule_overrides: Mapping[str, Any] | None = None,
    action_space_mode: Literal["compact", "shared"] = "compact",
) -> GameParameters:
    """Return a reusable Lite, Classic, or Deluxe parameter preset."""

    raw_name = (
        name.value if isinstance(name, RulesProfile) else str(name).strip().lower()
    )
    canonical_name = PROFILE_ALIASES.get(raw_name, raw_name)
    if canonical_name not in {profile.value for profile in RulesProfile}:
        raise ValueError(f"unknown parameter preset: {raw_name}")
    if deluxe_investors and canonical_name != RulesProfile.DELUXE.value:
        raise ValueError("Investor abilities are available only in Deluxe")
    overrides = dict(rule_overrides or {})
    allowed_overrides = {
        "round_count",
        "hand",
        "fees",
        "dividend",
        "dividends",
        "impact",
        "split",
        "majority",
        "stock_tracks",
        "tracks",
        "sell_order",
        # Compatible low-level spellings for the same optional layer.
        "starting_share",
        "starting_shares_per_player",
        "trading_fees",
        "forecast_dividends",
        "dividend_reveal_choice",
        "stock_splits",
        "repeat_split_bonus",
        "majority_bonus",
        "advanced_price_tracks",
        "advanced_track_dividends",
        "sequential_observable_selling",
    }
    unsupported = sorted(set(overrides) - allowed_overrides)
    if unsupported:
        raise ValueError(
            f"{canonical_name} preset does not allow rule overrides: "
            + ", ".join(unsupported)
        )
    if canonical_name != RulesProfile.LITE.value and "impact" in overrides:
        raise ValueError("impact is configurable only in Lite")
    if canonical_name == RulesProfile.LITE.value:
        unsupported_lite_overrides = {
            "split": "stock splits",
            "stock_splits": "stock splits",
            "repeat_split_bonus": "repeat-split bonuses",
            "majority": "majority-shareholder bonuses",
            "majority_bonus": "majority-shareholder bonuses",
            "stock_tracks": "advanced price tracks",
            "tracks": "advanced price tracks",
            "advanced_price_tracks": "advanced price tracks",
            "advanced_track_dividends": "advanced-track dividends",
        }
        enabled_unsupported = sorted(
            {
                label
                for key, label in unsupported_lite_overrides.items()
                if key in overrides
                and _decode_bool_scalar(overrides[key], name=key)
            }
        )
        if enabled_unsupported:
            raise ValueError(
                "Lite does not support " + ", ".join(enabled_unsupported)
            )
    if "starting_share" in overrides:
        _decode_bool_scalar(overrides["starting_share"], name="starting_share")
    if "starting_shares_per_player" in overrides and int(
        overrides["starting_shares_per_player"]
    ) not in {0, 1}:
        raise ValueError("starting_shares_per_player must be 0 or 1")
    return GameParameters(
        player_count=player_count,
        rules_profile=canonical_name,
        round_count=round_count,
        deluxe_investors=canonical_name == RulesProfile.DELUXE.value,
        board_side="standard",
        investor_mode="none",
        rule_overrides=overrides,
        action_space_mode=action_space_mode,
    )


def _coerce_parameters(
    value: GameParameters | Mapping[str, Any] | RulesProfile | str,
) -> GameParameters:
    if isinstance(value, GameParameters):
        return value
    if isinstance(value, (RulesProfile, str)):
        return get_parameter_preset(value)
    data = dict(value)
    if "players" in data and "player_count" not in data:
        data["player_count"] = data.pop("players")
    if "rounds" in data and "round_count" not in data:
        data["round_count"] = data.pop("rounds")
    return GameParameters.model_validate(data)


def configure_game(
    game_parameters_input: GameParameters | Mapping[str, Any] | RulesProfile | str,
) -> ConfiguredGame:
    parameters = _coerce_parameters(game_parameters_input)
    rule_set = _normalise_rule_set(parameters)
    game = StockpileGame(parameters=parameters, rule_set=rule_set)
    parameter_schema = GameParameters.model_json_schema()
    state_schema = {
        "type": "object",
        "required": [
            "round",
            "first_player",
            "phase",
            "players",
            "prices",
            "stockpiles",
            "terminal_status",
        ],
        "description": "Debug snapshot schema; native OpenSpiel serialization replays action history.",
    }
    return ConfiguredGame(parameters, rule_set, game, parameter_schema, state_schema)


def _validation_success(**normalized_fields: Any) -> ValidationReport:
    return ValidationReport(
        valid=True,
        legal=True,
        reachable=True,
        normalized_fields=normalized_fields,
    )


def _validation_failure(*errors: str) -> ValidationReport:
    return ValidationReport(
        valid=False,
        legal=False,
        reachable=False,
        errors=list(errors),
    )


def _validate_state(rule_set: RuleSet, state: GameState) -> ValidationReport:
    errors: list[str] = []
    if state.rule_set != rule_set:
        errors.append("state RuleSet does not match the supplied RuleSet")
    if len(state.players) != rule_set.player_count:
        errors.append("player count mismatch")
    if len(state.prices) != rule_set.company_count:
        errors.append("company price count mismatch")
    if any(player.cash < 0 for player in state.players):
        errors.append("cash cannot be negative")
    if any(value < 0 for player in state.players for value in player.regular_portfolio):
        errors.append("regular holdings cannot be negative")
    if any(value < 0 for player in state.players for value in player.split_portfolio):
        errors.append("split holdings cannot be negative")
    return _validation_failure(*errors) if errors else _validation_success()


def initialize_game(
    rule_set: RuleSet,
    game_definition: StockpileGame,
    initial_input: InitialInput,
) -> tuple[GameState, ValidationReport]:
    if game_definition.rule_set != rule_set:
        state = GameState(game_definition, initial_input)
        return state, _validation_failure("game definition and RuleSet do not match")
    try:
        state = GameState(game_definition, initial_input)
    except (IndexError, KeyError, RuntimeError, ValueError) as initialization_error:
        state = GameState(game_definition)
        return state, _validation_failure(str(initialization_error))
    return state, _validate_state(rule_set, state)


def load_game_state(
    rule_set: RuleSet,
    game_definition: StockpileGame,
    game_state_input: str | Mapping[str, Any],
) -> tuple[GameState, ValidationReport]:
    """Loads an OpenSpiel action-history serialization.

    Mapping snapshots are intentionally validation/debug artifacts rather than
    authoritative persistence, because they may contain masked private data.
    """

    if isinstance(game_state_input, str):
        try:
            state = game_definition.deserialize_state(game_state_input)
        except Exception as deserialization_error:  # Backend-specific exception types.
            return game_definition.new_initial_state(), _validation_failure(
                str(deserialization_error)
            )
        return state, _validate_state(rule_set, state)
    return game_definition.new_initial_state(), _validation_failure(
        "mapping snapshots are observation artifacts; load a serialized action history"
    )


def _public_state(state: GameState) -> dict[str, Any]:
    investor_keep = 2 if state.rule_set.player_count == 2 else 1
    investors_revealed = state._investors_revealed or (
        state.rule_set.investors
        and all(len(player.investors) >= investor_keep for player in state.players)
    )
    return {
        "round": state.round,
        "round_count": state.rule_set.round_count,
        "phase": state.phase,
        "stage": state.stage,
        "first_player": state.first_player,
        "current_player": int(state.current_player()),
        "prices": dict(state.prices),
        "cash": {player.player_id: player.cash for player in state.players},
        "fee_debts": {
            player.player_id: tuple(player.fees) for player in state.players
        },
        "investors": {
            player.player_id: (
                tuple(player.investors) if investors_revealed else ()
            )
            for player in state.players
        },
        "dividend_claims": tuple(dict(claim) for claim in state.public_dividend_claims),
        "stockpiles": [
            {
                "stockpile_id": pile.stockpile_id,
                "face_up_cards": [asdict(card) for card in pile.face_up_cards],
                "face_down_count": len(pile.face_down_cards),
                "bid_level": pile.bid_level,
                "bid_value": (
                    None
                    if pile.bid_level is None
                    else state.rule_set.bid_values[pile.bid_level]
                ),
                "occupying_player": pile.occupying_player,
                "locked": pile.locked,
                "purchaser": pile.purchaser,
            }
            for pile in state.stockpiles
        ],
        "public_information": list(state.public_information),
        "revealed_information": list(state.revealed_information),
        "terminal": state.terminal_status,
    }


def _observable_history(state: GameState, player_id: int | None) -> tuple[Mapping[str, Any], ...]:
    visible: list[Mapping[str, Any]] = []
    for record in state.history_records:
        if record["stage"] == "investor_select" and player_id != record["player"]:
            continue
        private_supply_step = record["phase"] == Phase.SUPPLY.value
        if private_supply_step and player_id != record["player"]:
            if record["stage"] != "supply_down_pile":
                continue
            record = {
                "sequence": record["sequence"],
                "player": record["player"],
                "phase": record["phase"],
                "stage": "supply_commit",
                "public_supply_commit": record.get("public_supply_commit", {}),
            }
        visible.append(dict(record))
    if player_id is not None and not state.rule_set.sequential_observable_selling:
        # A commitment is anchored after the last public record that preceded
        # it. Interleaving on that anchor preserves the owner's chronological
        # perfect-recall history without creating gaps in anyone's public
        # sequence or exposing an opponent's number of micro-actions.
        merged: list[tuple[int, int, int, Mapping[str, Any]]] = [
            (int(record.get("sequence", 0)), 0, index, record)
            for index, record in enumerate(visible)
        ]
        merged.extend(
            (
                int(record.get("after_public_sequence", 0)),
                1,
                int(record.get("private_sequence", index)),
                dict(record),
            )
            for index, record in enumerate(state._private_sale_history[player_id])
        )
        merged.sort(key=lambda item: item[:3])
        visible = [record for _anchor, _kind, _order, record in merged]
    return tuple(visible)


def _legal_action_objects(state: GameState, player_id: int) -> list[LegalAction]:
    actions: list[LegalAction] = []
    for action_id in state.legal_actions(player_id):
        namespace, ordinal = state.rule_set.action_codec.decode(int(action_id))
        target_ids: tuple[int, ...] = ()
        amount: int | None = None
        if namespace in {"pile", "company", "info_target"}:
            target_ids = (ordinal,)
        if namespace == "bid_level":
            amount = state.rule_set.bid_values[ordinal]
        actions.append(
            LegalAction(
                action_id=int(action_id),
                phase=state.phase,
                actor_ids=(player_id,),
                action_type=namespace,
                target_ids=target_ids,
                amount=amount,
                payload={"ordinal": ordinal, "stage": state.stage},
                display_label=state._action_label(int(action_id)),
            )
        )
    return actions


def get_presentation_state(
    rule_set: RuleSet,
    game_state: GameState,
    viewer_id: int | None = None,
) -> PresentationState:
    """Return browser-safe staged context without expanding observations.

    The accessor intentionally lives beside, rather than inside,
    :func:`observe_game_state`. In particular, its fields never participate in
    an information-state identity or tensor. Supply and partially selected
    Demand/Action decisions are visible only to the acting viewer. Default
    sealed selling additionally redacts the actor and company from every other
    viewer and from spectators.
    """

    if game_state.rule_set != rule_set:
        raise ValueError("game state and RuleSet do not match")
    if viewer_id is not None and not 0 <= viewer_id < rule_set.player_count:
        raise ValueError("viewer_id out of range")

    current = int(game_state.current_player())
    actor = current if current >= 0 else None
    viewer_is_actor = viewer_id is not None and actor == viewer_id
    sealed_selling = (
        game_state.phase == Phase.SELLING.value
        and not rule_set.sequential_observable_selling
    )
    stage = game_state.stage
    if sealed_selling and not viewer_is_actor:
        actor = None
        stage = "private_selling"

    markers = tuple(
        BidMarkerPresentation(
            stockpile_id=pile.stockpile_id,
            player_id=int(pile.occupying_player),
            marker_index=int(pile.occupying_token),
            bid_value=rule_set.bid_values[int(pile.bid_level)],
            status="locked" if pile.locked else "leading",
        )
        for pile in game_state.stockpiles
        if pile.occupying_player is not None
        and pile.occupying_token is not None
        and pile.bid_level is not None
    )

    demand_token = (
        tuple(game_state._demand_token)
        if game_state.phase == Phase.DEMAND.value
        and game_state._demand_token is not None
        else None
    )
    demand_pile = (
        game_state._demand_pile
        if game_state.phase == Phase.DEMAND.value and viewer_is_actor
        else None
    )
    supply_choice = (
        game_state._supply_choice
        if game_state.phase == Phase.SUPPLY.value and viewer_is_actor
        else None
    )
    supply_up_pile = (
        game_state._supply_up_pile
        if game_state.phase == Phase.SUPPLY.value and viewer_is_actor
        else None
    )
    selected_direction = (
        game_state._selected_direction
        if game_state.phase == Phase.ACTION.value and viewer_is_actor
        else None
    )
    selling_company = None
    if game_state.phase == Phase.SELLING.value:
        if rule_set.sequential_observable_selling or viewer_is_actor:
            selling_company = game_state._selling_company

    return PresentationState(
        phase=game_state.phase,
        stage=stage,
        current_actor=actor,
        demand_token=cast(tuple[int, int] | None, demand_token),
        demand_pile=demand_pile,
        supply_choice=supply_choice,
        supply_up_pile=supply_up_pile,
        selected_direction=selected_direction,
        selling_company=selling_company,
        stockpile_markers=markers,
    )


def get_presentation_events(
    rule_set: RuleSet,
    game_state: GameState,
    *,
    since_sequence: int = 0,
) -> tuple[PresentationEventRecord, ...]:
    """Return replay-stable public events newer than ``since_sequence``."""

    if game_state.rule_set != rule_set:
        raise ValueError("game state and RuleSet do not match")
    if isinstance(since_sequence, bool) or not isinstance(since_sequence, int):
        raise TypeError("since_sequence must be an integer")
    if since_sequence < 0:
        raise ValueError("since_sequence must be non-negative")
    return tuple(
        event
        for event in game_state._presentation_events
        if event.presentation_sequence > since_sequence
    )


def preview_sale_action(
    rule_set: RuleSet,
    game_state: GameState,
    player_id: int,
    action_id: int,
) -> SalePreview:
    """Preview one currently legal sale without mutating authoritative state."""

    if game_state.rule_set != rule_set:
        raise ValueError("game state and RuleSet do not match")
    if not 0 <= player_id < rule_set.player_count:
        raise ValueError("player_id out of range")
    if isinstance(action_id, bool) or not isinstance(action_id, int):
        raise TypeError("action_id must be an integer")
    if (
        game_state.phase != Phase.SELLING.value
        or game_state.stage != "selling"
        or game_state.current_player() != player_id
    ):
        raise ValueError("sale preview is available only to the current seller")
    if action_id not in game_state._legal_actions(player_id):
        raise ValueError("sale action is not currently legal")

    namespace, ordinal = rule_set.action_codec.decode(action_id)
    if namespace not in {"done", "sell_all", "sale_mode"}:
        raise ValueError("action is not a sale decision")
    company = game_state._selling_company
    observed_regular, observed_split = game_state._selling_holdings(player_id)
    regular = int(observed_regular[company])
    split = int(observed_split[company])
    quantity = 0
    if namespace == "sell_all":
        quantity = regular + 2 * split
        regular = 0
        split = 0
    elif namespace == "sale_mode":
        if ordinal == 0:
            quantity = 1
            regular -= 1
        elif ordinal == 1:
            quantity = 1
            split -= 1
            regular += 1
        elif ordinal == 2:
            quantity = regular + 2 * split
            regular = 0
            split = 0
        else:  # Defensive against a future wider sale namespace.
            raise ValueError("unsupported sale mode")

    unit_price = game_state._company_price(company)
    per_share_bonus = (
        1 if "golden_graham" in game_state.players[player_id].investors else 0
    )
    label = {
        "done": "Hold",
        "sell_all": "Sell all",
    }.get(namespace)
    if namespace == "sale_mode":
        label = {
            0: "Sell one share",
            1: "Sell one split share",
            2: "Sell all",
        }[ordinal]
    assert label is not None
    return SalePreview(
        action_id=action_id,
        action_type=namespace,
        label=label,
        company_id=company,
        company_name=rule_set.company_names[company],
        quantity_sold=quantity,
        unit_price=unit_price,
        gross_value=quantity * (unit_price + per_share_bonus),
        resulting_regular=regular,
        resulting_split=split,
        resulting_represented=regular + 2 * split,
    )


def observe_game_state(
    rule_set: RuleSet,
    game_state: GameState,
    player_id: int | None = None,
) -> tuple[InformationState, list[LegalAction]]:
    if game_state.rule_set != rule_set:
        raise ValueError("game state and RuleSet do not match")
    if player_id is not None and not 0 <= player_id < rule_set.player_count:
        raise ValueError("player_id out of range")
    public = _public_state(game_state)
    if player_id is None:
        private: tuple[tuple[int, int | str], ...] = ()
        owned: dict[str, Any] = {}
        viewed: tuple[int, ...] = ()
        legal: list[LegalAction] = []
        tensor: tuple[float, ...] = ()
        private_hand: tuple[Card, ...] = ()
        known_cards: tuple[Card, ...] = ()
        investor_offer: tuple[str, ...] = ()
        acquired_actions: tuple[str, ...] = ()
        viewed_information: tuple[tuple[int, int | str], ...] = ()
    else:
        player = game_state.players[player_id]
        owned_regular, owned_split = game_state._selling_holdings(player_id)
        private = tuple(player.private_information)
        owned = {
            "regular": dict(zip(rule_set.company_names, owned_regular, strict=True)),
            "split": dict(zip(rule_set.company_names, owned_split, strict=True)),
            "fee_debts": tuple(player.fees),
            "investors": tuple(player.investors),
        }
        if (
            not rule_set.sequential_observable_selling
            and game_state.phase == Phase.SELLING.value
            and game_state.current_player() == player_id
        ):
            owned["selling_company"] = game_state._selling_company
        viewed = tuple(sorted(player.viewed_cards))
        legal = (
            _legal_action_objects(game_state, player_id)
            if game_state.current_player() == player_id
            else []
        )
        tensor = tuple(float(value) for value in game_state.observation_tensor(player_id))
        private_hand = tuple(game_state._hands[player_id])
        known_cards = tuple(
            sorted(player.known_cards.values(), key=lambda card: card.card_id)
        )
        investor_offer = tuple(player.investor_offer)
        acquired_actions = tuple(player.acquired_actions)
        viewed_information = tuple(player.viewed_information)
    public_investors = {
        int(owner): tuple(investors)
        for owner, investors in public["investors"].items()
    }
    observable_history = _observable_history(game_state, player_id)
    identity_payload = {
        "player": player_id,
        "public": public,
        "owned": owned,
        "private": private,
        "viewed": viewed,
        "private_hand": [asdict(card) for card in private_hand],
        "known_cards": [asdict(card) for card in known_cards],
        "investor_offer": investor_offer,
        "acquired_actions": acquired_actions,
        "viewed_information": viewed_information,
        "history": observable_history,
    }
    information_id = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"))
    information = InformationState(
        player_id=player_id,
        public_state=public,
        owned_stocks=owned,
        private_information=private,
        legally_viewed_cards=viewed,
        public_cash_and_bids={
            "cash": public["cash"],
            "bids": [
                (pile["occupying_player"], pile["bid_value"])
                for pile in public["stockpiles"]
            ],
        },
        observable_history=observable_history,
        information_state_id=information_id,
        tensor=tensor,
        legal_action_ids=tuple(action.action_id for action in legal),
        private_hand=private_hand,
        known_cards=known_cards,
        private_investor_offer=investor_offer,
        public_investors=public_investors,
        acquired_actions=acquired_actions,
        viewed_information_pairs=viewed_information,
    )
    return information, legal


def advance_game(
    rule_set: RuleSet,
    game_state: GameState,
    legal_actions: Sequence[LegalAction] | None,
    action_request: ActionRequest | Mapping[str, Any],
) -> tuple[GameState, ActionRecord | None, ValidationReport]:
    request = (
        action_request
        if isinstance(action_request, ActionRequest)
        else ActionRequest(**dict(action_request))
    )
    clone = game_state.clone()
    if game_state.is_terminal():
        return clone, None, _validation_failure("cannot advance a terminal state")
    if game_state.is_chance_node():
        return clone, None, _validation_failure(
            "advance_game accepts player choices; apply a chance outcome through OpenSpiel"
        )
    if request.player_id != game_state.current_player():
        return clone, None, _validation_failure("action actor is not the current player")
    recomputed = _legal_action_objects(game_state, request.player_id)
    legal_ids = {action.action_id for action in recomputed}
    supplied_ids = {action.action_id for action in legal_actions or recomputed}
    if request.action_id not in legal_ids or request.action_id not in supplied_ids:
        return clone, None, _validation_failure("unknown, stale, or illegal action")
    phase = game_state.phase
    action_type, _ = rule_set.action_codec.decode(request.action_id)
    label = game_state._action_label(request.action_id)
    clone.apply_action(request.action_id)
    record = ActionRecord(
        player_id=request.player_id,
        action_id=request.action_id,
        phase=phase,
        action_type=action_type,
        description=label,
        sequence=clone._sequence,
    )
    return clone, record, _validate_state(rule_set, clone)


def resolve_automatic_events(
    rule_set: RuleSet,
    game_state: GameState,
    action_record: ActionRecord | None = None,
) -> tuple[GameState, list[AutomaticEventRecord]]:
    del action_record
    if game_state.rule_set != rule_set:
        raise ValueError("game state and RuleSet do not match")
    clone = game_state.clone()
    events = list(clone.pending_events)
    clone.pending_events.clear()
    return clone, events


def _training_utilities(cash: Sequence[int], winners: Sequence[int]) -> tuple[float, ...]:
    players = len(cash)
    winner_set = set(winners)
    if len(winner_set) == players:
        return tuple(0.0 for _ in cash)
    winner_utility = 1.0 / len(winner_set)
    loser_utility = -1.0 / (players - len(winner_set))
    return tuple(
        winner_utility if player in winner_set else loser_utility
        for player in range(players)
    )


def _cash_after_fifo_fee_debts(cash: int, debts: Sequence[int]) -> int:
    """Settle affordable retained fees in order without mutating a player."""

    remaining = int(cash)
    for debt in debts:
        amount = int(debt)
        if remaining < amount:
            break
        remaining -= amount
    return remaining


def score_game(rule_set: RuleSet, game_state: GameState) -> GameResult:
    majority_holders: dict[int, tuple[int, ...]] = {}
    bonuses = {player: 0 for player in range(rule_set.player_count)}
    for company in range(rule_set.company_count):
        shares = [
            game_state._represented_shares(player, company)
            for player in range(rule_set.player_count)
        ]
        maximum = max(shares, default=0)
        holders = tuple(player for player, amount in enumerate(shares) if amount == maximum and amount > 0)
        majority_holders[company] = holders
        if rule_set.majority_bonus and holders:
            award = 10 if len(holders) == 1 else 5
            for player in holders:
                bonuses[player] += award

    liquidation: dict[int, int] = {}
    final_cash: dict[int, int] = {}
    for player in range(rule_set.player_count):
        value = sum(
            game_state._represented_shares(player, company)
            * game_state._company_price(company)
            for company in range(rule_set.company_count)
        )
        if "golden_graham" in game_state.players[player].investors:
            value += sum(
                game_state._represented_shares(player, company)
                for company in range(rule_set.company_count)
            )
        liquidation[player] = value
        gross_cash = game_state.players[player].cash + value + bonuses[player]
        final_cash[player] = _cash_after_fifo_fee_debts(
            gross_cash,
            game_state.players[player].fees,
        )
    high = max(final_cash.values())
    winners = tuple(player for player, value in final_cash.items() if value == high)
    return GameResult(
        final_cash_by_player=final_cash,
        majority_shareholders=majority_holders,
        bonuses=bonuses,
        liquidation_values=liquidation,
        tie_break="shared_cash_tie" if len(winners) > 1 else None,
        winner_ids=winners,
        utilities=_training_utilities(tuple(final_cash.values()), winners),
    )


def terminal_liquidation_details(
    rule_set: RuleSet,
    game_state: GameState,
) -> tuple[TerminalPlayerLiquidation, ...]:
    """Return presentation-ready terminal liquidation and tied rankings."""

    if game_state.rule_set != rule_set:
        raise ValueError("game state and RuleSet do not match")
    if not game_state.is_terminal():
        raise ValueError("terminal liquidation is available only at game end")
    result = score_game(rule_set, game_state)
    final_cash_values = tuple(result.final_cash_by_player.values())
    winner_ids = set(result.winner_ids)
    rows: list[TerminalPlayerLiquidation] = []
    for player in range(rule_set.player_count):
        companies = tuple(
            TerminalCompanyLiquidation(
                company_id=company,
                company_name=rule_set.company_names[company],
                represented_shares=game_state._represented_shares(player, company),
                unit_price=game_state._company_price(company),
                value=(
                    game_state._represented_shares(player, company)
                    * game_state._company_price(company)
                ),
            )
            for company in range(rule_set.company_count)
        )
        cash = int(result.final_cash_by_player[player])
        rows.append(
            TerminalPlayerLiquidation(
                player_id=player,
                companies=companies,
                liquidation_value=int(result.liquidation_values[player]),
                final_cash=cash,
                rank=1 + sum(other > cash for other in final_cash_values),
                winner=player in winner_ids,
            )
        )
    return tuple(rows)


def run_game(
    rule_set: RuleSet,
    game_state: GameState,
    action_requests: Iterable[ActionRequest | Mapping[str, Any]],
    *,
    capture_snapshots: bool = False,
) -> tuple[Playthrough, GameState, GameResult | None]:
    state = game_state.clone()
    initial = str(state)
    snapshots: list[str] = [initial] if capture_snapshots else []
    records: list[ActionRecord] = []
    events: list[AutomaticEventRecord] = []
    iterator = iter(action_requests)
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            if not outcomes:
                break
            state.apply_action(max(outcomes, key=lambda item: item[1])[0])
            continue
        try:
            request = next(iterator)
        except StopIteration:
            break
        actor = state.current_player()
        _, legal = observe_game_state(rule_set, state, actor)
        state, record, report = advance_game(rule_set, state, legal, request)
        if not report.valid:
            playthrough = Playthrough(
                initial,
                snapshots,
                records,
                events,
                None,
                None,
                state._preset.random_seed if state._preset else None,
                "invalid",
            )
            return playthrough, state, None
        if record:
            records.append(record)
        state, new_events = resolve_automatic_events(rule_set, state, record)
        events.extend(new_events)
        if capture_snapshots:
            snapshots.append(str(state))
    result = score_game(rule_set, state) if state.is_terminal() else None
    playthrough = Playthrough(
        initial_state=initial,
        state_snapshots=snapshots,
        action_records=records,
        automatic_event_records=events,
        terminal_state=str(state) if state.is_terminal() else None,
        game_result=result,
        random_seed=state._preset.random_seed if state._preset else None,
        completion_status="complete" if state.is_terminal() else "awaiting_action",
    )
    return playthrough, state, result


def compute_information_set_complexity(
    parameters: (
        GameParameters
        | Mapping[str, Any]
        | RulesProfile
        | str
        | ConfiguredGame
        | StockpileGame
    ),
    *,
    max_states: int = 1_000_000,
    max_seconds: float = 60.0,
    require_exact: bool = False,
) -> InformationSetComplexity:
    """Counts reachable decision information sets and information-set actions.

    An information set is keyed by ``(player_id, information_state_string)``.
    ``information_set_actions`` is the sum of the legal-action count over each
    unique information set, so it is the number of distinct ``(I, a)`` pairs.

    Full core and full-profile trees are generally too large to enumerate. If
    either traversal budget is reached, the returned counts are deterministic
    observed lower bounds and ``exact`` is false. Set ``require_exact`` to turn
    that condition into :class:`InformationSetEnumerationLimit`; the exception
    retains the partial report in its ``result`` attribute.
    """

    if isinstance(max_states, bool) or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be a positive finite number")

    if isinstance(parameters, ConfiguredGame):
        configured = parameters
    elif isinstance(parameters, StockpileGame):
        configured = ConfiguredGame(
            parameters=parameters.parameters_model,
            rule_set=parameters.rule_set,
            game=parameters,
            parameter_schema=GameParameters.model_json_schema(),
            state_schema={},
        )
    else:
        configured = configure_game(parameters)

    start = time.perf_counter()
    deadline = start + max_seconds
    stack: list[GameState] = [configured.game.new_initial_state()]
    information_actions: dict[tuple[int, str], tuple[int, ...]] = {}
    player_count = configured.parameters.player_count
    per_player_sets = {player: 0 for player in range(player_count)}
    per_player_actions = {player: 0 for player in range(player_count)}
    states_visited = 0
    terminal_states = 0
    chance_nodes = 0
    truncation_reason: str | None = None

    while stack:
        if states_visited >= max_states:
            truncation_reason = "max_states"
            break
        if time.perf_counter() >= deadline:
            truncation_reason = "max_seconds"
            break

        state = stack.pop()
        states_visited += 1
        if state.is_terminal():
            terminal_states += 1
            continue
        if state.is_chance_node():
            chance_nodes += 1
            actions = [
                int(action)
                for action, probability in state.chance_outcomes()
                if probability > 0
            ]
        else:
            player = int(state.current_player())
            actions = sorted(int(action) for action in state.legal_actions(player))
            if not actions:
                raise ValueError(
                    "reachable non-terminal player node has no legal actions: "
                    f"phase={state.phase}, stage={state.stage}, player={player}"
                )
            information_key = (player, state.information_state_string(player))
            action_signature = tuple(actions)
            previous = information_actions.get(information_key)
            if previous is None:
                information_actions[information_key] = action_signature
                per_player_sets[player] += 1
                per_player_actions[player] += len(action_signature)
            elif previous != action_signature:
                raise ValueError(
                    "states in one information set expose different legal actions: "
                    f"player={player}, first={previous}, current={action_signature}"
                )

        # Reverse insertion preserves ascending action traversal under LIFO.
        for action in reversed(actions):
            child = state.clone()
            child.apply_action(action)
            stack.append(child)

    elapsed = time.perf_counter() - start
    exact = not stack and truncation_reason is None
    result = InformationSetComplexity(
        parameters=configured.parameters,
        exact=exact,
        count_kind="exact" if exact else "lower_bound",
        information_sets=len(information_actions),
        information_set_actions=sum(
            len(actions) for actions in information_actions.values()
        ),
        max_actions_per_information_set=max(
            (len(actions) for actions in information_actions.values()), default=0
        ),
        per_player_information_sets=per_player_sets,
        per_player_information_set_actions=per_player_actions,
        states_visited=states_visited,
        terminal_states=terminal_states,
        chance_nodes=chance_nodes,
        elapsed_seconds=elapsed,
        max_states=max_states,
        max_seconds=max_seconds,
        truncation_reason=truncation_reason,
    )
    if require_exact and not exact:
        raise InformationSetEnumerationLimit(result)
    return result


def complexity_report(
    configuration: (
        GameParameters
        | Mapping[str, Any]
        | RulesProfile
        | str
        | ConfiguredGame
        | StockpileGame
        | RuleSet
    ),
    *,
    benchmark_seconds: float = 0.0,
    random_seed: int = 0,
) -> dict[str, Any]:
    if isinstance(configuration, RuleSet):
        rule_set = configuration
        game = StockpileGame(
            parameters=GameParameters(
                player_count=rule_set.player_count,
                rules_profile=rule_set.profile,
                round_count=rule_set.round_count,
                deluxe_investors=rule_set.investors,
                board_side=(
                    "advanced" if rule_set.advanced_price_tracks else "standard"
                ),
                action_space_mode=rule_set.action_space_mode,
            ),
            rule_set=rule_set,
        )
    elif isinstance(configuration, StockpileGame):
        game = configuration
        rule_set = game.rule_set
    elif isinstance(configuration, ConfiguredGame):
        game = configuration.game
        rule_set = configuration.rule_set
    else:
        configured = configure_game(configuration)
        game = configured.game
        rule_set = configured.rule_set
    report: dict[str, Any] = {
        "num_distinct_actions": rule_set.action_codec.num_distinct_actions,
        "max_legal_actions": rule_set.max_legal_actions,
        "max_chance_outcomes": rule_set.max_chance_outcomes,
        "shared_action_head": rule_set.action_codec.shared_action_head,
        "max_game_length": rule_set.max_game_length,
        "observation_size": StockpileObserver.TENSOR_SIZE,
    }
    if benchmark_seconds > 0:
        rng = random.Random(random_seed)
        deadline = time.perf_counter() + benchmark_seconds
        episodes = decisions = 0
        while time.perf_counter() < deadline:
            state = game.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    actions, probabilities = zip(*outcomes, strict=True)
                    state.apply_action(rng.choices(actions, probabilities, k=1)[0])
                else:
                    legal = state.legal_actions()
                    state.apply_action(rng.choice(legal))
                    decisions += 1
            episodes += 1
        elapsed = max(1e-9, benchmark_seconds)
        report.update(
            {
                "benchmark_seconds": benchmark_seconds,
                "episodes_per_second": episodes / elapsed,
                "decisions_per_second": decisions / elapsed,
                "projected_72h_episodes": int(episodes / elapsed * 72 * 60 * 60),
            }
        )
    return report


try:
    pyspiel.register_game(_GAME_TYPE, StockpileGame)
except Exception as registration_error:  # Safe for a second module-name import.
    if (
        "already" not in str(registration_error).lower()
        and "registered" not in str(registration_error).lower()
    ):
        raise


__all__ = [
    "ActionCodec",
    "ActionRecord",
    "ActionRequest",
    "AutomaticEventRecord",
    "BidMarkerPresentation",
    "Card",
    "ConfiguredGame",
    "GameParameters",
    "GameResult",
    "GameState",
    "InformationState",
    "InformationSetComplexity",
    "InformationSetEnumerationLimit",
    "InitialInput",
    "LegalAction",
    "LiteOptionalRule",
    "Phase",
    "PlayerState",
    "Playthrough",
    "PresentationEventRecord",
    "PresentationState",
    "RuleSet",
    "RulesProfile",
    "SalePreview",
    "Stockpile",
    "StockpileGame",
    "TerminalCompanyLiquidation",
    "TerminalPlayerLiquidation",
    "ValidationReport",
    "advance_game",
    "complexity_report",
    "compute_information_set_complexity",
    "configure_game",
    "get_parameter_preset",
    "get_presentation_events",
    "get_presentation_state",
    "initialize_game",
    "load_game_state",
    "observe_game_state",
    "randomize_initial_input",
    "preview_sale_action",
    "resolve_automatic_events",
    "run_game",
    "score_game",
    "terminal_liquidation_details",
]
