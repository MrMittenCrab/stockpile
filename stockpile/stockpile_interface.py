"""UI-agnostic facade for configuring and analysing Stockpile games.

The platform module owns all game rules, validation, OpenSpiel integration,
and reachable information-set enumeration.  This module composes those APIs
into values that a terminal, GUI, notebook, or service can present without
performing any input or output itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Literal

from . import stockpile_platform as platform

if TYPE_CHECKING:
    from .complexity_cache import (
        ComplexityBounds,
        ComplexityCache,
        ComplexityProvenance,
    )


ActionSpaceMode = Literal["compact", "shared"]
CachePolicy = Literal["prefer", "refresh", "off"]
LiteOptionalRule = platform.LiteOptionalRule


class ConfigurationMode(str, Enum):
    """The three user-facing Stockpile rules profiles."""

    LITE = "lite"
    CLASSIC = "classic"
    DELUXE = "deluxe"

    # Source-compatible names. Iteration and serialization remain canonical.
    CORE = "classic"
    FULL = "deluxe"


@dataclass(frozen=True, slots=True)
class GameConfig:
    """One immutable, fully resolved Stockpile configuration.

    Optional inputs never survive as ``None`` in this value.  Each friendly
    switch is stored beside the one platform game produced from it, and the
    post-init checks prevent the public view from drifting away from the
    engine's effective rules.
    """

    mode: ConfigurationMode
    player_count: int
    round_count: int
    hand: bool
    fees: bool
    dividend: bool
    split: bool
    majority: bool
    stock_tracks: bool
    sell_order: bool
    action_space_mode: ActionSpaceMode
    configured_game: platform.ConfiguredGame

    def __post_init__(self) -> None:
        resolved_flags = {
            "hand": self.hand,
            "fees": self.fees,
            "dividend": self.dividend,
            "split": self.split,
            "majority": self.majority,
            "stock_tracks": self.stock_tracks,
            "sell_order": self.sell_order,
        }
        non_boolean = [
            name for name, value in resolved_flags.items() if type(value) is not bool
        ]
        if non_boolean:
            raise TypeError(
                "resolved GameConfig switches must be booleans: "
                + ", ".join(non_boolean)
            )

        parameters = self.configured_game.parameters
        rules = self.configured_game.rule_set
        mismatches: list[str] = []

        def require(name: str, actual: object, expected: object) -> None:
            if actual != expected:
                mismatches.append(f"{name}={actual!r}, expected {expected!r}")

        require("rules profile", rules.profile, self.mode.value)
        require("parameter profile", parameters.rules_profile, self.mode.value)
        require("player_count", rules.player_count, self.player_count)
        require("parameter player_count", parameters.player_count, self.player_count)
        require("round_count", rules.round_count, self.round_count)
        require("parameter round_count", parameters.round_count, self.round_count)
        require(
            "starting_shares_per_player",
            rules.starting_shares_per_player,
            int(self.hand),
        )
        require("trading_fees", rules.trading_fees, self.fees)
        require("forecast_dividends", rules.forecast_dividends, self.dividend)
        require(
            "dividend_reveal_choice",
            rules.dividend_reveal_choice,
            self.dividend,
        )
        require("stock_splits", rules.stock_splits, self.split)
        require("repeat_split_bonus", rules.repeat_split_bonus, self.split)
        require("majority_bonus", rules.majority_bonus, self.majority)
        require("advanced_price_tracks", rules.advanced_price_tracks, self.stock_tracks)
        require(
            "advanced_track_dividends",
            rules.advanced_track_dividends,
            self.stock_tracks and self.dividend,
        )
        require(
            "sequential_observable_selling",
            rules.sequential_observable_selling,
            self.sell_order,
        )
        require("action_space_mode", rules.action_space_mode, self.action_space_mode)
        require(
            "parameter action_space_mode",
            parameters.action_space_mode,
            self.action_space_mode,
        )
        require("stock_boom_cards", rules.stock_boom_cards, self.impact)
        require("stock_bust_cards", rules.stock_bust_cards, self.impact)
        require(
            "standard price ceiling",
            rules.standard_price_ceiling,
            None if self.mode is ConfigurationMode.LITE else 10,
        )
        if self.mode is ConfigurationMode.LITE:
            require("Lite stock splits", self.split, False)
            require("Lite majority bonuses", self.majority, False)
            require("Lite advanced price tracks", self.stock_tracks, False)
        else:
            require("Market Impact", self.impact, True)
        require(
            "Investors",
            self.investor,
            self.mode is ConfigurationMode.DELUXE,
        )
        if self.investor:
            require("enabled Investor count", len(rules.enabled_investors), 10)
            require(
                "unique Investor count",
                len(set(rules.enabled_investors)),
                10,
            )
        if mismatches:
            raise ValueError(
                "GameConfig does not match its configured platform game: "
                + "; ".join(mismatches)
            )

    @property
    def parameters(self) -> platform.GameParameters:
        return self.configured_game.parameters

    @property
    def rule_set(self) -> platform.RuleSet:
        return self.configured_game.rule_set

    @property
    def game(self) -> platform.StockpileGame:
        return self.configured_game.game

    @property
    def impact(self) -> bool:
        """Whether the Market Impact cards and Action phase are active."""

        return bool(self.rule_set.market_action_cards)

    @property
    def investor(self) -> bool:
        """Whether the profile's fixed Investor expansion is active."""

        return bool(self.rule_set.investors)

    @property
    def lite_options(self) -> tuple[LiteOptionalRule, ...]:
        """Compatibility projection of the supported Lite option names."""

        if self.mode is not ConfigurationMode.LITE:
            return ()
        enabled = {
            LiteOptionalRule.STARTING_SHARE: self.hand,
            LiteOptionalRule.TRADING_FEES: self.fees,
            LiteOptionalRule.DIVIDENDS: self.dividend,
            LiteOptionalRule.MARKET_IMPACT: self.impact,
        }
        return tuple(option for option in LiteOptionalRule if enabled[option])

    @property
    def deluxe_investors(self) -> bool:
        """Compatibility spelling for the now-fixed Deluxe Investor layer."""

        return self.investor


# The former public type remains an exact alias, not a second configuration
# representation that could disagree with GameConfig.
InterfaceConfiguration = GameConfig


@dataclass(frozen=True, slots=True)
class GameExplanation:
    """Presentation-ready setup, turn sequence, and ending for one game."""

    mode: ConfigurationMode
    title: str
    setup: tuple[str, ...]
    turns: tuple[str, ...]
    ending: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ActionCatalogComplexity:
    """Static OpenSpiel action and observation dimensions."""

    num_distinct_actions: int
    max_legal_actions: int
    max_chance_outcomes: int
    shared_action_head: int
    max_game_length: int
    observation_size: int


@dataclass(frozen=True, slots=True)
class InterfaceComplexity:
    """Static dimensions plus the platform's reachable infoset calculation.

    ``information_set_complexity`` is intentionally the same object returned
    by :func:`stockpile_platform.compute_information_set_complexity`.  In
    particular, the interface does not turn a bounded lower bound into an
    estimate or an apparent exact result.
    """

    configuration: GameConfig
    action_catalog: ActionCatalogComplexity
    information_set_complexity: platform.InformationSetComplexity

    @property
    def parameters(self) -> platform.GameParameters:
        return self.configuration.parameters

    @property
    def mode(self) -> ConfigurationMode:
        return self.configuration.mode

    @property
    def configured_game(self) -> platform.ConfiguredGame:
        return self.configuration.configured_game


@dataclass(frozen=True, slots=True)
class ResolvedInterfaceComplexity:
    """A live or remembered complexity result with auditable bounds.

    The contained :class:`InterfaceComplexity` has the same presentation shape
    regardless of where its traversal came from.  ``provenance`` identifies a
    current live traversal, a shipped preset, or a previously learned result;
    ``bounds`` keeps finite-tree ceilings separate from observed lower bounds.
    """

    complexity: InterfaceComplexity
    provenance: "ComplexityProvenance"
    bounds: "ComplexityBounds"

    @property
    def configuration(self) -> GameConfig:
        return self.complexity.configuration

    @property
    def action_catalog(self) -> ActionCatalogComplexity:
        return self.complexity.action_catalog

    @property
    def information_set_complexity(self) -> platform.InformationSetComplexity:
        return self.complexity.information_set_complexity

    @property
    def parameters(self) -> platform.GameParameters:
        return self.complexity.parameters

    @property
    def mode(self) -> ConfigurationMode:
        return self.complexity.mode

    @property
    def configured_game(self) -> platform.ConfiguredGame:
        return self.complexity.configured_game


_PROFILE_ALIASES = {
    "minimal_training": ConfigurationMode.LITE.value,
    "core": ConfigurationMode.CLASSIC.value,
    "full": ConfigurationMode.DELUXE.value,
    "expanded": ConfigurationMode.DELUXE.value,
    "expanded_variants": ConfigurationMode.DELUXE.value,
}

_LITE_OPTION_ALIASES = {
    "fee": LiteOptionalRule.TRADING_FEES.value,
    "fees": LiteOptionalRule.TRADING_FEES.value,
    "trading_fee": LiteOptionalRule.TRADING_FEES.value,
    "dividend": LiteOptionalRule.DIVIDENDS.value,
    "impact": LiteOptionalRule.MARKET_IMPACT.value,
    "market": LiteOptionalRule.MARKET_IMPACT.value,
    "market_impact": LiteOptionalRule.MARKET_IMPACT.value,
    "share": LiteOptionalRule.STARTING_SHARE.value,
    "starting_shares": LiteOptionalRule.STARTING_SHARE.value,
}


def _coerce_mode(profile: ConfigurationMode | str) -> ConfigurationMode:
    if isinstance(profile, ConfigurationMode):
        return profile
    raw_profile = str(profile).strip().lower()
    canonical_profile = _PROFILE_ALIASES.get(raw_profile, raw_profile)
    try:
        return ConfigurationMode(canonical_profile)
    except ValueError as error:
        choices = ", ".join(item.value for item in ConfigurationMode)
        raise ValueError(f"rules profile must be one of: {choices}") from error


def _coerce_lite_options(
    options: Iterable[LiteOptionalRule | str],
) -> tuple[LiteOptionalRule, ...]:
    if isinstance(options, (LiteOptionalRule, str)):
        options = (options,)

    enabled: set[LiteOptionalRule] = set()
    for option in options:
        if isinstance(option, LiteOptionalRule):
            enabled.add(option)
            continue
        raw_option = str(option).strip().lower()
        canonical_option = _LITE_OPTION_ALIASES.get(raw_option, raw_option)
        try:
            enabled.add(LiteOptionalRule(canonical_option))
        except ValueError as error:
            choices = ", ".join(item.value for item in LiteOptionalRule)
            raise ValueError(
                f"Lite optional rule must be one of: {choices}"
            ) from error
    return tuple(option for option in LiteOptionalRule if option in enabled)


_MODE_DEFAULTS: dict[ConfigurationMode, dict[str, bool]] = {
    ConfigurationMode.LITE: {
        "hand": False,
        "fees": False,
        "dividend": False,
        "split": False,
        "majority": False,
        "stock_tracks": False,
        "sell_order": False,
        "impact": False,
    },
    ConfigurationMode.CLASSIC: {
        "hand": True,
        "fees": True,
        "dividend": True,
        "split": True,
        "majority": True,
        "stock_tracks": False,
        "sell_order": True,
        "impact": True,
    },
    ConfigurationMode.DELUXE: {
        "hand": True,
        "fees": True,
        "dividend": True,
        "split": True,
        "majority": True,
        "stock_tracks": False,
        "sell_order": True,
        "impact": True,
    },
}


def _resolve_switch(name: str, value: bool | None, default: bool) -> bool:
    if value is None:
        return default
    if type(value) is not bool:
        raise TypeError(f"{name} must be true, false, or None")
    return value


def resolve_configuration(
    mode: ConfigurationMode | str,
    player_count: int = 2,
    round_count: int = 6,
    hand: bool | None = None,
    fees: bool | None = None,
    dividend: bool | None = None,
    split: bool | None = None,
    majority: bool | None = None,
    stock_tracks: bool | None = None,
    sell_order: bool | None = None,
    action_space_mode: ActionSpaceMode = "compact",
    impact: bool | None = None,
) -> GameConfig:
    """Resolve defaults and overrides into one immutable engine-backed value."""

    selected_mode = _coerce_mode(mode)
    defaults = _MODE_DEFAULTS[selected_mode]
    resolved = {
        "hand": _resolve_switch("hand", hand, defaults["hand"]),
        "fees": _resolve_switch("fees", fees, defaults["fees"]),
        "dividend": _resolve_switch(
            "dividend", dividend, defaults["dividend"]
        ),
        "split": _resolve_switch("split", split, defaults["split"]),
        "majority": _resolve_switch(
            "majority", majority, defaults["majority"]
        ),
        "stock_tracks": _resolve_switch(
            "stock_tracks", stock_tracks, defaults["stock_tracks"]
        ),
        "sell_order": _resolve_switch(
            "sell_order", sell_order, defaults["sell_order"]
        ),
    }
    if selected_mode is ConfigurationMode.LITE:
        resolved["impact"] = _resolve_switch(
            "impact", impact, defaults["impact"]
        )
        unsupported = [
            name
            for name in ("split", "majority", "stock_tracks")
            if resolved[name]
        ]
        if unsupported:
            raise ValueError(
                "Lite does not support enabled overrides: "
                + ", ".join(unsupported)
            )
    else:
        if impact is not None:
            if type(impact) is not bool:
                raise TypeError("impact must be true, false, or None")
            raise ValueError(
                "impact is fixed on for Classic and Deluxe and cannot be overridden"
            )
        resolved["impact"] = defaults["impact"]

    rule_overrides = dict(resolved)
    if selected_mode is not ConfigurationMode.LITE:
        # Market Impact is a fixed profile rule outside Lite. Omitting the
        # friendly key distinguishes the default from a forbidden override.
        rule_overrides.pop("impact")
    parameters = platform.GameParameters(
        player_count=player_count,
        rules_profile=selected_mode.value,
        round_count=round_count,
        deluxe_investors=selected_mode is ConfigurationMode.DELUXE,
        rule_overrides=rule_overrides,
        action_space_mode=action_space_mode,
    )
    configured_game = platform.configure_game(parameters)
    rules = configured_game.rule_set
    return GameConfig(
        mode=selected_mode,
        player_count=rules.player_count,
        round_count=rules.round_count,
        hand=resolved["hand"],
        fees=resolved["fees"],
        dividend=resolved["dividend"],
        split=resolved["split"],
        majority=resolved["majority"],
        stock_tracks=resolved["stock_tracks"],
        sell_order=resolved["sell_order"],
        action_space_mode=rules.action_space_mode,
        configured_game=configured_game,
    )


def create_configuration(
    profile: ConfigurationMode | str,
    *,
    player_count: int = 2,
    round_count: int = 6,
    lite_options: Iterable[LiteOptionalRule | str] = (),
    deluxe_investors: bool = False,
    action_space_mode: ActionSpaceMode = "compact",
) -> GameConfig:
    """Compatibility wrapper around :func:`resolve_configuration`.

    The old Lite option sequence maps to the supported corresponding friendly
    switches. Investors are now a fixed Deluxe rule, so ``False`` cannot turn
    them off and ``True`` remains invalid outside Deluxe.
    """

    selected_mode = _coerce_mode(profile)
    selected_lite_options = _coerce_lite_options(lite_options)
    if selected_mode is not ConfigurationMode.LITE and selected_lite_options:
        raise ValueError("lite_options can only be supplied for the Lite profile")
    if type(deluxe_investors) is not bool:
        raise TypeError("deluxe_investors must be true or false")
    if deluxe_investors and selected_mode is not ConfigurationMode.DELUXE:
        raise ValueError(
            "deluxe_investors is fixed on only for the Deluxe profile"
        )

    legacy_switches: dict[str, bool | None] = {
        "hand": None,
        "fees": None,
        "dividend": None,
        "split": None,
        "majority": None,
        "impact": None,
    }
    if selected_mode is ConfigurationMode.LITE:
        legacy_switches = {
            "hand": LiteOptionalRule.STARTING_SHARE in selected_lite_options,
            "fees": LiteOptionalRule.TRADING_FEES in selected_lite_options,
            "dividend": LiteOptionalRule.DIVIDENDS in selected_lite_options,
            "split": False,
            "majority": False,
            "impact": LiteOptionalRule.MARKET_IMPACT in selected_lite_options,
        }
    return resolve_configuration(
        selected_mode,
        player_count=player_count,
        round_count=round_count,
        action_space_mode=action_space_mode,
        **legacy_switches,
    )


def _coerce_interface_configuration(
    configuration: GameConfig | platform.ConfiguredGame,
) -> GameConfig:
    if isinstance(configuration, GameConfig):
        return configuration
    if isinstance(configuration, platform.ConfiguredGame):
        mode = _coerce_mode(configuration.parameters.rules_profile)
        rules = configuration.rule_set
        return GameConfig(
            mode=mode,
            player_count=rules.player_count,
            round_count=rules.round_count,
            hand=bool(rules.starting_shares_per_player),
            fees=bool(rules.trading_fees),
            dividend=bool(rules.forecast_dividends),
            split=bool(rules.stock_splits),
            majority=bool(rules.majority_bonus),
            stock_tracks=bool(rules.advanced_price_tracks),
            sell_order=bool(rules.sequential_observable_selling),
            action_space_mode=rules.action_space_mode,
            configured_game=configuration,
        )
    raise TypeError(
        "configuration must be a GameConfig or ConfiguredGame"
    )


def explain_configuration(
    configuration: GameConfig | platform.ConfiguredGame,
) -> GameExplanation:
    """Describe the effective game as ordered chance and player turns."""

    selected = _coerce_interface_configuration(configuration)
    rules = selected.rule_set
    player_word = "player" if rules.player_count == 1 else "players"
    round_word = "round" if rules.round_count == 1 else "rounds"
    title = (
        f"{selected.mode.value.title()} Stockpile — {rules.player_count} "
        f"{player_word}, {rules.company_count} companies, "
        f"{rules.round_count} {round_word}"
    )

    starting_shares = int(getattr(rules, "starting_shares_per_player", 1))
    if starting_shares:
        share_word = "share" if starting_shares == 1 else "shares"
        setup: list[str] = [
            (
                f"Chance gives each of the {rules.player_count} players "
                f"{starting_shares} starting {share_word} and selects the first player."
            )
        ]
    else:
        setup = [
            "Players begin without starting shares; chance selects the first player."
        ]
    if rules.advanced_price_tracks:
        setup.append(
            "Each company begins on its marked, company-specific starting space on "
            "the advanced side of the board."
        )
    else:
        setup.append(f"Every company starts at ${rules.starting_price}K.")
    if rules.investors:
        dealt = 4 if rules.player_count == 2 else 2
        kept = 2 if rules.player_count == 2 else 1
        setup.append(
            f"Chance deals {dealt} enabled Investors to each player; in first-player "
            f"order, each player chooses {kept}. Their selected Investors are then "
            "revealed and determine starting cash and available abilities."
        )
        setup.append(
            f"Players bid on {rules.stockpile_count} stockpiles using "
            f"{', '.join(f'${bid}K' for bid in rules.bid_values)}."
        )
        if "bill" in rules.enabled_investors:
            setup.append(
                "If selected, Bill changes starting cash but has no later ability turn."
            )
    else:
        setup.append(
            f"Each player starts with ${rules.starting_cash}K and bids on "
            f"{rules.stockpile_count} stockpiles using "
            f"{', '.join(f'${bid}K' for bid in rules.bid_values)}."
        )

    private_total = rules.player_count * rules.private_pairs_per_player
    remaining_pairs = rules.company_count - private_total
    public_pairs = 0
    if rules.two_player_topology != "official" and remaining_pairs:
        public_pairs = 1
        remaining_pairs -= 1
    blind_pairs = remaining_pairs if rules.blind_information_pairs else 0
    if not rules.blind_information_pairs:
        public_pairs += remaining_pairs

    information_parts = [
        f"{rules.private_pairs_per_player} private company/forecast "
        f"pair{'s' if rules.private_pairs_per_player != 1 else ''} per player"
    ]
    if public_pairs:
        information_parts.append(
            f"{public_pairs} public pair{'s' if public_pairs != 1 else ''}"
        )
    if blind_pairs:
        information_parts.append(
            f"{blind_pairs} blind pair{'s' if blind_pairs != 1 else ''} "
            "revealed only during Movement"
        )
    information_visibility = ", ".join(information_parts)

    turns: list[str] = [
        (
            f"At the start of each round, chance pairs the {rules.company_count} "
            f"companies with the configured forecasts and deals {information_visibility}."
        ),
        (
            f"Chance places one face-up Market card on each of the "
            f"{rules.stockpile_count} stockpiles and privately deals two cards to "
            f"each player for Supply batch 1 of {rules.supply_batches}."
        ),
    ]
    if "secretive_stuart" in rules.enabled_investors:
        turns.append(
            "Before each of Secretive Stuart's Supply turns, that player chooses "
            "whether to use the ability and place both supplied cards face down."
        )
    turns.append(
        "In first-player order for each Supply batch, a player chooses one of the "
        "two private cards, chooses its target stockpile (normally face up), and "
        "then chooses the stockpile for the other face-down card."
    )
    if rules.supply_batches > 1:
        turns.append(
            "After every player completes a Supply turn, chance deals the next "
            "two-card batch and the same player order repeats."
        )

    pre_demand = set(rules.enabled_investors) & {"maverick_mark", "wise_warren"}
    if pre_demand:
        turns.append(
            "Before bidding, eligible Investor owners act in first-player order: "
            + (
                "Maverick Mark may move a chosen face-up or face-down card between "
                "stockpiles; "
                if "maverick_mark" in pre_demand
                else ""
            )
            + (
                "Wise Warren may privately inspect a stockpile's face-down cards; "
                if "wise_warren" in pre_demand
                else ""
            )
            + "each ability may instead be skipped."
        )

    if "mayknow_martha" in rules.enabled_investors:
        turns.append(
            "Before each pending bid, MayKnow Martha's owner may spend the once-per-round "
            "ability to inspect an opponent's or blind information pair."
        )
    token_text = (
        "players place their two bidding tokens one token cycle at a time; for "
        "each token, its owner chooses"
        if rules.meeples_per_player == 2
        else "each player places their bidding token and chooses"
    )
    discount_text = (
        " If selected, Discount Donald pays the preceding bid-track value for a "
        "nonzero chosen bid."
        if "discount_donald" in rules.enabled_investors
        else ""
    )
    turns.extend(
        [
            (
                f"In first-player order, {token_text} a stockpile and then an "
                "affordable legal bid. An outbid token returns to the queue, "
                "so rebidding continues until every stockpile has a winner."
                + discount_text
            ),
            (
                "The game reserves and pays every winning bid, then awards each pile. "
                "Stock cards enter the winner's secret portfolio"
                + (
                    ", Trading Fees are paid or recorded as debt"
                    if rules.trading_fees
                    else ""
                )
                + (
                    ", and Boom/Bust cards enter the winner's action supply."
                    if rules.market_action_cards
                    else "."
                )
            ),
        ]
    )
    if "broker_bernie" in rules.enabled_investors:
        turns.append(
            "When a selected Broker Bernie acquires a Trading Fee, its face value is "
            "received as cash instead of paid or recorded as debt."
        )

    if rules.market_action_cards or "crazy_cramer" in rules.enabled_investors:
        action_parts: list[str] = []
        if rules.market_action_cards:
            action_parts.append(
                "each owner of an acquired Boom or Bust chooses its direction card and company"
            )
        if "crazy_cramer" in rules.enabled_investors:
            action_parts.append(
                "Crazy Cramer's owner may choose a one-space increase or decrease before or "
                "after acquired action cards"
            )
        turns.append(
            "In first-player order during the Action phase, "
            + "; ".join(action_parts)
            + "."
        )

    selling_choice = (
        "repeatedly sells one regular share, breaks and sells one split share, "
        "sells all holdings, or finishes that company"
        if rules.partial_sales
        else "chooses either to hold or sell all shares"
    )
    if rules.sequential_observable_selling:
        turns.append(
            f"In first-player order, each player visits every company they hold and "
            f"{selling_choice}; sale proceeds and public cash update immediately."
        )
    else:
        turns.append(
            "Without observing the others, each player privately "
            f"{selling_choice} for every company held; the committed sales then "
            "resolve together."
        )
    if "golden_graham" in rules.enabled_investors:
        turns.append(
            "A selected Golden Graham adds $1K per represented share to every sale."
        )
    turns.append(
        "The game reveals each information pair in deal order and applies its forecast "
        "to that company's price."
    )
    if rules.forecast_dividends:
        turns.append(
            (
                "For a Dividend forecast, each represented holding becomes a pay-or-decline "
                "player turn."
                if rules.dividend_reveal_choice
                else "A Dividend forecast automatically pays every shareholder."
            )
        )

    automatic_events: list[str] = []
    if rules.stock_splits:
        automatic_events.append("an upper-bound crossing triggers a stock split")
    if rules.repeat_split_bonus:
        automatic_events.append("later splits pay the configured repeat-split bonus")
    if rules.bankruptcy:
        automatic_events.append("a lower-bound crossing bankrupts the company")
    if rules.advanced_track_dividends:
        automatic_events.append("marked advanced-track spaces pay their dividends")
    if automatic_events:
        turns.append(
            "During every price movement, " + "; ".join(automatic_events) + "."
        )
    if "dividend_deborah" in rules.enabled_investors:
        turns.append(
            "After normal movement, each Dividend Deborah owner chooses a company for the "
            "Investor's extra dividend, or declines."
        )
    turns.append(
        "If rounds remain, the first-player marker moves clockwise and the next round "
        "returns to the information-deal turn."
    )

    ending: list[str] = []
    if rules.majority_bonus:
        ending.append(
            "After the final round, the game awards each company's majority-shareholder "
            "bonus, splitting tied bonuses according to the platform rules."
        )
    if "golden_graham" in rules.enabled_investors:
        ending.append(
            "A selected Golden Graham also adds $1K per represented share during "
            "final liquidation."
        )
    ending.append(
        "Every remaining regular and split share is liquidated at its final company "
        "price, and final cash determines the winner."
    )

    return GameExplanation(
        mode=selected.mode,
        title=title,
        setup=tuple(setup),
        turns=tuple(turns),
        ending=tuple(ending),
    )


def _action_catalog(
    configured_game: platform.ConfiguredGame,
) -> ActionCatalogComplexity:
    """Recalculate current static dimensions for a configured game."""

    report = platform.complexity_report(configured_game)
    return ActionCatalogComplexity(
        num_distinct_actions=int(report["num_distinct_actions"]),
        max_legal_actions=int(report["max_legal_actions"]),
        max_chance_outcomes=int(report["max_chance_outcomes"]),
        shared_action_head=int(report["shared_action_head"]),
        max_game_length=int(report["max_game_length"]),
        observation_size=int(report["observation_size"]),
    )


def compute_interface_complexity(
    configuration: GameConfig,
    *,
    max_states: int = 100_000,
    max_seconds: float = 10.0,
    require_exact: bool = False,
) -> InterfaceComplexity:
    """Force a live traversal and add current static dimensions.

    This function deliberately does not consult or update persistent memory.
    Its traversal object is returned by identity so existing programmatic
    callers retain the direct platform contract.  Applications that want
    cache-first behavior should call :func:`resolve_interface_complexity`.
    """

    if not isinstance(configuration, GameConfig):
        raise TypeError("configuration must be a resolved GameConfig")
    selected = configuration
    action_catalog = _action_catalog(selected.configured_game)
    information_set_complexity = platform.compute_information_set_complexity(
        selected.configured_game,
        max_states=max_states,
        max_seconds=max_seconds,
        require_exact=require_exact,
    )
    return InterfaceComplexity(
        configuration=selected,
        action_catalog=action_catalog,
        information_set_complexity=information_set_complexity,
    )


def resolve_interface_complexity(
    configuration: GameConfig,
    *,
    cache: "ComplexityCache | None" = None,
    cache_policy: CachePolicy = "prefer",
    max_states: int = 100_000,
    max_seconds: float = 10.0,
    require_exact: bool = False,
) -> ResolvedInterfaceComplexity:
    """Resolve complexity from memory or the live traversal.

    ``prefer`` returns the strongest valid remembered result without rerunning
    the tree search, even when its original budget differs from this call's
    budget.  A miss is calculated and learned.  ``refresh`` always calculates
    and learns a new result, while ``off`` performs a live calculation without
    reading or writing persistent memory.
    """

    from .complexity_cache import (
        compute_complexity_bounds,
        default_complexity_cache,
        live_complexity_provenance,
    )

    if cache_policy not in {"prefer", "refresh", "off"}:
        raise ValueError("cache_policy must be one of: prefer, refresh, off")

    if not isinstance(configuration, GameConfig):
        raise TypeError("configuration must be a resolved GameConfig")
    selected = configuration
    resolved_cache = None
    if cache_policy != "off":
        resolved_cache = cache if cache is not None else default_complexity_cache()

    if cache_policy == "prefer" and resolved_cache is not None:
        cached = resolved_cache.lookup(selected.configured_game)
        if cached is not None and (cached.result.exact or not require_exact):
            # Cache identity is based on normalized effective rules.  Reattach
            # the caller's current parameters so aliases and interface choices
            # are faithfully represented in the public result.
            remembered = cached.result.model_copy(
                update={"parameters": selected.parameters}
            )
            complexity = InterfaceComplexity(
                configuration=selected,
                action_catalog=_action_catalog(selected.configured_game),
                information_set_complexity=remembered,
            )
            return ResolvedInterfaceComplexity(
                complexity=complexity,
                provenance=cached.provenance,
                bounds=compute_complexity_bounds(
                    selected.configured_game,
                    remembered,
                ),
            )

    try:
        complexity = compute_interface_complexity(
            selected,
            max_states=max_states,
            max_seconds=max_seconds,
            require_exact=require_exact,
        )
    except platform.InformationSetEnumerationLimit as error:
        if resolved_cache is not None:
            try:
                resolved_cache.save(
                    selected.configured_game,
                    error.result,
                    source="learned",
                )
            except OSError:
                # Persistence is an optimization; preserve the platform's
                # original enumeration-limit result on read-only installs.
                pass
        raise
    if resolved_cache is not None:
        try:
            resolved_cache.save(
                selected.configured_game,
                complexity.information_set_complexity,
                source="learned",
            )
        except OSError:
            # A valid calculation remains useful even when its cache path is
            # not writable (for example, a system-wide package install).
            pass
    return ResolvedInterfaceComplexity(
        complexity=complexity,
        provenance=live_complexity_provenance(),
        bounds=compute_complexity_bounds(
            selected.configured_game,
            complexity.information_set_complexity,
        ),
    )


__all__ = [
    "ActionCatalogComplexity",
    "CachePolicy",
    "ConfigurationMode",
    "GameConfig",
    "GameExplanation",
    "InterfaceComplexity",
    "InterfaceConfiguration",
    "LiteOptionalRule",
    "ResolvedInterfaceComplexity",
    "compute_interface_complexity",
    "create_configuration",
    "explain_configuration",
    "resolve_configuration",
    "resolve_interface_complexity",
]
