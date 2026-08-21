"""Torch-free perfect-recall encoding for canonical two-player Stockpile Lite.

The game observer intentionally describes only the current decision.  Deep CFR
also needs perfect recall, so this module keeps a persistent, prefix-sharing
trace of a player's earlier decisions and actions.  Public and private-visible
events are encoded *only* from :attr:`InformationState.observable_history`;
the encoder never reads another player's private state.

The model-facing shapes are stable across the round curriculum::

    current       [batch, 259]       # 256 observation + 3 horizon values
    history       [batch, steps, 277] # current + one-hot 18-action choice
    events        [batch, events, 96]
    legal_mask    [batch, 18]

All stored trace values are immutable tuples.  ``TraceHandle.append`` creates
one new node whose ``parent`` is the previous handle, so reservoir samples
share prefixes through normal Python reference counting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from ..stockpile_platform import GameState, InformationState, RuleSet, observe_game_state


OBSERVATION_SIZE = 256
HORIZON_SIZE = 3
ACTION_COUNT = 18
CURRENT_FEATURE_SIZE = OBSERVATION_SIZE + HORIZON_SIZE
HISTORY_FEATURE_SIZE = CURRENT_FEATURE_SIZE + ACTION_COUNT
EVENT_FEATURE_SIZE = 96
ENCODING_SCHEMA_VERSION = "stockpile_lite_information_v1"


_EVENT_KINDS = (
    "other",
    "supply_private",
    "supply_public",
    "demand_pile",
    "demand_bid",
    "selling_commitment",
    "selling_batch",
)
_PHASES = (
    "setup",
    "information",
    "supply",
    "demand",
    "action",
    "selling",
    "movement",
    "terminal",
)
_STAGES = (
    "other",
    "supply_mode",
    "supply_card",
    "supply_up_pile",
    "supply_down_pile",
    "supply_commit",
    "demand_pile",
    "demand_bid",
)
_ACTION_NAMESPACES = (
    "card_slot",
    "pile",
    "bid_level",
    "done",
    "sale_mode",
    "other",
)

# Named offsets make the fixed event schema auditable in privacy and feature
# coverage tests.  Unassigned positions in the 96-wide vector are reserved for
# schema-compatible future additions and remain zero in v1.
EVENT_KIND_OFFSET = 0
EVENT_PHASE_OFFSET = 7
EVENT_ACTOR_OFFSET = 15
EVENT_STAGE_OFFSET = 18
EVENT_ACTION_NAMESPACE_OFFSET = 26
EVENT_SEQUENCE_INDEX = 32
EVENT_PUBLIC_ANCHOR_INDEX = 33
EVENT_PRIVATE_SEQUENCE_INDEX = 34
EVENT_ROUND_INDEX = 35
EVENT_ACTION_INDEX = 36
EVENT_ORDINAL_INDEX = 37
EVENT_SUPPLY_ACTOR_INDEX = 38
EVENT_SUPPLY_FACE_UP_PILE_INDEX = 39
EVENT_SUPPLY_FACE_DOWN_PILE_INDEX = 40
EVENT_SUPPLY_BOTH_DOWN_INDEX = 41
EVENT_SUPPLY_CARD_PRESENT_INDEX = 42
EVENT_SUPPLY_COMPANY_INDEX = 43
EVENT_SUPPLY_CARD_VALUE_INDEX = 44
EVENT_SUPPLY_STOCK_CARD_INDEX = 45
EVENT_DEMAND_PILE_INDEX = 46
EVENT_DEMAND_BID_LEVEL_INDEX = 47
EVENT_DEMAND_BID_AMOUNT_INDEX = 48
EVENT_SALE_COMPANY_INDEX = 49
EVENT_SALE_MODE_OFFSET = 50
EVENT_BATCH_SALES_OFFSET = 54
EVENT_BATCH_TOTALS_OFFSET = 66


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _lookup(mapping: Mapping[Any, Any], key: int) -> Any:
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key), {})


def _one_hot(vector: np.ndarray, offset: int, values: Sequence[str], value: str) -> None:
    try:
        index = values.index(value)
    except ValueError:
        index = values.index("other")
    vector[offset + index] = 1.0


def _action_parts(record: Mapping[str, Any], rule_set: RuleSet) -> tuple[str, int]:
    action_type = record.get("action_type")
    ordinal = record.get("ordinal")
    if isinstance(action_type, str) and ordinal is not None:
        return action_type, _as_int(ordinal)
    action = _as_int(record.get("action"), -1)
    if action < 0:
        return "other", 0
    try:
        return rule_set.action_codec.decode(action)
    except ValueError:
        return "other", 0


def _event_kind(record: Mapping[str, Any]) -> str:
    stage = str(record.get("stage", "other"))
    if stage == "selling_commitment":
        return "selling_commitment"
    if stage == "selling_batch":
        return "selling_batch"
    if stage == "supply_commit" or "public_supply_commit" in record:
        return "supply_public"
    if stage.startswith("supply_"):
        return "supply_private"
    if stage == "demand_pile":
        return "demand_pile"
    if stage == "demand_bid":
        return "demand_bid"
    return "other"


@dataclass(frozen=True, slots=True)
class VisibleEvent:
    """One exact visible record plus its fixed-width numeric representation."""

    kind: str
    payload_json: str
    features: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.features) != EVENT_FEATURE_SIZE:
            raise ValueError(
                f"event feature length must be {EVENT_FEATURE_SIZE}, "
                f"got {len(self.features)}"
            )


def encode_visible_event(record: Mapping[str, Any], rule_set: RuleSet) -> VisibleEvent:
    """Encode one already-filtered observable-history record.

    ``record`` is the sole source of game-state values.  ``rule_set`` is used
    only to decode public action ids and normalize fixed catalog values.
    """

    if not isinstance(record, Mapping):
        raise TypeError("observable-history records must be mappings")
    vector = np.zeros(EVENT_FEATURE_SIZE, dtype=np.float32)
    kind = _event_kind(record)
    _one_hot(vector, EVENT_KIND_OFFSET, _EVENT_KINDS, kind)
    _one_hot(vector, EVENT_PHASE_OFFSET, _PHASES, str(record.get("phase", "setup")))

    actor = _as_int(record.get("player"), -1)
    vector[EVENT_ACTOR_OFFSET + (actor if actor in (0, 1) else 2)] = 1.0

    stage = str(record.get("stage", "other"))
    _one_hot(
        vector,
        EVENT_STAGE_OFFSET,
        _STAGES,
        stage if stage in _STAGES else "other",
    )
    namespace, ordinal = _action_parts(record, rule_set)
    _one_hot(
        vector,
        EVENT_ACTION_NAMESPACE_OFFSET,
        _ACTION_NAMESPACES,
        namespace if namespace in _ACTION_NAMESPACES else "other",
    )

    game_length = max(1, rule_set.max_game_length)
    vector[EVENT_SEQUENCE_INDEX] = _as_int(record.get("sequence")) / game_length
    vector[EVENT_PUBLIC_ANCHOR_INDEX] = (
        _as_int(record.get("after_public_sequence")) / game_length
    )
    vector[EVENT_PRIVATE_SEQUENCE_INDEX] = (
        _as_int(record.get("private_sequence")) / game_length
    )
    vector[EVENT_ROUND_INDEX] = _as_int(record.get("round")) / 10.0
    action = _as_int(record.get("action"), -1)
    vector[EVENT_ACTION_INDEX] = (
        0.0 if action < 0 else (action + 1) / ACTION_COUNT
    )
    vector[EVENT_ORDINAL_INDEX] = (ordinal + 1) / ACTION_COUNT

    supply = record.get("public_supply_commit")
    if isinstance(supply, Mapping):
        supply_actor = _as_int(supply.get("player"), -1)
        vector[EVENT_SUPPLY_ACTOR_INDEX] = (
            0.0 if supply_actor < 0 else (supply_actor + 1) / 2.0
        )
        face_up_pile = _as_int(supply.get("face_up_pile"), -1)
        face_down_pile = _as_int(supply.get("face_down_pile"), -1)
        vector[EVENT_SUPPLY_FACE_UP_PILE_INDEX] = (
            0.0 if face_up_pile < 0 else (face_up_pile + 1) / 5.0
        )
        vector[EVENT_SUPPLY_FACE_DOWN_PILE_INDEX] = (
            0.0 if face_down_pile < 0 else (face_down_pile + 1) / 5.0
        )
        vector[EVENT_SUPPLY_BOTH_DOWN_INDEX] = float(
            bool(supply.get("both_face_down", False))
        )
        card = supply.get("face_up_card")
        if isinstance(card, Mapping):
            vector[EVENT_SUPPLY_CARD_PRESENT_INDEX] = 1.0
            company = _as_int(card.get("company_id"), -1)
            vector[EVENT_SUPPLY_COMPANY_INDEX] = (
                0.0 if company < 0 else (company + 1) / 6.0
            )
            value = card.get("value")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                vector[EVENT_SUPPLY_CARD_VALUE_INDEX] = float(value) / 10.0
            vector[EVENT_SUPPLY_STOCK_CARD_INDEX] = float(
                card.get("card_type") == "stock"
            )

    if stage == "demand_pile" and namespace == "pile":
        vector[EVENT_DEMAND_PILE_INDEX] = (ordinal + 1) / 5.0
    elif stage == "demand_bid" and namespace == "bid_level":
        vector[EVENT_DEMAND_BID_LEVEL_INDEX] = (ordinal + 1) / max(
            1, len(rule_set.bid_values)
        )
        if 0 <= ordinal < len(rule_set.bid_values):
            vector[EVENT_DEMAND_BID_AMOUNT_INDEX] = rule_set.bid_values[ordinal] / 50.0

    if stage == "selling_commitment":
        company = _as_int(record.get("company"), -1)
        vector[EVENT_SALE_COMPANY_INDEX] = (
            0.0 if company < 0 else (company + 1) / 6.0
        )
        if namespace == "done":
            sale_mode = 0
        elif namespace == "sale_mode" and ordinal in (0, 1, 2):
            sale_mode = ordinal + 1
        else:
            sale_mode = 0
        vector[EVENT_SALE_MODE_OFFSET + sale_mode] = 1.0

    if stage == "selling_batch":
        sales = record.get("sales", {})
        if isinstance(sales, Mapping):
            for player in range(2):
                player_sales = _lookup(sales, player)
                if not isinstance(player_sales, Mapping):
                    continue
                total = 0
                for company in range(6):
                    represented = max(0, _as_int(_lookup(player_sales, company)))
                    vector[
                        EVENT_BATCH_SALES_OFFSET + player * 6 + company
                    ] = represented / 10.0
                    total += represented
                vector[EVENT_BATCH_TOTALS_OFFSET + player] = total / 60.0

    vector.setflags(write=False)
    return VisibleEvent(
        kind=kind,
        payload_json=_canonical_json(record),
        features=tuple(float(value) for value in vector),
    )


def encode_visible_events(
    observable_history: Sequence[Mapping[str, Any]],
    rule_set: RuleSet,
) -> tuple[VisibleEvent, ...]:
    """Encode an ordered history already filtered for one observer."""

    return tuple(encode_visible_event(record, rule_set) for record in observable_history)


@dataclass(frozen=True, slots=True)
class HistoryStep:
    """One player decision and the action chosen from it."""

    player_id: int
    observation: tuple[float, ...]
    horizon_features: tuple[float, float, float]
    information_state_id: str
    new_visible_events: tuple[VisibleEvent, ...]
    action_id: int
    forced: bool

    def __post_init__(self) -> None:
        if self.player_id not in (0, 1):
            raise ValueError("history player_id must be 0 or 1")
        if len(self.observation) != OBSERVATION_SIZE:
            raise ValueError(
                f"history observation length must be {OBSERVATION_SIZE}"
            )
        if len(self.horizon_features) != HORIZON_SIZE:
            raise ValueError(f"horizon feature length must be {HORIZON_SIZE}")
        if not 0 <= self.action_id < ACTION_COUNT:
            raise ValueError(f"action_id must be in [0, {ACTION_COUNT})")

    def feature_vector(self) -> tuple[float, ...]:
        """Return the stable 277-value recurrent-history token."""

        action = [0.0] * ACTION_COUNT
        action[self.action_id] = 1.0
        return self.observation + self.horizon_features + tuple(action)


@dataclass(frozen=True, slots=True)
class TraceHandle:
    """Immutable handle to one node in a shared decision-history prefix tree."""

    parent: TraceHandle | None
    step: HistoryStep | None
    length: int
    digest: str

    def __post_init__(self) -> None:
        if self.step is None:
            if self.parent is not None or self.length != 0:
                raise ValueError("an empty trace cannot have a parent or positive length")
        elif self.parent is None or self.length != self.parent.length + 1:
            raise ValueError("a trace node must extend its parent by exactly one step")

    def append(self, step: HistoryStep) -> TraceHandle:
        payload = {
            "parent": self.digest,
            "player": step.player_id,
            "information": step.information_state_id,
            "events": [event.payload_json for event in step.new_visible_events],
            "action": step.action_id,
            "forced": step.forced,
        }
        return TraceHandle(
            parent=self,
            step=step,
            length=self.length + 1,
            digest=_digest_json(payload),
        )

    def steps(self) -> tuple[HistoryStep, ...]:
        """Materialize this prefix in chronological order."""

        values: list[HistoryStep] = []
        cursor: TraceHandle | None = self
        while cursor is not None and cursor.step is not None:
            values.append(cursor.step)
            cursor = cursor.parent
        values.reverse()
        return tuple(values)


EMPTY_TRACE = TraceHandle(
    parent=None,
    step=None,
    length=0,
    digest=hashlib.sha256(b"stockpile-empty-trace-v1").hexdigest(),
)


@dataclass(frozen=True, slots=True)
class InformationInput:
    """One actor's current perfect-recall network input."""

    player_id: int
    current_observation: tuple[float, ...]
    horizon_features: tuple[float, float, float]
    visible_event_features: tuple[VisibleEvent, ...]
    trace: TraceHandle
    information_state_id: str
    perfect_recall_id: str
    legal_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if self.player_id not in (0, 1):
            raise ValueError("information input player_id must be 0 or 1")
        if len(self.current_observation) != OBSERVATION_SIZE:
            raise ValueError(
                f"current observation length must be {OBSERVATION_SIZE}"
            )
        if len(self.horizon_features) != HORIZON_SIZE:
            raise ValueError(f"horizon feature length must be {HORIZON_SIZE}")
        if len(self.legal_mask) != ACTION_COUNT:
            raise ValueError(f"legal mask length must be {ACTION_COUNT}")

    @property
    def current_features(self) -> tuple[float, ...]:
        return self.current_observation + self.horizon_features

    @property
    def legal_action_ids(self) -> tuple[int, ...]:
        return tuple(index for index, legal in enumerate(self.legal_mask) if legal)


def _validate_rule_set(rule_set: RuleSet) -> None:
    unsupported = []
    if rule_set.profile != "lite":
        unsupported.append("profile")
    if rule_set.player_count != 2:
        unsupported.append("player_count")
    if rule_set.company_count != 6:
        unsupported.append("company_count")
    if rule_set.action_space_mode != "compact":
        unsupported.append("action_space_mode")
    if rule_set.action_codec.num_distinct_actions != ACTION_COUNT:
        unsupported.append("action_count")
    if rule_set.sequential_observable_selling:
        unsupported.append("sell_order")
    for name in (
        "trading_fees",
        "market_action_cards",
        "forecast_dividends",
        "stock_splits",
        "majority_bonus",
        "investors",
    ):
        if bool(getattr(rule_set, name)):
            unsupported.append(name)
    if rule_set.starting_shares_per_player != 0:
        unsupported.append("starting_shares_per_player")
    if not rule_set.partial_sales:
        unsupported.append("partial_sales")
    if unsupported:
        raise ValueError(
            "Deep CFR encoding supports only canonical two-player Lite with "
            "compact actions and sealed selling; incompatible: "
            + ", ".join(unsupported)
        )


def _rule_set_from(game_or_rule_set: Any) -> RuleSet:
    if isinstance(game_or_rule_set, RuleSet):
        rule_set = game_or_rule_set
    else:
        rule_set = getattr(game_or_rule_set, "rule_set", None)
    if not isinstance(rule_set, RuleSet):
        raise TypeError("expected a Stockpile game or RuleSet")
    _validate_rule_set(rule_set)
    return rule_set


def _horizon_features(state: GameState) -> tuple[float, float, float]:
    current = int(state.round)
    total = int(state.rule_set.round_count)
    # Remaining is inclusive: during round N, that round still has decisions
    # left, so a one-round game begins with 0.1 rather than 0.0.
    remaining = max(0, total - current + 1)
    return current / 10.0, total / 10.0, remaining / 10.0


def _build_from_information(
    state: GameState,
    player_id: int,
    trace: TraceHandle,
    information: InformationState,
) -> InformationInput:
    observation = tuple(float(value) for value in information.tensor)
    if len(observation) != OBSERVATION_SIZE:
        raise ValueError(
            f"Stockpile observer must provide {OBSERVATION_SIZE} values, "
            f"got {len(observation)}"
        )
    events = encode_visible_events(information.observable_history, state.rule_set)
    legal_mask = [False] * ACTION_COUNT
    for action_id in information.legal_action_ids:
        if not 0 <= action_id < ACTION_COUNT:
            raise ValueError(f"legal action {action_id} is outside the 18-action head")
        legal_mask[action_id] = True
    horizon = _horizon_features(state)
    perfect_recall_id = _digest_json(
        {
            "schema": ENCODING_SCHEMA_VERSION,
            "player": player_id,
            "information": information.information_state_id,
            "trace": trace.digest,
        }
    )
    return InformationInput(
        player_id=player_id,
        current_observation=observation,
        horizon_features=horizon,
        visible_event_features=events,
        trace=trace,
        information_state_id=information.information_state_id,
        perfect_recall_id=perfect_recall_id,
        legal_mask=tuple(legal_mask),
    )


class TraceSession:
    """Mutable cursor that builds one player's immutable shared trace."""

    def __init__(self, game_or_rule_set: Any, player_id: int | None = None):
        self.rule_set = _rule_set_from(game_or_rule_set)
        if player_id is not None and player_id not in (0, 1):
            raise ValueError("player_id must be 0 or 1")
        self.player_id = player_id
        self.trace = EMPTY_TRACE
        self._visible_payloads: tuple[str, ...] = ()
        self._pending: InformationInput | None = None
        self._pending_events: tuple[VisibleEvent, ...] = ()

    def _bind_player(self, state: GameState, player_id: int | None) -> int:
        resolved = int(state.current_player()) if player_id is None else int(player_id)
        if resolved not in (0, 1):
            raise ValueError("snapshot requires a player decision node")
        owner = self.player_id
        if owner is None:
            self.player_id = resolved
        elif owner != resolved:
            raise ValueError(
                f"trace session belongs to a different player, not {resolved}"
            )
        return resolved

    def _validate_state(self, state: GameState) -> None:
        if state.rule_set != self.rule_set:
            raise ValueError("trace session and state RuleSets do not match")
        _validate_rule_set(state.rule_set)

    def snapshot(
        self,
        state: GameState,
        player_id: int | None = None,
    ) -> InformationInput:
        """Snapshot a decision node without mutating the game state."""

        self._validate_state(state)
        resolved = self._bind_player(state, player_id)
        if int(state.current_player()) != resolved:
            raise ValueError("snapshot player must be the current actor")
        information, _legal = observe_game_state(self.rule_set, state, resolved)
        result = _build_from_information(state, resolved, self.trace, information)
        payloads = tuple(event.payload_json for event in result.visible_event_features)
        prefix_length = len(self._visible_payloads)
        if payloads[:prefix_length] != self._visible_payloads:
            raise RuntimeError("observable history is not append-only for this trace")
        same_pending_decision = (
            self._pending is not None
            and self._pending.trace is self.trace
            and self._pending.information_state_id == result.information_state_id
        )
        # Policy clients commonly query the same state more than once before
        # choosing an action.  The first query advances the visible-history
        # cursor; a repeated query must not replace its still-pending delta
        # with the now-empty suffix.
        if not same_pending_decision:
            self._pending_events = result.visible_event_features[prefix_length:]
        self._visible_payloads = payloads
        self._pending = result
        return result

    def record_action(
        self,
        state: GameState,
        action_id: int,
        forced: bool | None = None,
    ) -> TraceHandle:
        """Record the current actor's action, including one-legal forced nodes.

        Call this immediately before ``state.apply_action(action_id)``.  If no
        snapshot was taken explicitly, one is created first.
        """

        self._validate_state(state)
        player = self._bind_player(state, None)
        if int(state.current_player()) != player:
            raise ValueError("record_action requires this session's decision node")
        legal = tuple(int(value) for value in state.legal_actions(player))
        action_id = int(action_id)
        if action_id not in legal:
            raise ValueError(f"illegal action {action_id}; legal={list(legal)}")
        # A prior snapshot can only be reused while it still describes this
        # exact information state and trace prefix.
        information, _legal = observe_game_state(self.rule_set, state, player)
        if (
            self._pending is None
            or self._pending.trace is not self.trace
            or self._pending.information_state_id != information.information_state_id
        ):
            self.snapshot(state, player)
        assert self._pending is not None
        step = HistoryStep(
            player_id=player,
            observation=self._pending.current_observation,
            horizon_features=self._pending.horizon_features,
            information_state_id=self._pending.information_state_id,
            new_visible_events=self._pending_events,
            action_id=action_id,
            forced=len(legal) == 1 if forced is None else bool(forced),
        )
        self.trace = self.trace.append(step)
        self._pending = None
        self._pending_events = ()
        return self.trace

    @classmethod
    def from_state_history(cls, state: GameState, player_id: int) -> TraceSession:
        """Rebuild a player's trace by replaying ``state.history()``."""

        _validate_rule_set(state.rule_set)
        session = cls(state.get_game(), player_id)
        preset = getattr(state, "_preset", None)
        replay = GameState(state.get_game(), preset)
        for action in state.history():
            if replay.is_terminal():
                raise ValueError("state history contains actions after termination")
            if not replay.is_chance_node() and int(replay.current_player()) == player_id:
                session.snapshot(replay, player_id)
                legal = replay.legal_actions(player_id)
                session.record_action(
                    replay,
                    int(action),
                    forced=len(legal) == 1,
                )
            replay.apply_action(int(action))
        if replay.history() != state.history():
            raise RuntimeError("replayed history diverged from source state")
        return session


def reconstruct_trace(state: GameState, player_id: int) -> TraceHandle:
    """Return a perfect-recall prefix reconstructed from ``state.history()``."""

    return TraceSession.from_state_history(state, player_id).trace


def reconstruct_information_input(
    state: GameState,
    player_id: int | None = None,
) -> InformationInput:
    """Build a standalone input for the current actor by full history replay."""

    resolved = int(state.current_player()) if player_id is None else int(player_id)
    if resolved not in (0, 1) or int(state.current_player()) != resolved:
        raise ValueError("reconstruction requires the current decision actor")
    session = TraceSession.from_state_history(state, resolved)
    return session.snapshot(state, resolved)


def batch_information_inputs(
    inputs: Sequence[InformationInput],
) -> dict[str, np.ndarray]:
    """Pad immutable inputs into explicit NumPy model tensors.

    Empty batches and batches whose traces/events are all empty retain their
    rank: e.g. history is ``[B, 0, 277]`` rather than collapsing a dimension.
    """

    values = tuple(inputs)
    batch_size = len(values)
    history_lengths = np.asarray(
        [value.trace.length for value in values], dtype=np.int64
    )
    event_lengths = np.asarray(
        [len(value.visible_event_features) for value in values], dtype=np.int64
    )
    max_history = int(history_lengths.max()) if batch_size else 0
    max_events = int(event_lengths.max()) if batch_size else 0

    current = np.zeros((batch_size, CURRENT_FEATURE_SIZE), dtype=np.float32)
    history = np.zeros(
        (batch_size, max_history, HISTORY_FEATURE_SIZE), dtype=np.float32
    )
    events = np.zeros(
        (batch_size, max_events, EVENT_FEATURE_SIZE), dtype=np.float32
    )
    history_mask = np.zeros((batch_size, max_history), dtype=np.bool_)
    event_mask = np.zeros((batch_size, max_events), dtype=np.bool_)
    legal_mask = np.zeros((batch_size, ACTION_COUNT), dtype=np.bool_)

    for row, value in enumerate(values):
        current[row] = np.asarray(value.current_features, dtype=np.float32)
        legal_mask[row] = np.asarray(value.legal_mask, dtype=np.bool_)
        steps = value.trace.steps()
        for column, step in enumerate(steps):
            history[row, column] = np.asarray(
                step.feature_vector(), dtype=np.float32
            )
        history_mask[row, : len(steps)] = True
        for column, event in enumerate(value.visible_event_features):
            events[row, column] = np.asarray(event.features, dtype=np.float32)
        event_mask[row, : len(value.visible_event_features)] = True

    return {
        "current": current,
        "history": history,
        "events": events,
        "history_lengths": history_lengths,
        "event_lengths": event_lengths,
        "history_mask": history_mask,
        "event_mask": event_mask,
        "legal_mask": legal_mask,
    }


__all__ = [
    "ACTION_COUNT",
    "CURRENT_FEATURE_SIZE",
    "EMPTY_TRACE",
    "ENCODING_SCHEMA_VERSION",
    "EVENT_BATCH_SALES_OFFSET",
    "EVENT_BATCH_TOTALS_OFFSET",
    "EVENT_DEMAND_BID_AMOUNT_INDEX",
    "EVENT_DEMAND_BID_LEVEL_INDEX",
    "EVENT_DEMAND_PILE_INDEX",
    "EVENT_FEATURE_SIZE",
    "EVENT_KIND_OFFSET",
    "EVENT_SALE_COMPANY_INDEX",
    "EVENT_SALE_MODE_OFFSET",
    "EVENT_SUPPLY_BOTH_DOWN_INDEX",
    "EVENT_SUPPLY_CARD_PRESENT_INDEX",
    "EVENT_SUPPLY_COMPANY_INDEX",
    "EVENT_SUPPLY_FACE_DOWN_PILE_INDEX",
    "EVENT_SUPPLY_FACE_UP_PILE_INDEX",
    "HISTORY_FEATURE_SIZE",
    "HORIZON_SIZE",
    "HistoryStep",
    "InformationInput",
    "OBSERVATION_SIZE",
    "TraceHandle",
    "TraceSession",
    "VisibleEvent",
    "batch_information_inputs",
    "encode_visible_event",
    "encode_visible_events",
    "reconstruct_information_input",
    "reconstruct_trace",
]
