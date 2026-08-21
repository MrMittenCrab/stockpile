"""PyTorch networks for history-aware sampled Deep CFR."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .config import NetworkConfig
from .encoding import ACTION_COUNT, EVENT_FEATURE_SIZE, HORIZON_SIZE, OBSERVATION_SIZE


def validate_network_contract(config: NetworkConfig) -> None:
    """Reject shape settings incompatible with the fixed v1 encoder/action schema."""

    actual = (
        config.observation_size,
        config.horizon_size,
        config.event_size,
        config.action_count,
    )
    expected = (OBSERVATION_SIZE, HORIZON_SIZE, EVENT_FEATURE_SIZE, ACTION_COUNT)
    if actual != expected:
        raise ValueError(
            "network input/action dimensions must match the fixed encoder "
            f"contract {expected}; got {actual}"
        )


def _last_recurrent_output(
    recurrent: nn.GRU,
    values: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Encode padded variable sequences, including genuinely empty histories."""

    batch = values.shape[0]
    if values.shape[1] == 0:
        return values.new_zeros((batch, recurrent.hidden_size))
    safe_lengths = lengths.clamp(min=1).to("cpu")
    packed = nn.utils.rnn.pack_padded_sequence(
        values,
        safe_lengths,
        batch_first=True,
        enforce_sorted=False,
    )
    _output, hidden = recurrent(packed)
    encoded = hidden[-1]
    return encoded * (lengths > 0).to(encoded.dtype).unsqueeze(1)


class InformationStateEncoder(nn.Module):
    """Fuse the current observation with ordered private/public history."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        validate_network_contract(config)
        self.config = config
        current_size = config.observation_size + config.horizon_size
        history_size = current_size + config.action_count
        self.current = nn.Sequential(
            nn.Linear(current_size, config.observation_hidden),
            nn.ReLU(),
            nn.LayerNorm(config.observation_hidden),
        )
        self.history = nn.GRU(
            history_size,
            config.history_hidden,
            batch_first=True,
        )
        self.events = nn.GRU(
            config.event_size,
            config.event_hidden,
            batch_first=True,
        )
        fused_size = (
            config.observation_hidden
            + config.history_hidden
            + config.event_hidden
        )
        self.fusion = nn.Sequential(
            nn.Linear(fused_size, config.fusion_hidden),
            nn.ReLU(),
            nn.LayerNorm(config.fusion_hidden),
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        current = self.current(batch["current"])
        history = _last_recurrent_output(
            self.history,
            batch["history"],
            batch["history_lengths"],
        )
        events = _last_recurrent_output(
            self.events,
            batch["events"],
            batch["event_lengths"],
        )
        return self.fusion(torch.cat((current, history, events), dim=1))


class DeepCFRNetwork(nn.Module):
    """One information-state encoder and an unconstrained action head."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = InformationStateEncoder(config)
        self.head = nn.Linear(config.fusion_hidden, config.action_count)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.encoder(batch))


def masked_softmax(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Normalize logits over legal actions only."""

    mask = legal_mask.to(torch.bool)
    if torch.any(mask.sum(dim=-1) == 0):
        raise ValueError("every policy row must contain at least one legal action")
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(masked, dim=-1)
    return probabilities * mask.to(probabilities.dtype)


def regret_matching_tensor(
    advantages: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply regret matching to one or more full action vectors."""

    mask = legal_mask.to(torch.bool)
    positive = torch.clamp(advantages, min=0) * mask.to(advantages.dtype)
    totals = positive.sum(dim=-1, keepdim=True)
    legal_count = mask.sum(dim=-1, keepdim=True).clamp(min=1)
    uniform = mask.to(advantages.dtype) / legal_count
    return torch.where(totals > 0, positive / totals.clamp(min=1e-30), uniform)


__all__ = [
    "DeepCFRNetwork",
    "InformationStateEncoder",
    "masked_softmax",
    "regret_matching_tensor",
    "validate_network_contract",
]
