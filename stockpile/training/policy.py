"""Inference policy exported by the sampled Deep CFR trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import NetworkConfig
from .encoding import (
    ENCODING_SCHEMA_VERSION,
    InformationInput,
    TraceSession,
    batch_information_inputs,
    reconstruct_information_input,
)
from .models import DeepCFRNetwork, masked_softmax, validate_network_contract


ACTION_SCHEMA_VERSION = "stockpile_compact_18_v1"
POLICY_SCHEMA_VERSION = 1


def torch_batch(
    inputs: list[InformationInput],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Collate encoder inputs and move numeric arrays to one Torch device."""

    collated = batch_information_inputs(inputs)
    tensors: dict[str, torch.Tensor] = {}
    for name in (
        "current",
        "history",
        "events",
        "history_lengths",
        "event_lengths",
        "legal_mask",
    ):
        value = collated[name]
        tensor = torch.as_tensor(value, device=device)
        if name in {"history_lengths", "event_lengths"}:
            tensor = tensor.to(torch.long)
        elif name == "legal_mask":
            tensor = tensor.to(torch.bool)
        else:
            tensor = tensor.to(dtype)
        tensors[name] = tensor
    return tensors


class DeepCFRPolicy:
    """An OpenSpiel-style legal-action policy backed by the average network.

    When an incremental ``trace_session`` is supplied, the caller must invoke
    ``trace_session.record_action`` before applying the selected action. Use no
    session when that lifecycle cannot be guaranteed; the policy will rebuild
    perfect recall from the state's complete action history instead.
    """

    def __init__(
        self,
        network: DeepCFRNetwork,
        *,
        device: str | torch.device = "cpu",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        validate_network_contract(network.config)
        self.device = torch.device(device)
        self.network = network.to(self.device)
        self.network.eval()
        self.metadata = dict(metadata or {})

    @torch.inference_mode()
    def probabilities_for_input(self, information: InformationInput) -> np.ndarray:
        batch = torch_batch([information], device=self.device)
        probabilities = masked_softmax(
            self.network(batch),
            batch["legal_mask"],
        )[0]
        return probabilities.detach().cpu().numpy().astype(np.float64, copy=False)

    def action_probabilities(
        self,
        state,
        player_id: int | None = None,
        *,
        trace_session: TraceSession | None = None,
    ) -> dict[int, float]:
        if state.is_terminal() or state.is_chance_node():
            return {}
        player = int(state.current_player()) if player_id is None else int(player_id)
        if player != int(state.current_player()):
            return {}
        information = (
            trace_session.snapshot(state, player)
            if trace_session is not None
            else reconstruct_information_input(state, player)
        )
        probabilities = self.probabilities_for_input(information)
        legal = [int(action) for action in state.legal_actions(player)]
        selected = {action: float(probabilities[action]) for action in legal}
        total = sum(selected.values())
        if not np.isfinite(total) or total <= 0:
            probability = 1.0 / len(legal)
            return {action: probability for action in legal}
        return {action: value / total for action, value in selected.items()}

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "DeepCFRPolicy":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("kind") != "stockpile_deep_cfr_policy":
            raise ValueError("not a Stockpile Deep CFR policy checkpoint")
        if payload.get("schema_version") != POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported Deep CFR policy checkpoint schema")
        metadata = payload.get("metadata", {})
        if metadata.get("encoder_schema_version") != ENCODING_SCHEMA_VERSION:
            raise ValueError("policy encoder schema is incompatible")
        if metadata.get("action_schema_version") != ACTION_SCHEMA_VERSION:
            raise ValueError("policy action schema is incompatible")
        network_config = NetworkConfig(**payload["network_config"])
        network = DeepCFRNetwork(network_config)
        network.load_state_dict(payload["strategy_network"])
        return cls(
            network,
            device=device,
            metadata=metadata,
        )


__all__ = [
    "ACTION_SCHEMA_VERSION",
    "DeepCFRPolicy",
    "POLICY_SCHEMA_VERSION",
    "torch_batch",
]
