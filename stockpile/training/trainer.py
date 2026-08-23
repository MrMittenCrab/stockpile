"""Outcome-sampled, history-aware Deep CFR training for Stockpile Lite.

The full Stockpile tree is too large for canonical external-sampling Deep CFR.
This trainer therefore follows OpenSpiel's zero-baseline outcome-sampling
recursion while replacing tabular regrets and average strategies with neural
approximators.  It is intentionally restricted to the canonical two-player
Lite game with sealed selling and the compact 18-action head.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, TextIO

import numpy as np
import torch
from torch import nn

from .. import complexity_cache
from .. import stockpile_interface as interface
from .config import DeepCFRConfig
from .encoding import ENCODING_SCHEMA_VERSION, InformationInput, TraceSession
from .learning_curve import (
    LEARNING_CURVE_PLOT_NAME,
    LearningCurveStore,
    bootstrap_seed,
    checkpoint_evaluation_seed,
    cumulative_traversals,
    evaluate_learning_curve_checkpoint,
    evaluation_checkpoint_iterations,
    plot_learning_curve,
    stage_evaluation_seed,
    stage_traversals,
)
from .memory import ReservoirBuffer
from .models import DeepCFRNetwork, masked_softmax, validate_network_contract
from .policy import (
    ACTION_SCHEMA_VERSION,
    POLICY_SCHEMA_VERSION,
    DeepCFRPolicy,
    torch_batch,
)
from .regret import (
    REGRET_SIDECAR_SCHEMA_VERSION,
    RegretIterationCapture,
    RegretSidecarArchive,
    RegretTraversalCapture,
    TraversalRegretRecord,
)
from .sampling import (
    OutcomeSamplingReach,
    exploration_policy,
    forced_action,
    outcome_sampling_regret_target,
    outcome_sampling_value_estimate,
    regret_matching,
    zero_baseline_child_values,
)


LEGACY_CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 2
_RESUME_RECOVERY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class AdvantageSample:
    information: InformationInput
    iteration: int
    target: np.ndarray


@dataclass(frozen=True, slots=True)
class StrategySample:
    information: InformationInput
    iteration: int
    strategy: np.ndarray
    log_importance_weight: float
    zero_importance_weight: bool = False


@dataclass(frozen=True, slots=True)
class _TrajectoryNode:
    actor: int
    information: InformationInput
    policy: np.ndarray
    sample_policy: np.ndarray
    sampled_action: int
    reach: OutcomeSamplingReach


@dataclass(frozen=True, slots=True)
class _TraversalTelemetry:
    root_value: float
    absolute_regret_targets: tuple[float, ...]
    strategy_log_importance_weights: tuple[float, ...]
    signed_regret_record: TraversalRegretRecord


@dataclass(frozen=True, slots=True)
class _OptimizationTelemetry:
    loss: float
    maximum_gradient_norm: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    output_dir: Path
    completed_rounds: tuple[int, ...]
    final_checkpoint: Path
    final_policy: Path
    metrics: tuple[dict[str, Any], ...]
    target_reached: bool = False
    final_win_rate: float | None = None
    cumulative_traversals: int = 0


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _finite_mean(values: list[float], name: str) -> float:
    if not values:
        raise ValueError(f"{name} requires at least one value")
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError(f"{name} contains a nonfinite value")
    scale = max(abs(value) for value in values)
    if scale == 0:
        return 0.0
    result = scale * math.fsum(value / scale for value in values) / len(values)
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} mean is nonfinite")
    return result


def _absolute_distribution(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_maximum": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_p95": 0.0,
        }
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"{prefix} contains a nonfinite value")
    absolute = np.abs(array)
    return {
        f"{prefix}_maximum": float(np.max(absolute)),
        f"{prefix}_median": float(np.quantile(absolute, 0.5)),
        f"{prefix}_p95": float(np.quantile(absolute, 0.95)),
    }


def _choice(rng: random.Random, probabilities: np.ndarray) -> int:
    """Sample one full-vector action without depending on global RNG state."""

    total = float(np.sum(probabilities, dtype=np.float64))
    if not math.isfinite(total) or total <= 0:
        raise FloatingPointError("sampling policy has no finite probability mass")
    threshold = rng.random() * total
    cumulative = 0.0
    last_positive = -1
    for action, probability in enumerate(probabilities):
        probability = float(probability)
        if probability <= 0:
            continue
        last_positive = action
        cumulative += probability
        if threshold < cumulative:
            return action
    if last_positive < 0:
        raise FloatingPointError("sampling policy has no positive action")
    return last_positive


def _device(name: str) -> torch.device:
    if name != "auto":
        resolved = torch.device(name)
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        if name == "mps":
            raise RuntimeError(
                "MPS is not supported because correct outcome-sampling regret "
                "targets require float64 advantage networks; use --device cpu"
            )
        return resolved
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DeepCFRTrainer:
    """Train one final-horizon policy through weight-transfer stages."""

    def __init__(
        self,
        config: DeepCFRConfig,
        *,
        base_configuration: interface.GameConfig | None = None,
        output: TextIO | None = None,
    ) -> None:
        self.config = config
        validate_network_contract(config.network)
        self.output = sys.stdout if output is None else output
        self.device = _device(config.device)
        self.rng = random.Random(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(config.seed)

        target_rounds = config.curriculum.rounds[-1]
        self.base_configuration = base_configuration or interface.resolve_configuration(
            "lite",
            player_count=2,
            round_count=target_rounds,
            action_space_mode="compact",
        )
        self._validate_configuration(self.base_configuration, target_rounds)

        self.advantage_networks: list[DeepCFRNetwork] = []
        self.strategy_network: DeepCFRNetwork | None = None
        self.advantage_optimizers: list[torch.optim.Optimizer] = []
        self.strategy_optimizer: torch.optim.Optimizer | None = None
        self.advantage_memories: list[ReservoirBuffer[AdvantageSample]] = []
        self.strategy_memory: ReservoirBuffer[StrategySample] | None = None
        self.stage_configuration: interface.GameConfig | None = None
        self.stage_index = -1
        self.stage_iteration = 0
        self.global_iteration = 0
        self.stage_evaluated = False
        self.metrics: list[dict[str, Any]] = []
        self._resume_loaded = False
        self._loaded_checkpoint_path: Path | None = None
        self.regret_archive = RegretSidecarArchive(config.output_dir)
        self.learning_curve = LearningCurveStore(
            config.output_dir,
            run_seed=config.seed,
            evaluation_pairs=config.learning_curve_evaluation_pairs,
            bootstrap_resamples=config.learning_curve_bootstrap_resamples,
        )
        self._learning_curve_schedule = evaluation_checkpoint_iterations(
            config.iterations_per_stage,
            checkpoint_count=config.learning_curve_checkpoint_count,
        )
        self._until_target_reached = False
        self._stop_training = False
        self._last_evaluation_win_rate: float | None = None
        self._progress_is_tty = False
        try:
            self._progress_is_tty = bool(self.output.isatty())
        except (AttributeError, OSError):
            self._progress_is_tty = False
        self._progress_line_open = False
        self._progress_last_bucket = -1
        self._progress_completed = 0
        self._progress_total = 1
        self._epoch_started_at = time.perf_counter()

    @staticmethod
    def _validate_configuration(
        configuration: interface.GameConfig,
        target_rounds: int,
    ) -> None:
        if configuration.mode is not interface.ConfigurationMode.LITE:
            raise ValueError("Deep CFR currently supports only Stockpile Lite")
        if configuration.player_count != 2:
            raise ValueError("Deep CFR currently supports exactly two players")
        if configuration.round_count != target_rounds:
            raise ValueError(
                "configured rounds must match the curriculum's final horizon"
            )
        if configuration.action_space_mode != "compact":
            raise ValueError("Deep CFR requires the compact 18-action space")
        enabled = [
            name
            for name in (
                "impact",
                "hand",
                "fees",
                "dividend",
                "split",
                "majority",
                "stock_tracks",
                "sell_order",
            )
            if getattr(configuration, name)
        ]
        if enabled:
            raise ValueError(
                "Deep CFR requires default Lite rules; enabled overrides: "
                + ", ".join(enabled)
            )
        if configuration.game.num_distinct_actions() != 18:
            raise ValueError("Deep CFR requires a shape-stable 18-action head")
        if configuration.rule_set.standard_price_ceiling is not None:
            raise ValueError("Deep CFR requires uncapped Lite price semantics")

    def _stage_game(self, round_count: int) -> interface.GameConfig:
        configuration = interface.resolve_configuration(
            "lite",
            player_count=2,
            round_count=round_count,
            action_space_mode="compact",
        )
        self._validate_configuration(configuration, round_count)
        return configuration

    def _new_network(self, *, dtype: torch.dtype) -> DeepCFRNetwork:
        return DeepCFRNetwork(self.config.network).to(self.device, dtype=dtype)

    def _reset_stage(self, stage_index: int, *, transfer_weights: bool) -> None:
        round_count = self.config.curriculum.rounds[stage_index]
        self.stage_configuration = self._stage_game(round_count)
        if not transfer_weights or not self.advantage_networks:
            # Importance-weighted regrets can exceed float32 after only a few
            # dozen low-probability choices. Keeping the advantage path in
            # float64 avoids silent target overflow; the bounded probability
            # strategy network remains float32.
            self.advantage_networks = [
                self._new_network(dtype=torch.float64),
                self._new_network(dtype=torch.float64),
            ]
            self.strategy_network = self._new_network(dtype=torch.float32)
        assert self.strategy_network is not None
        self.advantage_optimizers = [
            torch.optim.Adam(network.parameters(), lr=self.config.learning_rate)
            for network in self.advantage_networks
        ]
        self.strategy_optimizer = torch.optim.Adam(
            self.strategy_network.parameters(),
            lr=self.config.learning_rate,
        )
        seed_base = self.config.seed + 10_000 * (stage_index + 1)
        self.advantage_memories = [
            ReservoirBuffer(self.config.memory_capacity, seed_base),
            ReservoirBuffer(self.config.memory_capacity, seed_base + 1),
        ]
        self.strategy_memory = ReservoirBuffer(
            self.config.memory_capacity,
            seed_base + 2,
        )
        self.stage_index = stage_index
        self.stage_iteration = 0
        self.stage_evaluated = False

    @torch.inference_mode()
    def _current_policy(
        self,
        actor: int,
        information: InformationInput,
    ) -> np.ndarray:
        network = self.advantage_networks[actor]
        network.eval()
        batch = torch_batch(
            [information],
            device=self.device,
            dtype=next(network.parameters()).dtype,
        )
        advantages = network(batch)[0].detach().cpu().numpy().astype(np.float64)
        return regret_matching(advantages, information.legal_mask)

    def _sample_chance(self, state) -> tuple[int, float]:
        outcomes = [
            (int(action), float(probability))
            for action, probability in state.chance_outcomes()
            if probability > 0
        ]
        if not outcomes:
            raise ValueError("reachable chance node has no positive-probability action")
        total = sum(probability for _, probability in outcomes)
        threshold = self.rng.random() * total
        cumulative = 0.0
        for action, probability in outcomes:
            cumulative += probability
            if threshold < cumulative:
                return action, probability / total
        action, probability = outcomes[-1]
        return action, probability / total

    def _traverse(
        self,
        update_player: int,
        traversal_ordinal: int = 0,
    ) -> _TraversalTelemetry:
        assert self.stage_configuration is not None
        assert self.strategy_memory is not None
        game = self.stage_configuration.game
        state = game.new_initial_state()
        sessions = [TraceSession(game, player) for player in range(2)]
        reach = OutcomeSamplingReach.root()
        nodes: list[_TrajectoryNode] = []
        absolute_regret_targets: list[float] = []
        strategy_log_weights: list[float] = []
        signed_regret = RegretTraversalCapture(
            player_id=update_player,
            traversal_ordinal=traversal_ordinal,
        )

        while not state.is_terminal():
            if state.is_chance_node():
                action, probability = self._sample_chance(state)
                reach = reach.after_chance(probability)
                state.apply_action(action)
                continue

            actor = int(state.current_player())
            legal = tuple(int(action) for action in state.legal_actions(actor))
            sole_action = forced_action(legal)
            if sole_action is not None:
                sessions[actor].record_action(state, sole_action, forced=True)
                state.apply_action(sole_action)
                continue

            information = sessions[actor].snapshot(state, actor)
            policy = self._current_policy(actor, information)
            sample_policy = (
                exploration_policy(
                    policy,
                    information.legal_mask,
                    self.config.exploration,
                )
                if actor == update_player
                else policy
            )
            sampled_action = _choice(self.rng, sample_policy)
            nodes.append(
                _TrajectoryNode(
                    actor=actor,
                    information=information,
                    policy=policy,
                    sample_policy=sample_policy,
                    sampled_action=sampled_action,
                    reach=reach,
                )
            )
            reach = reach.after_action(
                actor_is_update_player=actor == update_player,
                policy_probability=float(policy[sampled_action]),
                sample_probability=float(sample_policy[sampled_action]),
            )
            sessions[actor].record_action(state, sampled_action, forced=False)
            state.apply_action(sampled_action)

        child_value = float(state.returns()[update_player])
        for node in reversed(nodes):
            child_values = zero_baseline_child_values(
                node.sampled_action,
                child_value,
                node.sample_policy,
                node.information.legal_mask,
            )
            child_value = float(
                outcome_sampling_value_estimate(
                    child_values,
                    node.policy,
                    node.information.legal_mask,
                )
            )
            if node.actor != update_player:
                continue

            target = outcome_sampling_regret_target(
                child_values,
                node.policy,
                node.information.legal_mask,
                node.reach,
            )
            legal_target = target[np.asarray(node.information.legal_mask)]
            if not np.all(np.isfinite(legal_target)):
                raise FloatingPointError("outcome-sampling regret target is nonfinite")
            signed_regret.add_target(
                perfect_recall_id=node.information.perfect_recall_id,
                legal_mask=node.information.legal_mask,
                target=target,
            )
            absolute_regret_targets.extend(
                float(abs(value)) for value in legal_target
            )
            self.advantage_memories[update_player].append(
                AdvantageSample(
                    information=node.information,
                    iteration=self.stage_iteration,
                    target=target,
                )
            )
            log_weight = float(
                node.reach.log_my_reach - node.reach.log_sample_reach
            )
            if not math.isfinite(log_weight):
                raise FloatingPointError(
                    "average-strategy log importance weight is nonfinite"
                )
            strategy_log_weights.append(log_weight)
            self.strategy_memory.append(
                StrategySample(
                    information=node.information,
                    iteration=self.stage_iteration,
                    strategy=node.policy.astype(np.float32),
                    log_importance_weight=log_weight,
                    zero_importance_weight=node.reach.my_reach_is_zero,
                )
            )
        if not math.isfinite(child_value):
            raise FloatingPointError("sampled root value is nonfinite")
        return _TraversalTelemetry(
            root_value=child_value,
            absolute_regret_targets=tuple(absolute_regret_targets),
            strategy_log_importance_weights=tuple(strategy_log_weights),
            signed_regret_record=signed_regret.finish(),
        )

    def _learn_advantages(self, player: int) -> _OptimizationTelemetry | None:
        memory = self.advantage_memories[player]
        if not memory:
            return None
        network = self.advantage_networks[player]
        optimizer = self.advantage_optimizers[player]
        losses: list[float] = []
        gradient_norms: list[float] = []
        network.train()
        for _ in range(self.config.advantage_train_steps):
            samples = memory.sample(self.config.batch_size)
            batch = torch_batch(
                [sample.information for sample in samples],
                device=self.device,
                dtype=torch.float64,
            )
            target = torch.as_tensor(
                np.stack([sample.target for sample in samples]),
                dtype=torch.float64,
                device=self.device,
            )
            if not torch.all(torch.isfinite(target)):
                raise FloatingPointError("advantage training target is nonfinite")
            iterations = torch.as_tensor(
                [sample.iteration for sample in samples],
                dtype=torch.float32,
                device=self.device,
            )
            predicted = network(batch)
            if not torch.all(torch.isfinite(predicted)):
                raise FloatingPointError("advantage network output is nonfinite")
            mask = batch["legal_mask"].to(predicted.dtype)
            squared = (predicted - target).square() * mask
            row_loss = squared.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            weights = iterations / iterations.mean().clamp(min=1.0)
            loss = (row_loss * weights).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("advantage loss is nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                network.parameters(),
                self.config.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            if not all(
                torch.all(torch.isfinite(parameter)).item()
                for parameter in network.parameters()
            ):
                raise FloatingPointError("advantage network parameter is nonfinite")
            losses.append(float(loss.detach().cpu()))
            gradient_norms.append(float(gradient_norm.detach().cpu()))
        network.eval()
        return _OptimizationTelemetry(
            loss=_finite_mean(losses, "advantage losses"),
            maximum_gradient_norm=max(gradient_norms),
        )

    def _learn_strategy(self) -> _OptimizationTelemetry | None:
        assert self.strategy_network is not None
        assert self.strategy_optimizer is not None
        assert self.strategy_memory is not None
        if not self.strategy_memory:
            return None
        network = self.strategy_network
        losses: list[float] = []
        gradient_norms: list[float] = []
        network.train()
        for _ in range(self.config.strategy_train_steps):
            samples = self.strategy_memory.sample(self.config.batch_size)
            batch = torch_batch(
                [sample.information for sample in samples],
                device=self.device,
            )
            target = torch.as_tensor(
                np.stack([sample.strategy for sample in samples]),
                dtype=torch.float32,
                device=self.device,
            )
            logits = network(batch)
            predicted = masked_softmax(logits, batch["legal_mask"])
            if not torch.all(torch.isfinite(predicted)):
                raise FloatingPointError("strategy network output is nonfinite")
            mask = batch["legal_mask"].to(predicted.dtype)
            row_loss = ((predicted - target).square() * mask).sum(dim=1)
            row_loss /= mask.sum(dim=1).clamp(min=1)

            log_weights = torch.as_tensor(
                [
                    -torch.inf
                    if sample.zero_importance_weight
                    else sample.log_importance_weight
                    + math.log(float(sample.iteration))
                    for sample in samples
                ],
                dtype=torch.float64,
                device=self.device,
            )
            finite = torch.isfinite(log_weights)
            if not torch.any(finite):
                network.eval()
                return None
            maximum = torch.max(log_weights[finite])
            weights = torch.zeros_like(log_weights)
            weights[finite] = torch.exp(log_weights[finite] - maximum)
            weights = (weights / weights.sum().clamp(min=1e-300)).to(
                row_loss.dtype
            )
            loss = torch.sum(row_loss * weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("strategy loss is nonfinite")
            self.strategy_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                network.parameters(),
                self.config.gradient_clip,
                error_if_nonfinite=True,
            )
            self.strategy_optimizer.step()
            if not all(
                torch.all(torch.isfinite(parameter)).item()
                for parameter in network.parameters()
            ):
                raise FloatingPointError("strategy network parameter is nonfinite")
            losses.append(float(loss.detach().cpu()))
            gradient_norms.append(float(gradient_norm.detach().cpu()))
        network.eval()
        return _OptimizationTelemetry(
            loss=_finite_mean(losses, "strategy losses"),
            maximum_gradient_norm=max(gradient_norms),
        )

    def _metadata(self) -> dict[str, Any]:
        assert self.stage_configuration is not None
        return {
            "algorithm": self.config.algorithm,
            "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
            "rounds": self.stage_configuration.round_count,
            "players": 2,
            "mode": "lite",
            "market_impact": self.stage_configuration.impact,
            "sell_order": False,
            "action_space_mode": "compact",
            "action_count": self.config.network.action_count,
            "observation_size": self.config.network.observation_size,
            "information_encoder": "strict_visible_history_v1",
            "encoder_schema_version": ENCODING_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "advantage_dtype": "float64",
            "strategy_dtype": "float32",
            "utility": "terminal_rank_zero_sum",
            "semantic_fingerprint": complexity_cache.semantic_fingerprint(
                self.stage_configuration.configured_game
            ),
            "price_semantics": {
                "standard_price_ceiling": (
                    self.stage_configuration.rule_set.standard_price_ceiling
                ),
            },
            "equilibrium_claim": False,
            "resolved_device": str(self.device),
            "sampled_regret_sidecar_schema_version": (
                REGRET_SIDECAR_SCHEMA_VERSION
            ),
        }

    def _training_signature(self) -> dict[str, Any]:
        payload = _json_value(asdict(self.config))
        payload.pop("output_dir", None)
        return payload

    def _stage_dir(self) -> Path:
        assert self.stage_configuration is not None
        return self.config.output_dir / f"round_{self.stage_configuration.round_count:02d}"

    def _full_checkpoint_payload(self) -> dict[str, Any]:
        assert self.strategy_network is not None
        assert self.strategy_optimizer is not None
        assert self.strategy_memory is not None
        return {
            "kind": "stockpile_deep_cfr_training",
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "metadata": self._metadata(),
            "training_signature": self._training_signature(),
            "stage_index": self.stage_index,
            "stage_iteration": self.stage_iteration,
            "global_iteration": self.global_iteration,
            "stage_evaluated": self.stage_evaluated,
            "advantage_networks": [
                network.state_dict() for network in self.advantage_networks
            ],
            "strategy_network": self.strategy_network.state_dict(),
            "advantage_optimizers": [
                optimizer.state_dict() for optimizer in self.advantage_optimizers
            ],
            "strategy_optimizer": self.strategy_optimizer.state_dict(),
            "advantage_memories": [
                memory.state_dict() for memory in self.advantage_memories
            ],
            "strategy_memory": self.strategy_memory.state_dict(),
            "trainer_rng_state": self.rng.getstate(),
            "python_global_rng_state": random.getstate(),
            "numpy_global_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "torch_mps_rng_state": (
                torch.mps.get_rng_state()
                if torch.backends.mps.is_available()
                else None
            ),
            "sampled_regret_telemetry": self.regret_archive.checkpoint_state(
                embed_records=True
            ),
            "metrics": self.metrics,
        }

    def save_checkpoint(
        self,
        *,
        preserve_intermediate: bool = False,
        cumulative_traversal_count: int | None = None,
    ) -> tuple[Path, Path]:
        """Atomically replace the full-resume and compact policy artifacts."""

        assert self.strategy_network is not None
        stage_dir = self._stage_dir()
        stage_dir.mkdir(parents=True, exist_ok=True)
        full_path = stage_dir / "full.pt"
        full_temporary = stage_dir / "full.pt.tmp"
        torch.save(self._full_checkpoint_payload(), full_temporary)
        full_temporary.replace(full_path)

        policy_path = stage_dir / "policy.pt"
        policy_temporary = stage_dir / "policy.pt.tmp"
        torch.save(
            {
                "kind": "stockpile_deep_cfr_policy",
                "schema_version": POLICY_SCHEMA_VERSION,
                "network_config": asdict(self.config.network),
                "strategy_network": self.strategy_network.state_dict(),
                "metadata": self._metadata(),
            },
            policy_temporary,
        )
        policy_temporary.replace(policy_path)

        if preserve_intermediate:
            if cumulative_traversal_count is None:
                raise ValueError(
                    "cumulative_traversal_count is required when preserving "
                    "intermediate checkpoints"
                )
            archive_dir = (
                self.config.output_dir
                / "checkpoints"
                / f"traversals_{int(cumulative_traversal_count):09d}"
            )
            archive_dir.mkdir(parents=True, exist_ok=True)
            archived_full = archive_dir / "full.pt"
            archived_policy = archive_dir / "policy.pt"
            if not archived_full.exists():
                archived_full.write_bytes(full_path.read_bytes())
            if not archived_policy.exists():
                archived_policy.write_bytes(policy_path.read_bytes())

        return full_path, policy_path

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore an exact same-stage run, including memories and RNGs."""

        checkpoint_path = Path(path).expanduser().resolve(strict=False)
        payload = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        if payload.get("kind") != "stockpile_deep_cfr_training":
            raise ValueError("not a Stockpile Deep CFR training checkpoint")
        checkpoint_schema = payload.get("schema_version")
        if checkpoint_schema not in {
            LEGACY_CHECKPOINT_SCHEMA_VERSION,
            CHECKPOINT_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported Deep CFR checkpoint schema")
        if payload.get("training_signature") != self._training_signature():
            raise ValueError("checkpoint training configuration does not match")
        stage_index = int(payload["stage_index"])
        if not 0 <= stage_index < len(self.config.curriculum.rounds):
            raise ValueError("checkpoint curriculum stage is out of range")
        stage_configuration = self._stage_game(
            self.config.curriculum.rounds[stage_index]
        )
        checkpoint_metadata = payload.get("metadata", {})
        expected_fingerprint = complexity_cache.semantic_fingerprint(
            stage_configuration.configured_game
        )
        if checkpoint_metadata.get("semantic_fingerprint") != expected_fingerprint:
            raise ValueError("checkpoint game semantics do not match")
        if checkpoint_metadata.get("encoder_schema_version") != (
            ENCODING_SCHEMA_VERSION
        ):
            raise ValueError("checkpoint encoder schema does not match")
        if checkpoint_metadata.get("action_schema_version") != ACTION_SCHEMA_VERSION:
            raise ValueError("checkpoint action schema does not match")
        if checkpoint_metadata.get("advantage_dtype") != "float64":
            raise ValueError("checkpoint advantage dtype does not match")
        if checkpoint_metadata.get("resolved_device") != str(self.device):
            raise ValueError("checkpoint resolved device does not match")
        if len(payload["advantage_networks"]) != 2:
            raise ValueError("checkpoint must contain two advantage networks")
        regret_state = payload.get("sampled_regret_telemetry")
        if checkpoint_schema == CHECKPOINT_SCHEMA_VERSION:
            if regret_state is None:
                raise ValueError("checkpoint is missing sampled regret telemetry")
            self.regret_archive.validate_checkpoint_state(regret_state)

        self._reset_stage(stage_index, transfer_weights=False)
        assert self.stage_configuration is not None
        for network, state in zip(
            self.advantage_networks,
            payload["advantage_networks"],
            strict=True,
        ):
            network.load_state_dict(state)
        assert self.strategy_network is not None
        self.strategy_network.load_state_dict(payload["strategy_network"])
        for optimizer, state in zip(
            self.advantage_optimizers,
            payload["advantage_optimizers"],
            strict=True,
        ):
            optimizer.load_state_dict(state)
        assert self.strategy_optimizer is not None
        self.strategy_optimizer.load_state_dict(payload["strategy_optimizer"])
        for memory, state in zip(
            self.advantage_memories,
            payload["advantage_memories"],
            strict=True,
        ):
            memory.load_state_dict(state)
        assert self.strategy_memory is not None
        self.strategy_memory.load_state_dict(payload["strategy_memory"])
        self.stage_iteration = int(payload["stage_iteration"])
        self.global_iteration = int(payload["global_iteration"])
        self.stage_evaluated = bool(payload.get("stage_evaluated", False))
        self.metrics = list(payload.get("metrics", []))
        self.rng.setstate(payload["trainer_rng_state"])
        random.setstate(payload["python_global_rng_state"])
        np.random.set_state(payload["numpy_global_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"].to("cpu"))
        cuda_states = payload.get("torch_cuda_rng_states")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.to("cpu") for state in cuda_states])
        mps_state = payload.get("torch_mps_rng_state")
        if mps_state is not None and torch.backends.mps.is_available():
            torch.mps.set_rng_state(mps_state.to("cpu"))
        if regret_state is not None:
            self.regret_archive.restore_checkpoint_state(regret_state)
        self._rewrite_metrics_file(preserve_existing=True)
        self._resume_loaded = True
        self._loaded_checkpoint_path = checkpoint_path

    @staticmethod
    def _write_recovery_file(path: Path, content: bytes) -> None:
        """Create one immutable recovery file, accepting an identical retry."""

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if not path.is_file() or path.read_bytes() != content:
                    raise RuntimeError(
                        "resume recovery archive conflicts with existing file: "
                        f"{path}"
                    )
        finally:
            temporary.unlink(missing_ok=True)

    def _preserve_resume_reconciliation_input(
        self,
        path: Path,
        replacement: bytes,
    ) -> Path | None:
        """Content-address differing bytes before resume reconciliation."""

        if not path.is_file():
            return None
        existing = path.read_bytes()
        if existing == replacement:
            return None

        digest = hashlib.sha256(existing).hexdigest()
        recovery_dir = (
            self.config.output_dir
            / "recovery"
            / "resume_reconciliation"
            / path.name
            / f"sha256-{digest}"
        )
        recovery_dir.mkdir(parents=True, exist_ok=True)
        recovered_path = recovery_dir / path.name
        self._write_recovery_file(recovered_path, existing)
        provenance = {
            "kind": "stockpile_deep_cfr_resume_recovery",
            "schema_version": _RESUME_RECOVERY_SCHEMA_VERSION,
            "reason": "resume_reconciliation",
            "source_path": path.name,
            "preserved_sha256": digest,
            "preserved_byte_count": len(existing),
        }
        provenance_bytes = (
            json.dumps(
                provenance,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        self._write_recovery_file(recovery_dir / "provenance.json", provenance_bytes)
        return recovered_path

    def _render_run_config(self) -> bytes:
        return (
            json.dumps(
                {
                    "sampled_regret_telemetry": {
                        "record_schema_version": REGRET_SIDECAR_SCHEMA_VERSION,
                    },
                    "training": _json_value(asdict(self.config)),
                    "base_game": {
                        "mode": "lite",
                        "players": 2,
                        "rounds": self.base_configuration.round_count,
                        "market_impact": self.base_configuration.impact,
                        "sell_order": False,
                        "action_space_mode": "compact",
                        "price_semantics": {
                            "standard_price_ceiling": (
                                self.base_configuration.rule_set.standard_price_ceiling
                            ),
                        },
                    },
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    def _write_run_config(self, *, preserve_existing: bool = False) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.output_dir / "config.json"
        rendered = self._render_run_config()
        if preserve_existing:
            self._preserve_resume_reconciliation_input(path, rendered)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_bytes(rendered)
        temporary.replace(path)

    def _append_metric(self, metric: dict[str, Any]) -> None:
        self.metrics.append(metric)
        path = self.config.output_dir / "metrics.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    _json_value(metric),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )

    def _render_metrics(self) -> bytes:
        return "".join(
            json.dumps(
                _json_value(metric),
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for metric in self.metrics
        ).encode("utf-8")

    def _rewrite_metrics_file(self, *, preserve_existing: bool = False) -> None:
        """Make the human-readable log an exact projection of checkpoint state."""

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.output_dir / "metrics.jsonl"
        rendered = self._render_metrics()
        if preserve_existing:
            self._preserve_resume_reconciliation_input(path, rendered)
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_bytes(rendered)
        temporary.replace(path)

    def train(
        self,
        *,
        resume: str | Path | None = None,
        overwrite: bool = False,
    ) -> TrainingResult:
        """Run all requested stages and return paths to the final artifacts."""

        metrics_path = self.config.output_dir / "metrics.jsonl"
        if resume is None:
            if self.config.output_dir.exists() and not self.config.output_dir.is_dir():
                raise ValueError(
                    f"output path is not a directory: {self.config.output_dir}"
                )
            existing = (
                []
                if not self.config.output_dir.exists()
                else [
                    entry
                    for entry in self.config.output_dir.iterdir()
                    if entry.name != "run.json"
                ]
            )
            if existing and not overwrite:
                raise ValueError(
                    f"output directory is not empty: {self.config.output_dir}; "
                    "choose another directory or enable overwrite"
                )
            self._write_run_config()
            metrics_path.write_text("", encoding="utf-8")
            self.learning_curve.reset()
            self._reset_stage(0, transfer_weights=False)
        else:
            if overwrite:
                raise ValueError("overwrite cannot be combined with resume")
            checkpoint = Path(resume).expanduser().resolve(strict=False)
            output_root = self.config.output_dir.expanduser().resolve(strict=False)
            in_place_resume = checkpoint.is_relative_to(output_root)
            if self._resume_loaded:
                if self._loaded_checkpoint_path != checkpoint:
                    raise ValueError(
                        "preloaded resume checkpoint does not match requested path"
                    )
            else:
                self.load_checkpoint(checkpoint)
            self.learning_curve.load()
            # A preload is single-use.  A later train() call on this object
            # must restore the requested checkpoint again rather than silently
            # continuing mutated state from an earlier call.
            self._resume_loaded = False
            self._loaded_checkpoint_path = None
            # Do not touch run metadata until the checkpoint has passed all
            # configuration, semantics, schema, and device validation.  An
            # in-place run's config is descriptive metadata, not checkpoint
            # state; retain its exact bytes (including additive future fields)
            # instead of regenerating it.  Forks need a destination config.
            if not in_place_resume or not (
                self.config.output_dir / "config.json"
            ).is_file():
                self._write_run_config(preserve_existing=True)

        completed: list[int] = list(
            self.config.curriculum.rounds[: self.stage_index]
        )
        final_checkpoint = final_policy = Path()
        self._until_target_reached = False
        self._stop_training = False
        self._last_evaluation_win_rate = None
        if self.config.until_win_rate_enabled:
            required = int(self.config.until_win_rate_consecutive)
            streak = self.learning_curve.consecutive_win_rate_streak(
                float(self.config.until_win_rate)
            )
            if streak >= required and self.learning_curve.checkpoints:
                latest = self.learning_curve.checkpoints[-1]
                self._until_target_reached = True
                self._stop_training = True
                self._last_evaluation_win_rate = float(latest["win_rate"])
                print(
                    "TARGET REACHED: "
                    f"{100.0 * float(latest['win_rate']):.1f}% win rate vs random "
                    f"after {int(latest['cumulative_traversals']):,} traversals",
                    file=self.output,
                )

        for stage_index in range(self.stage_index, len(self.config.curriculum.rounds)):
            if self._stop_training:
                break
            if stage_index != self.stage_index:
                self._reset_stage(stage_index, transfer_weights=True)
            assert self.stage_configuration is not None
            round_count = self.stage_configuration.round_count
            print(
                f"Deep CFR stage {stage_index + 1}/{len(self.config.curriculum.rounds)}: "
                f"{round_count} round(s)",
                file=self.output,
            )
            self._epoch_started_at = time.perf_counter()
            while self._stage_should_continue(stage_index):
                started = time.perf_counter()
                self.stage_iteration += 1
                self.global_iteration += 1
                traversals: list[list[_TraversalTelemetry]] = [[], []]
                regret_iteration = RegretIterationCapture(
                    stage_index=stage_index,
                    round_count=round_count,
                    stage_iteration=self.stage_iteration,
                    global_iteration=self.global_iteration,
                    encoder_schema_version=ENCODING_SCHEMA_VERSION,
                    action_count=self.config.network.action_count,
                )
                advantage_optimization: list[_OptimizationTelemetry | None] = []
                for player in range(2):
                    for traversal_ordinal in range(
                        self.config.traversals_per_player
                    ):
                        traversal = self._traverse(player, traversal_ordinal)
                        traversals[player].append(traversal)
                        regret_iteration.add_traversal(
                            traversal.signed_regret_record
                        )
                    advantage_optimization.append(self._learn_advantages(player))
                strategy_optimization = self._learn_strategy()
                self.regret_archive.commit(regret_iteration.finish())
                root_values = [
                    traversal.root_value
                    for player_traversals in traversals
                    for traversal in player_traversals
                ]
                regret_targets = [
                    value
                    for player_traversals in traversals
                    for traversal in player_traversals
                    for value in traversal.absolute_regret_targets
                ]
                log_importance_weights = [
                    value
                    for player_traversals in traversals
                    for traversal in player_traversals
                    for value in traversal.strategy_log_importance_weights
                ]
                metric = {
                    "kind": "training_iteration",
                    "rounds": round_count,
                    "stage_index": stage_index,
                    "stage_iteration": self.stage_iteration,
                    "global_iteration": self.global_iteration,
                    "advantage_loss_player_0": (
                        None
                        if advantage_optimization[0] is None
                        else advantage_optimization[0].loss
                    ),
                    "advantage_loss_player_1": (
                        None
                        if advantage_optimization[1] is None
                        else advantage_optimization[1].loss
                    ),
                    "advantage_gradient_norm_player_0": (
                        None
                        if advantage_optimization[0] is None
                        else advantage_optimization[0].maximum_gradient_norm
                    ),
                    "advantage_gradient_norm_player_1": (
                        None
                        if advantage_optimization[1] is None
                        else advantage_optimization[1].maximum_gradient_norm
                    ),
                    "strategy_loss": (
                        None
                        if strategy_optimization is None
                        else strategy_optimization.loss
                    ),
                    "strategy_gradient_norm": (
                        None
                        if strategy_optimization is None
                        else strategy_optimization.maximum_gradient_norm
                    ),
                    "sampled_root_mean_player_0": _finite_mean(
                        [item.root_value for item in traversals[0]],
                        "player 0 sampled roots",
                    ),
                    "sampled_root_mean_player_1": _finite_mean(
                        [item.root_value for item in traversals[1]],
                        "player 1 sampled roots",
                    ),
                    "sampled_root_mean_absolute": _finite_mean(
                        [abs(value) for value in root_values],
                        "absolute sampled roots",
                    ),
                    "advantage_memory_player_0": len(self.advantage_memories[0]),
                    "advantage_memory_player_1": len(self.advantage_memories[1]),
                    "strategy_memory": len(self.strategy_memory or ()),
                    "elapsed_seconds": time.perf_counter() - started,
                    **_absolute_distribution(
                        regret_targets,
                        "absolute_regret_target",
                    ),
                    **_absolute_distribution(
                        log_importance_weights,
                        "strategy_log_importance_weight",
                    ),
                }
                self._append_metric(metric)
                iteration_budget = self._stage_iteration_budget(stage_index)
                self._render_training_progress(
                    self.stage_iteration,
                    iteration_budget,
                )
                due_until_eval = self._until_win_rate_evaluation_due()
                if (
                    self.stage_iteration % self.config.checkpoint_every == 0
                    or due_until_eval
                ):
                    traversal_count = self._cumulative_traversal_count(stage_index)
                    final_checkpoint, final_policy = self.save_checkpoint(
                        preserve_intermediate=due_until_eval,
                        cumulative_traversal_count=(
                            traversal_count if due_until_eval else None
                        ),
                    )
                self._evaluate_learning_curve_if_due(stage_index, round_count)
                if self._stop_training:
                    break

            self._finish_training_progress()

            if self.config.until_win_rate_enabled:
                if (
                    not self._until_target_reached
                    and self.stage_iteration > 0
                    and not self.learning_curve.contains(
                        stage_index, self.stage_iteration
                    )
                ):
                    traversal_count = self._cumulative_traversal_count(stage_index)
                    final_checkpoint, final_policy = self.save_checkpoint(
                        preserve_intermediate=True,
                        cumulative_traversal_count=traversal_count,
                    )
                    self._evaluate_learning_curve_if_due(
                        stage_index,
                        round_count,
                        force=True,
                    )
                final_checkpoint, final_policy = self.save_checkpoint()
                if round_count not in completed:
                    completed.append(round_count)
                if (
                    self._stop_training
                    or stage_index == len(self.config.curriculum.rounds) - 1
                ):
                    break
                continue

            if not self.stage_evaluated:
                assert self.strategy_network is not None
                from .evaluation import evaluate_policy

                evaluation = evaluate_policy(
                    self.stage_configuration.configured_game,
                    DeepCFRPolicy(
                        self.strategy_network,
                        device=self.device,
                        metadata=self._metadata(),
                    ),
                    pairs=self.config.evaluation_pairs,
                    seed=self.config.seed + round_count * 100_000,
                )
                self.stage_evaluated = True
                self._append_metric(
                    {
                        "kind": "evaluation",
                        "rounds": round_count,
                        "stage_index": stage_index,
                        **evaluation,
                    }
                )
            final_checkpoint, final_policy = self.save_checkpoint()
            completed.append(round_count)

        plot_path = self._write_learning_curve_plot()
        if plot_path is not None:
            print(f"Learning curve: {plot_path}", file=self.output)

        cumulative = (
            self.global_iteration * self.config.traversals_per_iteration()
            if self.global_iteration > 0
            else 0
        )
        if (
            self.config.until_win_rate_enabled
            and not self._until_target_reached
            and self._last_evaluation_win_rate is not None
        ):
            print(
                "MAX ITERATIONS REACHED: "
                f"{100.0 * self._last_evaluation_win_rate:.1f}% win rate vs random "
                f"after {self.global_iteration:,} iterations "
                f"({cumulative:,} traversals; "
                f"target {100.0 * float(self.config.until_win_rate):.1f}%)",
                file=self.output,
            )

        return TrainingResult(
            output_dir=self.config.output_dir,
            completed_rounds=tuple(completed),
            final_checkpoint=final_checkpoint,
            final_policy=final_policy,
            metrics=tuple(self.metrics),
            target_reached=self._until_target_reached,
            final_win_rate=self._last_evaluation_win_rate,
            cumulative_traversals=cumulative,
        )

    def _evaluate_learning_curve_if_due(
        self,
        stage_index: int,
        round_count: int,
        *,
        force: bool = False,
    ) -> None:
        """Pause briefly for a frozen learning-curve evaluation when scheduled."""

        if not force and not self._learning_curve_evaluation_due(stage_index):
            return
        if self.learning_curve.contains(stage_index, self.stage_iteration):
            return
        assert self.stage_configuration is not None
        assert self.strategy_network is not None

        # Wall time for the training block since the last evaluation (or stage start).
        epoch_seconds = time.perf_counter() - self._epoch_started_at

        trainer_rng_state = self.rng.getstate()
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        mps_rng_state = (
            torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
        )
        network_modes = [
            network.training
            for network in (*self.advantage_networks, self.strategy_network)
        ]

        cumulative = self._cumulative_traversal_count(stage_index)
        if self.config.until_win_rate_enabled:
            evaluation_seed = checkpoint_evaluation_seed(
                self.config.seed,
                stage_index=stage_index,
                stage_iteration=self.stage_iteration,
            )
        else:
            evaluation_seed = stage_evaluation_seed(self.config.seed, stage_index)

        try:
            for network in (*self.advantage_networks, self.strategy_network):
                network.eval()
            record = evaluate_learning_curve_checkpoint(
                self.stage_configuration.configured_game,
                DeepCFRPolicy(
                    self.strategy_network,
                    device=self.device,
                    metadata=self._metadata(),
                ),
                pairs=self.config.learning_curve_evaluation_pairs,
                evaluation_seed=evaluation_seed,
                bootstrap_resamples=self.config.learning_curve_bootstrap_resamples,
                bootstrap_rng_seed=bootstrap_seed(
                    self.config.seed,
                    stage_index=stage_index,
                    stage_iteration=self.stage_iteration,
                ),
                round_horizon=round_count,
                stage_index=stage_index,
                stage_iteration=self.stage_iteration,
                global_iteration=self.global_iteration,
                stage_traversal_count=stage_traversals(
                    self.stage_iteration,
                    self.config.traversals_per_player,
                ),
                cumulative_traversal_count=cumulative,
            )
        finally:
            for network, was_training in zip(
                (*self.advantage_networks, self.strategy_network),
                network_modes,
                strict=True,
            ):
                network.train(was_training)
            self.rng.setstate(trainer_rng_state)
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.set_rng_state(torch_rng_state)
            if cuda_rng_states is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_rng_states)
            if mps_rng_state is not None and torch.backends.mps.is_available():
                torch.mps.set_rng_state(mps_rng_state)

        wrote = self.learning_curve.append(record)
        if wrote:
            self._last_evaluation_win_rate = float(record["win_rate"])
            self._emit_progress_log(
                "  evaluation "
                f"iteration={record['global_iteration']} "
                f"epoch={self._format_duration(epoch_seconds)} "
                f"traversals={record['cumulative_traversals']:,} "
                f"games={record['evaluation_games']} "
                f"wins={record['wins']} losses={record['losses']} "
                f"ties={record['ties']} "
                f"win_rate={100.0 * float(record['win_rate']):.1f}% "
                f"mean_utility={record['mean_utility']:.4f} "
                f"ci95=[{100.0 * float(record['win_rate_ci95_lower']):.1f}%, "
                f"{100.0 * float(record['win_rate_ci95_upper']):.1f}%]",
                restore_progress=True,
            )
            if self.config.until_win_rate_enabled:
                self._append_metric(
                    {
                        "kind": "learning_curve_evaluation",
                        **record,
                    }
                )
                self._maybe_stop_for_until_win_rate(record)
        self._epoch_started_at = time.perf_counter()

    def _learning_curve_evaluation_due(self, stage_index: int) -> bool:
        if self.config.until_win_rate_enabled:
            return self._until_win_rate_evaluation_due()
        return self.stage_iteration in self._learning_curve_schedule

    def _until_win_rate_evaluation_due(self) -> bool:
        if not self.config.until_win_rate_enabled:
            return False
        every = self.config.eval_every_iterations
        assert every is not None
        return self.global_iteration > 0 and self.global_iteration % every == 0

    def _stage_iteration_budget(self, stage_index: int) -> int:
        if not self.config.until_win_rate_enabled:
            return int(self.config.iterations_per_stage)
        if stage_index < len(self.config.curriculum.rounds) - 1:
            return int(self.config.iterations_per_stage)
        assert self.config.max_iterations is not None
        remaining = max(0, int(self.config.max_iterations) - self.global_iteration)
        return int(self.stage_iteration + remaining)

    def _render_training_progress(
        self,
        completed: int,
        total: int | str,
    ) -> None:
        """Overwrite a single progress line while training between evaluations."""

        try:
            total_count = max(1, int(total))
        except (TypeError, ValueError):
            total_count = max(1, int(completed))
        completed_count = max(0, min(int(completed), total_count))
        self._progress_completed = completed_count
        self._progress_total = total_count
        fraction = completed_count / total_count
        width = 28
        filled = int(round(fraction * width))
        bar = "#" * filled + "-" * (width - filled)
        line = (
            f"  training [{bar}] "
            f"{100.0 * fraction:5.1f}%  {completed_count}/{total_count}"
        )
        if self._progress_is_tty:
            # Keep the bar on one bottom line; clear any leftover width.
            self.output.write("\r" + line + "\033[K")
            self.output.flush()
            self._progress_line_open = True
            return
        # Non-TTY logs stay sparse: mark every 10% boundary and the finish.
        previous = getattr(self, "_progress_last_bucket", -1)
        bucket = int(fraction * 10)
        if completed_count >= total_count or bucket != previous:
            print(line, file=self.output)
            self._progress_last_bucket = bucket
            self._progress_line_open = False

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format a wall-clock duration as compact seconds/minutes/hours."""

        total = max(0.0, float(seconds))
        if total < 60.0:
            if total < 10.0:
                return f"{total:.1f}s"
            return f"{total:.0f}s"
        whole = int(round(total))
        minutes, secs = divmod(whole, 60)
        if minutes < 60:
            return f"{minutes}m {secs:02d}s" if secs else f"{minutes}m"
        hours, minutes = divmod(minutes, 60)
        if minutes:
            return f"{hours}h {minutes}m"
        return f"{hours}h"

    def _clear_training_progress_line(self) -> None:
        """Erase an in-place progress line without committing it to the log."""

        if self._progress_is_tty and self._progress_line_open:
            self.output.write("\r\033[K")
            self.output.flush()
            self._progress_line_open = False

    def _emit_progress_log(
        self,
        message: str,
        *,
        restore_progress: bool,
    ) -> None:
        """Print a permanent line; optionally redraw the progress bar below it."""

        if self._progress_is_tty and self._progress_line_open:
            self._clear_training_progress_line()
        elif self._progress_line_open:
            self._finish_training_progress()
        print(message, file=self.output)
        if restore_progress and self._progress_is_tty:
            self._render_training_progress(
                self._progress_completed,
                self._progress_total,
            )

    def _finish_training_progress(self) -> None:
        if self._progress_line_open:
            self.output.write("\n")
            self.output.flush()
            self._progress_line_open = False
        self._progress_last_bucket = -1

    def _stage_should_continue(self, stage_index: int) -> bool:
        if self._stop_training:
            return False
        if not self.config.until_win_rate_enabled:
            return self.stage_iteration < self.config.iterations_per_stage
        if stage_index < len(self.config.curriculum.rounds) - 1:
            return self.stage_iteration < self.config.iterations_per_stage
        assert self.config.max_iterations is not None
        return (self.global_iteration + 1) <= int(self.config.max_iterations)

    def _cumulative_traversal_count(self, stage_index: int) -> int:
        if self.config.until_win_rate_enabled:
            return self.global_iteration * self.config.traversals_per_iteration()
        return cumulative_traversals(
            stage_index=stage_index,
            stage_iteration=self.stage_iteration,
            iterations_per_stage=self.config.iterations_per_stage,
            traversals_per_player=self.config.traversals_per_player,
        )

    def _maybe_stop_for_until_win_rate(self, record: dict[str, Any]) -> None:
        assert self.config.until_win_rate is not None
        assert self.config.max_iterations is not None
        win_rate = float(record["win_rate"])
        traversals = int(record["cumulative_traversals"])
        iteration = int(record["global_iteration"])
        required = int(self.config.until_win_rate_consecutive)
        streak = self.learning_curve.consecutive_win_rate_streak(
            float(self.config.until_win_rate)
        )
        if streak >= required:
            self._until_target_reached = True
            self._stop_training = True
            self._emit_progress_log(
                "TARGET REACHED: "
                f"{100.0 * win_rate:.1f}% win rate vs random "
                f"after {iteration:,} iterations ({traversals:,} traversals)",
                restore_progress=False,
            )
            return
        if iteration >= int(self.config.max_iterations):
            self._stop_training = True

    def _write_learning_curve_plot(self) -> Path | None:
        if not self.learning_curve.checkpoints:
            return None
        destination = (
            self.config.output_dir / "analysis" / LEARNING_CURVE_PLOT_NAME
        )
        try:
            return plot_learning_curve(self.learning_curve.json_path, destination)
        except ImportError:
            return None


__all__ = [
    "AdvantageSample",
    "CHECKPOINT_SCHEMA_VERSION",
    "DeepCFRTrainer",
    "LEGACY_CHECKPOINT_SCHEMA_VERSION",
    "StrategySample",
    "TrainingResult",
]
