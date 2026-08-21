"""Command-line entry point for the Stockpile platform.

Parsing is intentionally isolated from configuration resolution.  Every
subcommand receives the same resolved :class:`stockpile_interface.GameConfig`.
Importing this module has no terminal side effects.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from collections.abc import Sequence
from typing import TextIO

from . import stockpile_interface as interface


_MODES = tuple(mode.value for mode in interface.ConfigurationMode)
_OPTION_FLAGS = (
    ("hand", "starting hand"),
    ("fees", "trading fees"),
    ("dividend", "dividends"),
    ("split", "stock splits"),
    ("majority", "majority bonus"),
    ("stock_tracks", "stock-dependent price tracks"),
    ("sell_order", "sequential observable selling"),
)


def _bounded_integer(name: str, minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from error
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be between {minimum} and {maximum}"
            )
        return parsed

    return parse


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("max-states must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("max-states must be positive")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("max-seconds must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("max-seconds must be positive and finite")
    return parsed


def _named_positive_integer(name: str):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from error
        if parsed <= 0:
            raise argparse.ArgumentTypeError(f"{name} must be positive")
        return parsed

    return parse


def _named_positive_float(name: str):
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{name} must be a number") from error
        if not math.isfinite(parsed) or parsed <= 0:
            raise argparse.ArgumentTypeError(f"{name} must be positive and finite")
        return parsed

    return parse


def _exploration_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("exploration must be a number") from error
    if not math.isfinite(parsed) or not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("exploration must be in (0, 1]")
    return parsed


def _common_configuration_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mode", required=True, choices=_MODES)
    common.add_argument(
        "--players",
        type=_bounded_integer("players", 2, 5),
        default=2,
        metavar="INT",
    )
    common.add_argument(
        "--rounds",
        type=_bounded_integer("rounds", 1, 10),
        default=6,
        metavar="INT",
    )
    for destination, description in _OPTION_FLAGS:
        common.add_argument(
            "--" + destination.replace("_", "-"),
            choices=("on", "off"),
            default=None,
            metavar="{on,off}",
            help=f"override {description}",
        )
    return common


def build_parser() -> argparse.ArgumentParser:
    """Build the side-effect-free Stockpile argument parser."""

    parser = argparse.ArgumentParser(
        prog="stockpile",
        description="Configure and analyse Stockpile games.",
    )
    commands = parser.add_subparsers(dest="command", title="commands")
    common = _common_configuration_parser()

    commands.add_parser(
        "rules",
        parents=(common,),
        help="show resolved game parameters",
        description="Show resolved Stockpile game parameters.",
    )
    complexity_parser = commands.add_parser(
        "complexity",
        parents=(common,),
        help="calculate information-set complexity",
        description="Calculate complexity for a resolved Stockpile game.",
    )
    complexity_parser.add_argument(
        "--max-states",
        type=_positive_integer,
        default=1_000,
        metavar="INT",
        help="live traversal state budget on a cache miss (default: 1000)",
    )
    complexity_parser.add_argument(
        "--max-seconds",
        type=_positive_float,
        default=3.0,
        metavar="SECONDS",
        help="live traversal time budget on a cache miss (default: 3)",
    )
    complexity_parser.add_argument(
        "--refresh",
        action="store_true",
        help="ignore remembered results and calculate a fresh result",
    )
    commands.add_parser(
        "play",
        parents=(common,),
        help="play a game (not implemented)",
        description="Play a resolved Stockpile game (not implemented).",
    )
    solve_parser = commands.add_parser(
        "solve",
        parents=(common,),
        help="train an outcome-sampled Deep CFR policy",
        description=(
            "Train Stockpile Lite with outcome-sampled, strict-history Deep CFR. "
            "Only the default two-player compact Lite game is supported."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    curriculum = solve_parser.add_argument_group("curriculum")
    curriculum.add_argument(
        "--curriculum",
        default=None,
        metavar="ROUNDS",
        help=(
            "comma-delimited horizons ending at --rounds "
            "(default for six rounds: 1,2,3,4,6)"
        ),
    )
    training = solve_parser.add_argument_group("training")
    training.add_argument(
        "--iterations-per-stage",
        type=_named_positive_integer("iterations-per-stage"),
        default=100,
        metavar="INT",
        help="training iterations at each curriculum horizon",
    )
    training.add_argument(
        "--traversals-per-player",
        type=_named_positive_integer("traversals-per-player"),
        default=20,
        metavar="INT",
        help="sampled trajectories for each update player per iteration",
    )
    training.add_argument(
        "--advantage-train-steps",
        type=_named_positive_integer("advantage-train-steps"),
        default=1,
        metavar="INT",
        help="advantage optimizer steps per player and iteration",
    )
    training.add_argument(
        "--strategy-train-steps",
        type=_named_positive_integer("strategy-train-steps"),
        default=1,
        metavar="INT",
        help="average-strategy optimizer steps per iteration",
    )
    training.add_argument(
        "--batch-size",
        type=_named_positive_integer("batch-size"),
        default=32,
        metavar="INT",
        help="maximum sampled rows per optimizer step",
    )
    training.add_argument(
        "--memory-capacity",
        type=_named_positive_integer("memory-capacity"),
        default=2_000,
        metavar="INT",
        help="capacity of each of three stage-local reservoir memories",
    )
    training.add_argument(
        "--learning-rate",
        type=_named_positive_float("learning-rate"),
        default=1e-4,
        metavar="FLOAT",
        help="Adam learning rate",
    )
    training.add_argument(
        "--exploration",
        type=_exploration_float,
        default=0.6,
        metavar="FLOAT",
        help="updating-player uniform exploration mixture",
    )
    training.add_argument(
        "--gradient-clip",
        type=_named_positive_float("gradient-clip"),
        default=5.0,
        metavar="FLOAT",
        help="maximum gradient norm",
    )
    training.add_argument(
        "--checkpoint-every",
        type=_named_positive_integer("checkpoint-every"),
        default=10,
        metavar="INT",
        help="iterations between full checkpoint replacements",
    )
    training.add_argument(
        "--evaluation-pairs",
        type=_named_positive_integer("evaluation-pairs"),
        default=100,
        metavar="INT",
        help="seat-swapped evaluation seed pairs after each stage",
    )
    training.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INT",
        help="training, reservoir, and evaluation seed",
    )
    training.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="training device (auto selects CUDA, otherwise CPU; MPS is unsupported)",
    )
    training.add_argument(
        "--output-dir",
        type=Path,
        default=argparse.SUPPRESS,
        metavar="PATH",
        help=(
            "artifact directory (default: artifacts/deep_cfr/smoke with "
            "--smoke, otherwise artifacts/deep_cfr/default)"
        ),
    )
    training.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="CHECKPOINT",
        help="resume a same-stage full.pt checkpoint exactly",
    )
    training.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "run the fixed, small one-round preset; other compute/device flags "
            "are ignored (seed, output, resume, and overwrite still apply)"
        ),
    )
    training.add_argument(
        "--overwrite",
        action="store_true",
        help="allow a fresh run to replace artifacts in a nonempty output directory",
    )
    return parser


def _optional_value(value: str | None) -> bool | None:
    if value is None:
        return None
    return value == "on"


def resolve_arguments(arguments: argparse.Namespace) -> interface.GameConfig:
    """Resolve one parsed namespace through the shared configuration facade."""

    return interface.resolve_configuration(
        arguments.mode,
        player_count=arguments.players,
        round_count=arguments.rounds,
        hand=_optional_value(arguments.hand),
        fees=_optional_value(arguments.fees),
        dividend=_optional_value(arguments.dividend),
        split=_optional_value(arguments.split),
        majority=_optional_value(arguments.majority),
        stock_tracks=_optional_value(arguments.stock_tracks),
        sell_order=_optional_value(arguments.sell_order),
        action_space_mode="compact",
    )


def _switch(value: bool) -> str:
    return "on" if value else "off"


def _rules_lines(configuration: interface.GameConfig) -> list[str]:
    rule_set = configuration.rule_set
    starting_cash = (
        "investor-dependent"
        if configuration.investor
        else f"{rule_set.starting_cash}K"
    )
    return [
        f"Mode: {configuration.mode.value}",
        f"Players: {configuration.player_count}",
        f"Rounds: {configuration.round_count}",
        "",
        f"Impact: {_switch(configuration.impact)}",
        f"Investor: {_switch(configuration.investor)}",
        "",
        f"Hand: {_switch(configuration.hand)}",
        f"Fees: {_switch(configuration.fees)}",
        f"Dividend: {_switch(configuration.dividend)}",
        f"Split: {_switch(configuration.split)}",
        f"Majority: {_switch(configuration.majority)}",
        f"Stock tracks: {_switch(configuration.stock_tracks)}",
        f"Sell order: {_switch(configuration.sell_order)}",
        "",
        f"Starting cash: {starting_cash}",
        f"Bidding meeples: {rule_set.meeples_per_player}",
        f"Stockpiles: {rule_set.stockpile_count}",
    ]


def rules(configuration: interface.GameConfig, *, output: TextIO) -> int:
    """Render a resolved configuration without narrating gameplay."""

    print("\n".join(_rules_lines(configuration)), file=output)
    return 0


def _count(value: int, *, exact: bool) -> str:
    prefix = "" if exact else ">="
    return f"{prefix}{value:,}"


def complexity(
    configuration: interface.GameConfig,
    *,
    max_states: int,
    max_seconds: float,
    refresh: bool,
    output: TextIO,
) -> int:
    """Resolve and render the existing bounded traversal result."""

    resolved = interface.resolve_interface_complexity(
        configuration,
        cache_policy="refresh" if refresh else "prefer",
        max_states=max_states,
        max_seconds=max_seconds,
        require_exact=False,
    )
    information = resolved.information_set_complexity
    catalog = resolved.action_catalog
    source = str(resolved.provenance.source).replace("_", " ")
    lines = [
        f"Mode: {configuration.mode.value}",
        f"Players: {configuration.player_count}",
        f"Rounds: {configuration.round_count}",
        "",
        f"Status: {'exact' if information.exact else 'lower bound'}",
        "Information sets: "
        + _count(information.information_sets, exact=information.exact),
        "Information-set actions: "
        + _count(information.information_set_actions, exact=information.exact),
        "Maximum actions per information set: "
        + f"{information.max_actions_per_information_set:,}",
        f"Distinct actions: {catalog.num_distinct_actions:,}",
        f"Maximum legal actions: {catalog.max_legal_actions:,}",
        f"States visited: {information.states_visited:,}",
        f"Source: {source}",
    ]
    print("\n".join(lines), file=output)
    return 0


def play(configuration: interface.GameConfig, *, output: TextIO) -> int:
    del configuration
    print("Not implemented.", file=output)
    return 0


def solve(
    configuration: interface.GameConfig,
    *,
    arguments: argparse.Namespace,
    output: TextIO,
) -> int:
    """Train the supported Lite game without importing Torch for other commands."""

    from .training.config import CurriculumConfig, DeepCFRConfig

    if configuration.mode is not interface.ConfigurationMode.LITE:
        raise ValueError("Deep CFR currently supports only Stockpile Lite")
    if configuration.player_count != 2:
        raise ValueError("Deep CFR currently supports exactly two players")
    if configuration.action_space_mode != "compact":
        raise ValueError("Deep CFR requires the compact 18-action space")
    enabled = [
        name
        for name in (
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
    if arguments.resume is not None and arguments.overwrite:
        raise ValueError("--resume cannot be combined with --overwrite")

    output_dir = getattr(arguments, "output_dir", None) or Path(
        "artifacts/deep_cfr/smoke"
        if arguments.smoke
        else "artifacts/deep_cfr/default"
    )
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path is not a directory: {output_dir}")
    if (
        arguments.resume is None
        and output_dir.exists()
        and any(output_dir.iterdir())
        and not arguments.overwrite
    ):
        raise ValueError(
            f"output directory is not empty: {output_dir}; "
            "choose another directory or pass --overwrite"
        )

    if arguments.smoke:
        if configuration.round_count != 1:
            raise ValueError("--smoke requires --rounds 1")
        if arguments.curriculum is not None:
            raise ValueError("--smoke uses its fixed one-round curriculum")
        training_config = DeepCFRConfig.smoke(
            output_dir=output_dir,
            seed=arguments.seed,
        )
    else:
        curriculum = CurriculumConfig.for_target(
            configuration.round_count,
            arguments.curriculum,
        )
        training_config = DeepCFRConfig(
            curriculum=curriculum,
            iterations_per_stage=arguments.iterations_per_stage,
            traversals_per_player=arguments.traversals_per_player,
            advantage_train_steps=arguments.advantage_train_steps,
            strategy_train_steps=arguments.strategy_train_steps,
            batch_size=arguments.batch_size,
            memory_capacity=arguments.memory_capacity,
            learning_rate=arguments.learning_rate,
            exploration=arguments.exploration,
            gradient_clip=arguments.gradient_clip,
            checkpoint_every=arguments.checkpoint_every,
            evaluation_pairs=arguments.evaluation_pairs,
            seed=arguments.seed,
            device=arguments.device,
            output_dir=output_dir,
        )
    try:
        from .training.trainer import DeepCFRTrainer
    except ModuleNotFoundError as error:
        if error.name == "torch":
            print(
                "Deep CFR requires the optional training dependencies; "
                "install requirements-training.txt.",
                file=output,
            )
            return 2
        raise

    print(
        "Solver: outcome-sampled strict-history Deep CFR "
        "(sampled evaluation; no exact exploitability claim)",
        file=output,
    )
    trainer = DeepCFRTrainer(
        training_config,
        base_configuration=configuration,
        output=output,
    )
    print(
        "Curriculum: "
        + ",".join(map(str, training_config.curriculum.rounds)),
        file=output,
    )
    print(f"Device: {trainer.device}", file=output)
    print(
        f"Batch size: {training_config.batch_size}; reservoir capacity: "
        f"{training_config.memory_capacity} each (3 reservoirs)",
        file=output,
    )
    result = trainer.train(
        resume=arguments.resume,
        overwrite=arguments.overwrite,
    )
    print(f"Completed rounds: {','.join(map(str, result.completed_rounds))}", file=output)
    print(f"Full checkpoint: {result.final_checkpoint}", file=output)
    print(f"Inference policy: {result.final_policy}", file=output)
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    output: TextIO | None = None,
) -> int:
    """Parse, resolve once, and dispatch one non-interactive subcommand."""

    parser = build_parser()
    output = sys.stdout if output is None else output
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help(file=output)
        return 0

    try:
        configuration = resolve_arguments(arguments)
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    if arguments.command == "rules":
        return rules(configuration, output=output)
    if arguments.command == "complexity":
        return complexity(
            configuration,
            max_states=arguments.max_states,
            max_seconds=arguments.max_seconds,
            refresh=arguments.refresh,
            output=output,
        )
    if arguments.command == "play":
        return play(configuration, output=output)
    if arguments.command == "solve":
        try:
            return solve(configuration, arguments=arguments, output=output)
        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
            OSError,
        ) as error:
            parser.error(str(error))
    parser.error(f"unknown command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
