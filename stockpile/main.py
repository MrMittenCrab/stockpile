"""Command-line entry point for the Stockpile platform.

Parsing is intentionally isolated from configuration resolution. Rules,
complexity, and solve share the same resolved configuration; browser play
lazily launches its local API and Vite workstation. Importing this module has
no terminal side effects.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
from pathlib import Path
import sys
from collections.abc import Sequence
from typing import TextIO

from . import stockpile_interface as interface


_MODES = tuple(mode.value for mode in interface.ConfigurationMode)
_OPTION_FLAGS = (
    ("impact", "Market Impact cards and Action phase"),
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


def _unit_interval_float(name: str):
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{name} must be a number") from error
        if not math.isfinite(parsed) or not 0.0 < parsed <= 1.0:
            raise argparse.ArgumentTypeError(f"{name} must be in (0, 1]")
        return parsed

    return parse


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


def _confidence_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("confidence must be a number") from error
    if not math.isfinite(parsed) or not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("confidence must be in (0, 1)")
    return parsed


def _loopback_host(value: str) -> str:
    """Accept only loopback bind addresses for the ephemeral browser server."""

    if value.casefold() == "localhost":
        return "localhost"
    try:
        address = ipaddress.ip_address(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "host must be localhost or a loopback IP address"
        ) from error
    if not address.is_loopback:
        raise argparse.ArgumentTypeError(
            "host must be localhost or a loopback IP address"
        )
    return value


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
        description="Configure, analyze, train, or locally play Stockpile games.",
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
    play_parser = commands.add_parser(
        "play",
        help="launch the local Stockpile Trainer",
        description=(
            "Launch the local-only Stockpile Lite API and browser interface. "
            "Games are configured in the browser."
        ),
    )
    play_parser.add_argument("--mode", default="lite", choices=("lite",))
    play_parser.add_argument(
        "--host",
        type=_loopback_host,
        default="127.0.0.1",
        help="loopback interface to bind (default: 127.0.0.1)",
    )
    play_parser.add_argument(
        "--port",
        type=_bounded_integer("port", 1, 65_535),
        default=8_000,
        metavar="INT",
        help="local API port (default: 8000)",
    )
    play_parser.add_argument(
        "--policy",
        default=None,
        metavar="PATH",
        help=(
            "Deep CFR policy.pt for the computer seat, or 'random' for the "
            "uniform placeholder"
        ),
    )
    play_parser.add_argument(
        "--run",
        type=_named_positive_integer("run"),
        default=None,
        metavar="INT",
        help="managed Deep CFR run whose round_01 policy.pt the computer should use",
    )
    analyze_parser = commands.add_parser(
        "analyze",
        help="analyze a saved Deep CFR run",
        description=(
            "Report stored policy evaluation, learning-curve history, or sampled "
            "regret for one Deep CFR run. Select the run either by its output "
            "directory or by mode and run number."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analyze_parser.add_argument(
        "--method",
        choices=("evaluation", "regret", "learning-curve"),
        default="evaluation",
        help="analysis method",
    )
    analysis_source = analyze_parser.add_argument_group("source")
    analysis_source.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="run artifact directory",
    )
    analysis_source.add_argument(
        "--mode",
        choices=_MODES,
        default=None,
        help="saved run mode (requires --run)",
    )
    analysis_source.add_argument(
        "--run",
        type=_named_positive_integer("run"),
        default=None,
        metavar="INT",
        help="numbered saved run (requires --mode)",
    )
    analyze_parser.add_argument(
        "--confidence",
        type=_confidence_float,
        default=argparse.SUPPRESS,
        metavar="FLOAT",
        help="regret confidence level (regret default: 0.90)",
    )
    analyze_parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "learning-curve plot destination "
            "(default: <run>/analysis/learning_curve.png)"
        ),
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
        "--until-win-rate",
        type=_unit_interval_float("until-win-rate"),
        default=None,
        metavar="RATE",
        help=(
            "keep training past a normal solve until this win rate vs random "
            "is reached in two consecutive evaluations"
        ),
    )
    training.add_argument(
        "--eval-every",
        type=_named_positive_integer("eval-every"),
        default=None,
        metavar="ITERATIONS",
        help=(
            "with --until-win-rate, train this many iterations between "
            "checkpointed evaluations (default: 100)"
        ),
    )
    training.add_argument(
        "--eval-games",
        type=_named_positive_integer("eval-games"),
        default=None,
        metavar="INT",
        help=(
            "with --until-win-rate, seat-balanced evaluation games per "
            "checkpoint (default: 2000; must be even)"
        ),
    )
    training.add_argument(
        "--max-iterations",
        type=_named_positive_integer("max-iterations"),
        default=None,
        metavar="INT",
        help=(
            "with --until-win-rate, stop after this many cumulative training "
            "iterations if the target has not been reached (default: 10000)"
        ),
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
    artifacts = solve_parser.add_argument_group("artifacts")
    artifact_destination = artifacts.add_mutually_exclusive_group()
    artifact_destination.add_argument(
        "--output-dir",
        type=Path,
        default=argparse.SUPPRESS,
        metavar="PATH",
        help=(
            "explicit unmanaged artifact directory (otherwise a numbered "
            "managed run is reserved)"
        ),
    )
    artifact_destination.add_argument(
        "--run",
        type=_named_positive_integer("run"),
        default=None,
        metavar="INT",
        help="reserve or resume a specific numbered managed run",
    )
    artifacts.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="CHECKPOINT",
        help="resume a same-stage full.pt checkpoint exactly",
    )
    artifacts.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "run the fixed, small one-round preset; other compute/device flags "
            "are ignored (seed, destination, resume, and overwrite still apply)"
        ),
    )
    artifacts.add_argument(
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
        impact=_optional_value(arguments.impact),
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


def play(
    *,
    mode: str,
    host: str,
    port: int,
    output: TextIO,
    policy: str | None = None,
    run: int | None = None,
) -> int:
    """Launch the optional local workstation without importing it elsewhere."""

    if mode != interface.ConfigurationMode.LITE.value:
        raise ValueError("the browser interface supports only Stockpile Lite")
    if policy is not None and run is not None:
        raise ValueError("play accepts only one of --policy and --run")
    from .web import run_trainer
    from .web.policy import RANDOM_POLICY_TOKEN, resolve_computer_policy_path

    if policy is not None and policy.strip().casefold() == RANDOM_POLICY_TOKEN:
        computer_policy = RANDOM_POLICY_TOKEN
    elif policy is not None:
        computer_policy = str(resolve_computer_policy_path(policy=policy))
    elif run is not None:
        computer_policy = str(
            resolve_computer_policy_path(mode=mode, run=run)
        )
    else:
        computer_policy = None
    return int(
        run_trainer(
            host=host,
            port=port,
            output=output,
            computer_policy=computer_policy,
        )
    )


def _analysis_source(arguments: argparse.Namespace) -> Path:
    """Resolve exactly one explicit analysis source without loading Torch."""

    output_dir = arguments.output_dir
    has_output = output_dir is not None
    has_mode = arguments.mode is not None
    has_run = arguments.run is not None
    if has_output:
        if has_mode or has_run:
            raise ValueError(
                "analyze requires exactly --output-dir or the pair --mode and --run"
            )
    elif not has_mode and not has_run:
        raise ValueError(
            "analyze requires exactly --output-dir or the pair --mode and --run"
        )
    elif has_mode != has_run:
        raise ValueError("--mode and --run must be provided together for analyze")
    if has_output:
        path = Path(output_dir).expanduser().resolve(strict=False)
        if not path.is_dir():
            raise FileNotFoundError(f"run output directory does not exist: {path}")
        return path

    from .training.artifacts import resolve_run

    return resolve_run(
        arguments.mode,
        run=arguments.run,
        smoke=False,
    ).path


def _analysis_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _analysis_integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return int(value)


def _trimmed_decimal(
    value: float,
    *,
    signed: bool = False,
    places: int = 3,
) -> str:
    parsed = _analysis_float(value, label="analysis value")
    if parsed == 0.0:
        return "0"
    absolute = abs(parsed)
    rendered = f"{absolute:.{places}f}".rstrip("0").rstrip(".")
    if not rendered or float(rendered) == 0.0:
        rendered = format(absolute, f".{places}g")
    if parsed < 0.0:
        return "-" + rendered
    return ("+" if signed else "") + rendered


def _percentage(value: object) -> str:
    parsed = _analysis_float(value, label="win rate")
    if not 0.0 <= parsed <= 1.0:
        raise ValueError("win rate must be between zero and one")
    return _trimmed_decimal(parsed * 100.0, places=2) + "%"


def _signed_interval(value: object, *, label: str) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain two endpoints")
    lower = _analysis_float(value[0], label=f"{label} lower endpoint")
    upper = _analysis_float(value[1], label=f"{label} upper endpoint")
    if lower > upper:
        raise ValueError(f"{label} endpoints are reversed")
    return (
        f"{_trimmed_decimal(lower, signed=True)} to "
        f"{_trimmed_decimal(upper, signed=True)}"
    )


def _scientific(value: object, *, label: str) -> str:
    if value is None:
        return "N/A"
    rendered = format(_analysis_float(value, label=label), ".2e")
    mantissa, exponent = rendered.split("e", maxsplit=1)
    return f"{mantissa}e{int(exponent)}"


def _scientific_interval(value: object, *, label: str) -> str:
    if value is None:
        return "N/A"
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain two endpoints")
    lower = _analysis_float(value[0], label=f"{label} lower endpoint")
    upper = _analysis_float(value[1], label=f"{label} upper endpoint")
    if lower > upper:
        raise ValueError(f"{label} endpoints are reversed")
    return (
        f"{_scientific(lower, label=label)} to "
        f"{_scientific(upper, label=label)}"
    )


def _print_analysis_table(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    output: TextIO,
) -> None:
    print(title, file=output)
    print(file=output)
    if not rows:
        print("N/A", file=output)
        return
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print(
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        file=output,
    )
    for row in rows:
        print(
            "  ".join(value.rjust(widths[index]) for index, value in enumerate(row)),
            file=output,
        )


def _stored_evaluation_rows(run_dir: Path) -> list[tuple[str, ...]]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    if not metrics_path.is_file():
        raise OSError(f"evaluation metrics path is not a file: {metrics_path}")

    stages: dict[int, tuple[str, ...]] = {}
    with metrics_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid evaluation metrics JSON on line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"evaluation metrics line {line_number} must be an object"
                )
            if record.get("kind") != "evaluation":
                continue

            stage_index = _analysis_integer(
                record.get("stage_index"),
                label=f"evaluation stage on line {line_number}",
            )
            rounds = _analysis_integer(
                record.get("rounds"),
                label=f"evaluation rounds on line {line_number}",
                minimum=1,
            )
            utility = _analysis_float(
                record.get("trained_seat_mean_utility"),
                label=f"evaluation mean utility on line {line_number}",
            )
            margin = _analysis_float(
                record.get("mean_final_cash_differential"),
                label=f"evaluation cash margin on line {line_number}",
            )
            row = (
                str(rounds),
                _percentage(record.get("win_rate")),
                _trimmed_decimal(utility, signed=True),
                _signed_interval(
                    record.get("trained_seat_utility_ci95"),
                    label=f"evaluation CL on line {line_number}",
                ),
                _trimmed_decimal(margin, signed=True) + "K",
            )
            previous = stages.get(stage_index)
            if previous is not None and previous[0] != row[0]:
                raise ValueError(
                    "evaluation stage "
                    f"{stage_index} has conflicting rounds on line {line_number}"
                )
            stages[stage_index] = row
    return [stages[index] for index in sorted(stages)]


class _RegretProgressRenderer:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        try:
            self.is_tty = bool(stream.isatty())
        except (AttributeError, OSError):
            self.is_tty = False
        self.stage_index: int | None = None
        self.last_completed = 0
        self.line_open = False
        self.rendered = False

    def __call__(self, progress: object) -> None:
        stage_index = int(getattr(progress, "stage_index"))
        rounds = int(getattr(progress, "round_count"))
        completed = int(getattr(progress, "completed_replicates"))
        total = int(getattr(progress, "total_replicates"))
        if self.is_tty:
            self.stream.write(
                "\rAnalyzing sampled regret: "
                f"rounds {rounds}, bootstrap {completed}/{total}"
            )
            self.stream.flush()
            self.rendered = True
            return

        if stage_index != self.stage_index:
            if self.line_open:
                self.stream.write("\n")
            self.stream.write(f"Analyzing sampled regret: rounds {rounds} ")
            self.stage_index = stage_index
            self.last_completed = 0
            self.line_open = True
            self.rendered = True
        if completed > self.last_completed:
            self.stream.write(".")
            self.last_completed = completed
        if self.line_open and completed >= total:
            self.stream.write(" done\n")
            self.line_open = False
        self.stream.flush()

    def finish(self) -> None:
        if self.is_tty and self.rendered:
            self.stream.write("\n")
        elif self.line_open:
            self.stream.write("\n")
            self.line_open = False
        if self.rendered:
            self.stream.flush()


def _analyze_evaluation(run_dir: Path, *, output: TextIO) -> int:
    _print_analysis_table(
        "Evaluation vs random",
        ("Rounds", "Win rate", "Mean utility", "CL", "Cash margin"),
        _stored_evaluation_rows(run_dir),
        output=output,
    )
    return 0


def _analyze_regret(
    run_dir: Path,
    *,
    confidence: float,
    output: TextIO,
) -> int:
    from .training.regret import analyze_run, write_analysis_report

    progress = _RegretProgressRenderer(sys.stderr)
    try:
        report = analyze_run(
            run_dir,
            confidence=confidence,
            bootstrap_replicates=10_000,
            seed=0,
            progress=progress,
        )
    finally:
        progress.finish()

    availability = report.get("availability", {})
    stages = report.get("stages", ())
    statistics_available = bool(availability.get("available", False))
    telemetry_declared = bool(availability.get("telemetry_declared", False))
    if statistics_available or telemetry_declared:
        write_analysis_report(run_dir, report)
    rows = []
    if statistics_available:
        for stage in stages:
            point = stage.get("point") or {}
            intervals = stage.get("confidence_interval") or {}
            rows.append(
                (
                    str(
                        _analysis_integer(
                            stage.get("round_count"),
                            label="regret rounds",
                            minimum=1,
                        )
                    ),
                    _scientific(
                        point.get("maximum"),
                        label="final regret",
                    ),
                    _scientific_interval(
                        intervals.get("maximum"),
                        label="regret CL",
                    ),
                )
            )
    _print_analysis_table(
        "Sampled regret",
        ("Rounds", "Final regret", "CL"),
        rows,
        output=output,
    )
    return 0


def _analyze_learning_curve(
    run_dir: Path,
    *,
    plot: Path | None,
    output: TextIO,
) -> int:
    from .training.learning_curve import (
        LEARNING_CURVE_PLOT_NAME,
        load_learning_curve_history,
        plot_learning_curve,
    )

    history = load_learning_curve_history(run_dir)
    checkpoints = list(history["checkpoints"])
    checkpoints.sort(key=lambda item: int(item["cumulative_traversals"]))
    print("Deep CFR learning curve vs random", file=output)
    print(
        "traversals\twin_rate\tci95_lower\tci95_upper\twins\tlosses\tties",
        file=output,
    )
    for checkpoint in checkpoints:
        win_rate = checkpoint.get("win_rate", checkpoint["score"])
        ci_low = checkpoint.get(
            "win_rate_ci95_lower", checkpoint["score_ci95_lower"]
        )
        ci_high = checkpoint.get(
            "win_rate_ci95_upper", checkpoint["score_ci95_upper"]
        )
        print(
            f"{checkpoint['cumulative_traversals']}\t"
            f"{float(win_rate):.6f}\t"
            f"{float(ci_low):.6f}\t"
            f"{float(ci_high):.6f}\t"
            f"{checkpoint['wins']}\t"
            f"{checkpoint['losses']}\t"
            f"{checkpoint['ties']}",
            file=output,
        )
    destination = plot or (run_dir / "analysis" / LEARNING_CURVE_PLOT_NAME)
    plot_path = plot_learning_curve(history, destination)
    print(f"Plot: {plot_path}", file=output)
    return 0


def analyze(*, arguments: argparse.Namespace, output: TextIO) -> int:
    """Report the selected stored analysis for one run."""

    method = str(arguments.method)
    explicit_confidence = getattr(arguments, "confidence", None)
    if method != "regret" and explicit_confidence is not None:
        raise ValueError("--confidence requires --method regret")
    if method != "learning-curve" and getattr(arguments, "plot", None) is not None:
        raise ValueError("--plot requires --method learning-curve")
    run_dir = _analysis_source(arguments)
    if method == "evaluation":
        return _analyze_evaluation(run_dir, output=output)
    if method == "regret":
        confidence = 0.90 if explicit_confidence is None else explicit_confidence
        return _analyze_regret(
            run_dir,
            confidence=_analysis_float(confidence, label="confidence"),
            output=output,
        )
    if method == "learning-curve":
        return _analyze_learning_curve(
            run_dir,
            plot=arguments.plot,
            output=output,
        )
    raise ValueError(f"unknown analyze method: {method}")


def solve(
    configuration: interface.GameConfig,
    *,
    arguments: argparse.Namespace,
    output: TextIO,
) -> int:
    """Train the supported Lite game without importing Torch for other commands."""

    from .training.artifacts import (
        plan_resume_destination,
        resolve_fresh_output,
        update_run_manifest,
    )
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
    if arguments.smoke:
        if configuration.round_count != 1:
            raise ValueError("--smoke requires --rounds 1")
        if arguments.curriculum is not None:
            raise ValueError("--smoke uses its fixed one-round curriculum")
        if arguments.until_win_rate is not None:
            raise ValueError("--smoke cannot be combined with --until-win-rate")
        curriculum = None
    else:
        # Validate the schedule before reserving a numbered destination.
        curriculum = CurriculumConfig.for_target(
            configuration.round_count,
            arguments.curriculum,
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

    requested_output = getattr(arguments, "output_dir", None)
    mode_name = configuration.rule_set.profile
    if (
        arguments.resume is not None
        and arguments.overwrite
        and requested_output is None
    ):
        raise ValueError(
            "--overwrite on resume requires a differing unmanaged --output-dir"
        )
    if arguments.resume is None:
        run_ref = resolve_fresh_output(
            mode_name,
            output_dir=requested_output,
            run=arguments.run,
            smoke=arguments.smoke,
            overwrite=arguments.overwrite,
        )
        resume_checkpoint = None
    else:
        resume_plan = plan_resume_destination(
            arguments.resume,
            mode=mode_name,
            output_dir=requested_output,
            run=arguments.run,
            smoke=True if arguments.smoke else None,
            overwrite=arguments.overwrite,
        )
        run_ref = resume_plan.destination
        resume_checkpoint = resume_plan.checkpoint

    if run_ref.smoke:
        if configuration.round_count != 1:
            raise ValueError("--smoke requires --rounds 1")
        if arguments.curriculum is not None:
            raise ValueError("--smoke uses its fixed one-round curriculum")
        training_config = DeepCFRConfig.smoke(
            output_dir=run_ref.output_dir,
            seed=arguments.seed,
        )
    else:
        assert curriculum is not None
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
            until_win_rate=arguments.until_win_rate,
            eval_every_iterations=arguments.eval_every,
            eval_games=arguments.eval_games,
            max_iterations=arguments.max_iterations,
            seed=arguments.seed,
            device=arguments.device,
            output_dir=run_ref.output_dir,
        )

    print(
        "Solver: outcome-sampled strict-history Deep CFR "
        "(sampled evaluation; no exact exploitability claim)",
        file=output,
    )
    if training_config.until_win_rate_enabled:
        print(
            "Until win rate: "
            f"{100.0 * float(training_config.until_win_rate):.1f}% vs random "
            f"(eval every {training_config.eval_every_iterations:,} iterations, "
            f"{training_config.eval_games:,} games, "
            f"max {training_config.max_iterations:,} iterations, "
            f"{training_config.until_win_rate_consecutive} consecutive hits)",
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
    if resume_checkpoint is not None:
        # Validate and restore before advancing the destination lifecycle.  A
        # corrupt or incompatible checkpoint therefore leaves a newly
        # reserved fork visibly unstarted rather than incorrectly active.
        trainer.load_checkpoint(resume_checkpoint)
    if run_ref.managed and run_ref.state != "completed":
        run_ref = update_run_manifest(run_ref, state="active")
    result = trainer.train(
        resume=resume_checkpoint,
        overwrite=arguments.overwrite if resume_checkpoint is None else False,
    )
    if run_ref.managed:
        update_run_manifest(run_ref, state="completed")
    print(f"Completed rounds: {','.join(map(str, result.completed_rounds))}", file=output)
    if training_config.until_win_rate_enabled:
        if result.final_win_rate is not None:
            print(
                f"Final win rate vs random: {100.0 * result.final_win_rate:.1f}% "
                f"after {result.cumulative_traversals:,} traversals"
                + (" (target reached)" if result.target_reached else ""),
                file=output,
            )
        history_csv = training_config.output_dir / "evaluation_history.csv"
        if history_csv.is_file():
            print(f"Evaluation history: {history_csv}", file=output)
    print(f"Full checkpoint: {result.final_checkpoint}", file=output)
    print(f"Inference policy: {result.final_policy}", file=output)
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    output: TextIO | None = None,
) -> int:
    """Parse and dispatch one Stockpile command."""

    parser = build_parser()
    output = sys.stdout if output is None else output
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help(file=output)
        return 0

    if arguments.command == "analyze":
        try:
            return analyze(arguments=arguments, output=output)
        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
            OSError,
        ) as error:
            parser.error(str(error))

    if arguments.command == "play":
        try:
            return play(
                mode=arguments.mode,
                host=arguments.host,
                port=arguments.port,
                output=output,
                policy=arguments.policy,
                run=arguments.run,
            )
        except (ValueError, RuntimeError, OSError) as error:
            parser.error(str(error))

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
