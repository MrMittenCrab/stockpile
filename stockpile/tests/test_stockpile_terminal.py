"""Contracts for the non-interactive ``python -m stockpile`` CLI."""

from __future__ import annotations

from contextlib import chdir, redirect_stderr, redirect_stdout
from io import StringIO
import importlib
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import stockpile


terminal = importlib.import_module("stockpile.main")


def _resolved_complexity(
    configuration: stockpile.GameConfig,
    *,
    exact: bool,
    max_states: int,
    max_seconds: float,
) -> stockpile.ResolvedInterfaceComplexity:
    players = configuration.player_count
    per_player_sets = {player: 0 for player in range(players)}
    per_player_actions = {player: 0 for player in range(players)}
    per_player_sets[0] = 5
    per_player_actions[0] = 14
    per_player_sets[1] = 7
    per_player_actions[1] = 20
    information = stockpile.InformationSetComplexity(
        parameters=configuration.parameters,
        exact=exact,
        count_kind="exact" if exact else "lower_bound",
        information_sets=12,
        information_set_actions=34,
        max_actions_per_information_set=4,
        per_player_information_sets=per_player_sets,
        per_player_information_set_actions=per_player_actions,
        states_visited=91,
        terminal_states=13,
        chance_nodes=17,
        elapsed_seconds=0.25,
        max_states=max_states,
        max_seconds=max_seconds,
        truncation_reason=None if exact else "max_states",
    )
    report = stockpile.complexity_report(configuration.configured_game)
    complexity = stockpile.InterfaceComplexity(
        configuration=configuration,
        action_catalog=stockpile.ActionCatalogComplexity(
            num_distinct_actions=int(report["num_distinct_actions"]),
            max_legal_actions=int(report["max_legal_actions"]),
            max_chance_outcomes=int(report["max_chance_outcomes"]),
            shared_action_head=int(report["shared_action_head"]),
            max_game_length=int(report["max_game_length"]),
            observation_size=int(report["observation_size"]),
        ),
        information_set_complexity=information,
    )
    return stockpile.ResolvedInterfaceComplexity(
        complexity=complexity,
        provenance=stockpile.live_complexity_provenance(),
        bounds=stockpile.compute_complexity_bounds(
            configuration.configured_game,
            information,
        ),
    )


class TerminalTests(unittest.TestCase):
    def test_imports_have_no_terminal_side_effects(self):
        repository = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [sys.executable, "-c", "import stockpile; import stockpile.main"],
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")

    def test_bare_command_prints_help_without_a_menu(self):
        output = StringIO()
        status = terminal.main([], output=output)
        rendered = output.getvalue()
        self.assertEqual(status, 0)
        self.assertIn("usage: stockpile", rendered)
        for command in ("rules", "complexity", "play", "analyze", "solve"):
            self.assertIn(command, rendered)
        self.assertNotIn("Selection", rendered)
        self.assertNotIn("Quit", rendered)
        self.assertNotIn("developer", rendered.lower())

    def test_package_module_bare_command_prints_help(self):
        repository = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [sys.executable, "-m", "stockpile"],
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage: stockpile", completed.stdout)
        self.assertNotIn("Selection", completed.stdout)

    def test_every_command_help_has_common_options_and_no_fixed_override(self):
        for command in ("rules", "complexity", "solve"):
            with self.subTest(command=command):
                output = StringIO()
                with self.assertRaises(SystemExit) as raised, redirect_stdout(output):
                    terminal.main([command, "--help"])
                self.assertEqual(raised.exception.code, 0)
                rendered = output.getvalue()
                for option in (
                    "--mode",
                    "--players",
                    "--rounds",
                    "--impact",
                    "--hand",
                    "--fees",
                    "--dividend",
                    "--split",
                    "--majority",
                    "--stock-tracks",
                    "--sell-order",
                ):
                    self.assertIn(option, rendered)
                self.assertNotIn("--investor", rendered)

        output = StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(output):
            terminal.main(["play", "--help"])
        self.assertEqual(raised.exception.code, 0)
        rendered = output.getvalue()
        self.assertIn("--mode {lite}", rendered)
        self.assertIn("--host", rendered)
        self.assertIn("--port", rendered)
        self.assertNotIn("--players", rendered)
        self.assertNotIn("--rounds", rendered)

    def test_solve_help_explains_safe_memory_and_numbered_runs(self):
        output = StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(output):
            terminal.main(["solve", "--help"])

        self.assertEqual(raised.exception.code, 0)
        rendered = output.getvalue()
        self.assertIn("default: 32", rendered)
        self.assertIn("default: 2000", rendered)
        self.assertIn("three stage-local reservoir", rendered)
        self.assertIn("--run", rendered)
        self.assertIn("numbered managed run", rendered)

    def test_analyze_requires_exactly_one_source_form_and_valid_confidence(self):
        invalid = (
            ["analyze"],
            ["analyze", "--mode", "lite"],
            ["analyze", "--run", "1"],
            [
                "analyze",
                "--output-dir",
                "somewhere",
                "--mode",
                "lite",
                "--run",
                "1",
            ],
            ["analyze", "--mode", "lite", "--run", "0"],
            ["analyze", "--mode", "lite", "--run", "1", "--confidence", "1"],
        )
        for argv in invalid:
            with self.subTest(argv=argv), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    terminal.main(argv, output=StringIO())
                self.assertEqual(raised.exception.code, 2)

    def test_analyze_output_directory_uses_ninety_percent_default_and_legacy_na(self):
        captured = {}

        def analyze_run(path, *, confidence, bootstrap_replicates, seed):
            captured.update(
                path=path,
                confidence=confidence,
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
            )
            return {
                "availability": {"available": False},
                "stages": [],
            }

        def write_analysis_report(path, report):
            captured.update(report_path=path, report=report)
            return path / "analysis" / "sampled_average_regret.json"

        regret_module = ModuleType("stockpile.training.regret")
        regret_module.analyze_run = analyze_run
        regret_module.write_analysis_report = write_analysis_report
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary).resolve()
            output = StringIO()
            with patch.dict(
                sys.modules,
                {"stockpile.training.regret": regret_module},
            ):
                status = terminal.main(
                    ["analyze", "--output-dir", str(run_dir)],
                    output=output,
                )

        self.assertEqual(status, 0)
        self.assertEqual(captured["path"], run_dir)
        self.assertNotIn("report_path", captured)
        self.assertEqual(captured["confidence"], 0.90)
        self.assertEqual(captured["bootstrap_replicates"], 10_000)
        self.assertEqual(captured["seed"], 0)
        self.assertEqual(
            output.getvalue(),
            "training_average_regret = N/A\nconfidence_interval = N/A\n",
        )

    def test_analyze_persists_declared_new_format_before_first_iteration(self):
        captured = {}
        report = {
            "availability": {
                "available": False,
                "telemetry_declared": True,
            },
            "stages": [],
        }
        regret_module = ModuleType("stockpile.training.regret")
        regret_module.analyze_run = lambda *_args, **_kwargs: report

        def write_analysis_report(path, value):
            captured.update(path=path, report=value)
            return path / "analysis" / "sampled_average_regret.json"

        regret_module.write_analysis_report = write_analysis_report
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary).resolve()
            output = StringIO()
            with patch.dict(
                sys.modules,
                {"stockpile.training.regret": regret_module},
            ):
                status = terminal.main(
                    ["analyze", "--output-dir", str(run_dir)],
                    output=output,
                )

        self.assertEqual(status, 0)
        self.assertEqual(captured, {"path": run_dir, "report": report})
        self.assertEqual(
            output.getvalue(),
            "training_average_regret = N/A\nconfidence_interval = N/A\n",
        )

    def test_analyze_mode_run_renders_each_stage_without_equilibrium_claims(self):
        report = {
            "availability": {"available": True},
            "stages": [
                {
                    "stage_index": 2,
                    "round_count": 3,
                    "last_stage_iteration": 7,
                    "point": {
                        "player_0": 1.25,
                        "player_1": 2.5,
                        "maximum": 2.5,
                    },
                    "confidence_interval": {
                        "player_0": [1.0, 1.5],
                        "player_1": [2.0, 3.0],
                        "maximum": [2.1, 3.1],
                    },
                    "series": [
                        {
                            "stage_iteration": 6,
                            "point": {
                                "player_0": 1.0,
                                "player_1": 2.0,
                                "maximum": 2.0,
                            },
                            "confidence_interval": {
                                "player_0": [0.8, 1.2],
                                "player_1": [1.7, 2.3],
                                "maximum": [1.8, 2.4],
                            },
                        },
                        {
                            "stage_iteration": 7,
                            "point": {
                                "player_0": 1.25,
                                "player_1": 2.5,
                                "maximum": 2.5,
                            },
                            "confidence_interval": {
                                "player_0": [1.0, 1.5],
                                "player_1": [2.0, 3.0],
                                "maximum": [2.1, 3.1],
                            },
                        },
                    ],
                }
            ],
        }
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary).resolve()
            output = StringIO()
            with (
                patch(
                    "stockpile.training.artifacts.resolve_run",
                    return_value=SimpleNamespace(path=run_dir),
                ) as resolve_run,
                patch(
                    "stockpile.training.regret.analyze_run",
                    return_value=report,
                ),
            ):
                status = terminal.main(
                    [
                        "analyze",
                        "--mode",
                        "lite",
                        "--run",
                        "3",
                        "--confidence",
                        ".8",
                    ],
                    output=output,
                )
            report_path = (
                run_dir / "analysis" / "sampled_average_regret.json"
            )
            self.assertTrue(report_path.is_file())
            self.assertIn(
                '"availability"',
                report_path.read_text(encoding="utf-8"),
            )

        self.assertEqual(status, 0)
        resolve_run.assert_called_once_with("lite", run=3, smoke=False)
        rendered = output.getvalue()
        self.assertIn(
            "sampled average regret with an empirical confidence interval",
            rendered,
        )
        self.assertIn("Stage 2 (rounds 3)", rendered)
        self.assertIn("Iteration 6", rendered)
        self.assertIn("Iteration 7", rendered)
        self.assertIn("player 0: estimate=1.25; 80% CI=[1, 1.5]", rendered)
        self.assertIn("player 1: estimate=2.5; 80% CI=[2, 3]", rendered)
        self.assertIn("maximum: estimate=2.5; 80% CI=[2.1, 3.1]", rendered)
        for forbidden in ("exploitability", "NashConv", "equilibrium"):
            self.assertNotIn(forbidden.lower(), rendered.lower())

    def test_lite_rules_defaults_are_resolved_parameters_only(self):
        output = StringIO()
        status = terminal.main(["rules", "--mode", "lite"], output=output)
        rendered = output.getvalue()
        self.assertEqual(status, 0)
        self.assertEqual(
            rendered.splitlines()[:3],
            ["Mode: lite", "Players: 2", "Rounds: 6"],
        )
        for label in (
            "Impact: off",
            "Investor: off",
            "Hand: off",
            "Fees: off",
            "Dividend: off",
            "Split: off",
            "Majority: off",
            "Stock tracks: off",
            "Sell order: off",
            "Starting cash: 30K",
            "Bidding meeples: 2",
            "Stockpiles: 4",
        ):
            self.assertIn(label, rendered)
        self.assertNotIn("Phase", rendered)
        self.assertNotIn("Turn ", rendered)

    def test_rules_explicit_values_override_mode_defaults(self):
        output = StringIO()
        status = terminal.main(
            [
                "rules",
                "--mode",
                "lite",
                "--players",
                "3",
                "--impact",
                "on",
                "--hand",
                "on",
                "--fees",
                "on",
                "--dividend",
                "on",
                "--sell-order",
                "on",
            ],
            output=output,
        )
        rendered = output.getvalue()
        self.assertEqual(status, 0)
        self.assertIn("Players: 3", rendered)
        self.assertIn("Impact: on", rendered)
        self.assertIn("Investor: off", rendered)
        for label in (
            "Hand: on",
            "Fees: on",
            "Dividend: on",
            "Split: off",
            "Majority: off",
            "Stock tracks: off",
        ):
            self.assertIn(label, rendered)
        self.assertIn("Sell order: on", rendered)
        self.assertIn("Starting cash: 20K", rendered)
        self.assertIn("Bidding meeples: 1", rendered)
        self.assertIn("Stockpiles: 3", rendered)

    def test_classic_and_deluxe_fixed_mechanics(self):
        cases = (
            ("classic", "Impact: on", "Investor: off"),
            ("deluxe", "Impact: on", "Investor: on"),
        )
        for mode, impact, investor in cases:
            with self.subTest(mode=mode):
                output = StringIO()
                status = terminal.main(
                    ["rules", "--mode", mode],
                    output=output,
                )
                self.assertEqual(status, 0)
                rendered = output.getvalue()
                self.assertIn(impact, rendered)
                self.assertIn(investor, rendered)
                for optional_default in (
                    "Hand: on",
                    "Fees: on",
                    "Dividend: on",
                    "Split: on",
                    "Majority: on",
                    "Stock tracks: off",
                    "Sell order: on",
                ):
                    self.assertIn(optional_default, rendered)

    def test_size_upper_bound_is_accepted_end_to_end(self):
        output = StringIO()
        status = terminal.main(
            [
                "rules",
                "--mode",
                "deluxe",
                "--players",
                "5",
                "--rounds",
                "10",
            ],
            output=output,
        )
        self.assertEqual(status, 0)
        self.assertIn("Players: 5", output.getvalue())
        self.assertIn("Rounds: 10", output.getvalue())

    def test_every_optional_rule_can_override_classic_and_deluxe(self):
        for mode in ("classic", "deluxe"):
            with self.subTest(mode=mode):
                output = StringIO()
                status = terminal.main(
                    [
                        "rules",
                        "--mode",
                        mode,
                        "--hand",
                        "off",
                        "--fees",
                        "off",
                        "--dividend",
                        "off",
                        "--split",
                        "off",
                        "--majority",
                        "off",
                        "--stock-tracks",
                        "on",
                        "--sell-order",
                        "off",
                    ],
                    output=output,
                )
                rendered = output.getvalue()
                self.assertEqual(status, 0)
                for label in (
                    "Hand: off",
                    "Fees: off",
                    "Dividend: off",
                    "Split: off",
                    "Majority: off",
                    "Stock tracks: on",
                    "Sell order: off",
                ):
                    self.assertIn(label, rendered)

    def test_rules_does_not_calculate_complexity_or_explain_gameplay(self):
        with (
            patch(
                "stockpile.main.interface.resolve_interface_complexity"
            ) as complexity,
            patch("stockpile.main.interface.explain_configuration") as explain,
        ):
            status = terminal.main(
                ["rules", "--mode", "lite"],
                output=StringIO(),
            )
        self.assertEqual(status, 0)
        complexity.assert_not_called()
        explain.assert_not_called()

    def test_complexity_receives_the_single_resolved_configuration(self):
        real_configuration = stockpile.resolve_configuration(
            "lite",
            player_count=3,
            round_count=4,
            fees=True,
            sell_order=False,
        )

        def resolve_complexity(configuration, **kwargs):
            self.assertIs(configuration, real_configuration)
            self.assertEqual(kwargs["cache_policy"], "refresh")
            self.assertEqual(kwargs["max_states"], 123)
            self.assertEqual(kwargs["max_seconds"], 1.5)
            return _resolved_complexity(
                configuration,
                exact=False,
                max_states=kwargs["max_states"],
                max_seconds=kwargs["max_seconds"],
            )

        output = StringIO()
        with (
            patch(
                "stockpile.main.interface.resolve_configuration",
                return_value=real_configuration,
            ) as configure,
            patch(
                "stockpile.main.interface.resolve_interface_complexity",
                side_effect=resolve_complexity,
            ) as calculate,
        ):
            status = terminal.main(
                [
                    "complexity",
                    "--mode",
                    "lite",
                    "--players",
                    "3",
                    "--rounds",
                    "4",
                    "--fees",
                    "on",
                    "--sell-order",
                    "off",
                    "--max-states",
                    "123",
                    "--max-seconds",
                    "1.5",
                    "--refresh",
                ],
                output=output,
            )
        self.assertEqual(status, 0)
        configure.assert_called_once_with(
            "lite",
            player_count=3,
            round_count=4,
            hand=None,
            fees=True,
            dividend=None,
            split=None,
            majority=None,
            stock_tracks=None,
            sell_order=False,
            action_space_mode="compact",
            impact=None,
        )
        calculate.assert_called_once()
        rendered = output.getvalue()
        self.assertIn("Status: lower bound", rendered)
        self.assertIn("Information sets: >=12", rendered)
        self.assertIn("Information-set actions: >=34", rendered)
        self.assertIn("States visited: 91", rendered)

    def test_exact_complexity_is_not_prefixed_as_a_lower_bound(self):
        configuration = stockpile.resolve_configuration("lite")
        remembered = _resolved_complexity(
            configuration,
            exact=True,
            max_states=1000,
            max_seconds=3.0,
        )
        output = StringIO()
        with patch(
            "stockpile.main.interface.resolve_interface_complexity",
            return_value=remembered,
        ):
            status = terminal.main(
                ["complexity", "--mode", "lite"],
                output=output,
            )
        self.assertEqual(status, 0)
        rendered = output.getvalue()
        self.assertIn("Status: exact", rendered)
        self.assertIn("Information sets: 12", rendered)
        self.assertNotIn("Information sets: >=", rendered)

    def test_play_lazily_launches_the_complete_local_trainer(self):
        output = StringIO()
        calls: list[tuple[str, int, object]] = []
        web_module = ModuleType("stockpile.web")

        def run_trainer(*, host: str, port: int, output: object) -> int:
            calls.append((host, port, output))
            return 0

        web_module.run_trainer = run_trainer
        with patch.dict(sys.modules, {"stockpile.web": web_module}):
            status = terminal.main(
                [
                    "play",
                    "--mode",
                    "lite",
                    "--host",
                    "localhost",
                    "--port",
                    "8123",
                ],
                output=output,
            )
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(calls, [("localhost", 8123, output)])

    def test_play_defaults_to_lite_and_canonical_loopback_ports(self):
        output = StringIO()
        calls: list[tuple[str, int, object]] = []
        web_module = ModuleType("stockpile.web")

        def run_trainer(*, host: str, port: int, output: object) -> int:
            calls.append((host, port, output))
            return 130

        web_module.run_trainer = run_trainer
        with patch.dict(sys.modules, {"stockpile.web": web_module}):
            status = terminal.main(["play"], output=output)

        self.assertEqual(status, 130)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(calls, [("127.0.0.1", 8000, output)])

    def test_play_reports_a_child_failure_without_a_traceback(self):
        web_module = ModuleType("stockpile.web")

        def run_trainer(*, host: str, port: int, output: object) -> int:
            del host, port, output
            raise RuntimeError(
                "Stockpile frontend exited before readiness with status 1"
            )

        web_module.run_trainer = run_trainer
        stderr = StringIO()
        with (
            patch.dict(sys.modules, {"stockpile.web": web_module}),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            terminal.main(["play"], output=StringIO())

        self.assertEqual(raised.exception.code, 2)
        self.assertIn(
            "frontend exited before readiness with status 1", stderr.getvalue()
        )
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_solve_rejects_games_outside_the_supported_lite_contract(self):
        cases = (
            ["solve", "--mode", "classic", "--players", "5"],
            ["solve", "--mode", "lite", "--rounds", "1", "--impact", "on"],
        )
        for argv in cases:
            with self.subTest(argv=argv):
                with (
                    patch(
                        "stockpile.training.artifacts.resolve_fresh_output"
                    ) as reserve,
                    redirect_stderr(StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    terminal.main(argv, output=StringIO())
                self.assertEqual(raised.exception.code, 2)
                reserve.assert_not_called()

    def test_solve_routes_normal_and_smoke_to_distinct_numbered_runs(self):
        captured = []

        class FakeTrainer:
            def __init__(self, config, *, base_configuration, output):
                del base_configuration, output
                self.config = config
                self.device = "cpu"
                captured.append(config)

            def train(self, *, resume, overwrite):
                del resume, overwrite
                rounds = self.config.curriculum.rounds
                return SimpleNamespace(
                    completed_rounds=rounds,
                    final_checkpoint=self.config.output_dir / "full.pt",
                    final_policy=self.config.output_dir / "policy.pt",
                )

        trainer_module = ModuleType("stockpile.training.trainer")
        trainer_module.DeepCFRTrainer = FakeTrainer
        cases = (
            (["solve", "--mode", "lite", "--rounds", "1"], "lite", 2_000),
            (
                ["solve", "--mode", "lite", "--rounds", "1", "--smoke"],
                "smoke",
                512,
            ),
        )
        for argv, namespace, expected_capacity in cases:
            with self.subTest(argv=argv), TemporaryDirectory() as temporary:
                output = StringIO()
                artifact_root = Path(temporary) / "artifacts"
                with (
                    chdir(temporary),
                    patch(
                        "stockpile.training.artifacts.DEFAULT_ARTIFACT_ROOT",
                        artifact_root,
                    ),
                    patch.dict(
                        sys.modules,
                        {"stockpile.training.trainer": trainer_module},
                    ),
                ):
                    status = terminal.main(argv, output=output)
                self.assertEqual(status, 0)
                config = captured.pop(0)
                self.assertEqual(
                    config.output_dir,
                    (artifact_root / namespace / "run_01").resolve(),
                )
                self.assertTrue((config.output_dir / "run.json").is_file())
                self.assertIn(
                    '"state": "completed"',
                    (config.output_dir / "run.json").read_text(encoding="utf-8"),
                )
                self.assertEqual(config.batch_size, 32)
                self.assertEqual(config.memory_capacity, expected_capacity)
                rendered = output.getvalue()
                self.assertIn("Curriculum: 1", rendered)
                self.assertIn("Device: cpu", rendered)
                self.assertIn("(3 reservoirs)", rendered)

    def test_solve_rejects_run_output_conflicts_and_existing_run_numbers(self):
        with TemporaryDirectory() as temporary:
            artifact_root = Path(temporary) / "artifacts"
            with (
                patch(
                    "stockpile.training.artifacts.DEFAULT_ARTIFACT_ROOT",
                    artifact_root,
                ),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                terminal.main(
                    ["solve", "--mode", "lite", "--smoke"],
                    output=StringIO(),
                )
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(artifact_root.exists())

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            terminal.main(
                [
                    "solve",
                    "--mode",
                    "lite",
                    "--rounds",
                    "1",
                    "--run",
                    "2",
                    "--output-dir",
                    "elsewhere",
                ],
                output=StringIO(),
            )
        self.assertEqual(raised.exception.code, 2)

        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            terminal.main(
                [
                    "solve",
                    "--mode",
                    "lite",
                    "--rounds",
                    "1",
                    "--resume",
                    "source.pt",
                    "--overwrite",
                ],
                output=StringIO(),
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("requires a differing unmanaged --output-dir", stderr.getvalue())

        from stockpile.training.artifacts import reserve_run

        with TemporaryDirectory() as temporary:
            artifact_root = Path(temporary) / "artifacts"
            reserve_run("lite", run=2, artifact_root=artifact_root)
            stderr = StringIO()
            with (
                patch(
                    "stockpile.training.artifacts.DEFAULT_ARTIFACT_ROOT",
                    artifact_root,
                ),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                terminal.main(
                    [
                        "solve",
                        "--mode",
                        "lite",
                        "--rounds",
                        "1",
                        "--run",
                        "2",
                    ],
                    output=StringIO(),
                )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("artifact run already exists", stderr.getvalue())

    def test_solve_delegates_resume_destination_without_reusing_raw_cli_path(self):
        captured = {}

        class FakeTrainer:
            def __init__(self, config, *, base_configuration, output):
                del base_configuration, output
                self.config = config
                self.device = "cpu"

            def load_checkpoint(self, path):
                captured["preloaded"] = path

            def train(self, *, resume, overwrite):
                captured.update(resume=resume, overwrite=overwrite)
                return SimpleNamespace(
                    completed_rounds=self.config.curriculum.rounds,
                    final_checkpoint=self.config.output_dir / "full.pt",
                    final_policy=self.config.output_dir / "policy.pt",
                )

        trainer_module = ModuleType("stockpile.training.trainer")
        trainer_module.DeepCFRTrainer = FakeTrainer
        with TemporaryDirectory() as temporary:
            destination = Path(temporary).resolve() / "destination"
            resolved_checkpoint = Path(temporary).resolve() / "source" / "full.pt"
            run_ref = SimpleNamespace(
                output_dir=destination,
                smoke=False,
                managed=False,
                state=None,
            )
            plan = SimpleNamespace(
                destination=run_ref,
                checkpoint=resolved_checkpoint,
            )
            with (
                patch(
                    "stockpile.training.artifacts.plan_resume_destination",
                    return_value=plan,
                ) as plan_resume,
                patch.dict(
                    sys.modules,
                    {"stockpile.training.trainer": trainer_module},
                ),
            ):
                status = terminal.main(
                    [
                        "solve",
                        "--mode",
                        "lite",
                        "--rounds",
                        "1",
                        "--resume",
                        "raw-checkpoint.pt",
                        "--output-dir",
                        str(destination),
                        "--overwrite",
                    ],
                    output=StringIO(),
                )

        self.assertEqual(status, 0)
        plan_resume.assert_called_once_with(
            Path("raw-checkpoint.pt"),
            mode="lite",
            output_dir=destination,
            run=None,
            smoke=None,
            overwrite=True,
        )
        self.assertEqual(captured["resume"], resolved_checkpoint)
        self.assertEqual(captured["preloaded"], resolved_checkpoint)
        self.assertFalse(captured["overwrite"])

    def test_invalid_resume_does_not_advance_reserved_run_lifecycle(self):
        class FakeTrainer:
            def __init__(self, config, *, base_configuration, output):
                del base_configuration, output
                self.config = config
                self.device = "cpu"

            def load_checkpoint(self, path):
                del path
                raise ValueError("checkpoint is incompatible")

            def train(self, *, resume, overwrite):  # pragma: no cover
                del resume, overwrite
                raise AssertionError("training must not start after invalid resume")

        trainer_module = ModuleType("stockpile.training.trainer")
        trainer_module.DeepCFRTrainer = FakeTrainer
        with TemporaryDirectory() as temporary:
            run_ref = SimpleNamespace(
                output_dir=Path(temporary).resolve() / "lite" / "run_01",
                smoke=False,
                managed=True,
                state="reserved",
            )
            plan = SimpleNamespace(
                destination=run_ref,
                checkpoint=Path(temporary).resolve() / "source" / "full.pt",
            )
            stderr = StringIO()
            with (
                patch(
                    "stockpile.training.artifacts.plan_resume_destination",
                    return_value=plan,
                ),
                patch(
                    "stockpile.training.artifacts.update_run_manifest"
                ) as update_manifest,
                patch.dict(
                    sys.modules,
                    {"stockpile.training.trainer": trainer_module},
                ),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                terminal.main(
                    [
                        "solve",
                        "--mode",
                        "lite",
                        "--rounds",
                        "1",
                        "--resume",
                        "raw-checkpoint.pt",
                    ],
                    output=StringIO(),
                )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("checkpoint is incompatible", stderr.getvalue())
        update_manifest.assert_not_called()

    def test_solve_reports_device_checkpoint_and_numerical_failures_cleanly(self):
        cases = (
            (
                RuntimeError("CUDA was requested but is not available"),
                True,
                ["--device", "cuda"],
            ),
            (
                FileNotFoundError("checkpoint does not exist"),
                False,
                ["--resume", "missing.pt"],
            ),
            (
                FloatingPointError("regret target is nonfinite"),
                False,
                [],
            ),
        )
        for error, fail_during_construction, extra in cases:
            with self.subTest(error=type(error).__name__), TemporaryDirectory() as temporary:
                class FailingTrainer:
                    def __init__(self, config, *, base_configuration, output):
                        del config, base_configuration, output
                        if fail_during_construction:
                            raise error
                        self.device = "cpu"

                    def load_checkpoint(self, path):
                        del path
                        if extra and extra[0] == "--resume":
                            raise error

                    def train(self, *, resume, overwrite):
                        del resume, overwrite
                        raise error

                trainer_module = ModuleType("stockpile.training.trainer")
                trainer_module.DeepCFRTrainer = FailingTrainer
                stderr = StringIO()
                argv = [
                    "solve",
                    "--mode",
                    "lite",
                    "--rounds",
                    "1",
                    "--output-dir",
                    str(Path(temporary) / "output"),
                    *extra,
                ]
                with (
                    patch.dict(
                        sys.modules,
                        {"stockpile.training.trainer": trainer_module},
                    ),
                    redirect_stderr(stderr),
                    self.assertRaises(SystemExit) as raised,
                ):
                    terminal.main(argv, output=StringIO())

                self.assertEqual(raised.exception.code, 2)
                rendered = stderr.getvalue()
                self.assertIn(str(error), rendered)
                self.assertNotIn("Traceback", rendered)

    def test_configuration_mode_is_required_and_invalid_values_fail_in_argparse(self):
        invalid_commands = (
            ["rules"],
            ["complexity"],
            ["play", "--mode", "classic"],
            ["play", "--mode", "lite", "--host", "0.0.0.0"],
            ["play", "--mode", "lite", "--port", "0"],
            ["solve"],
            ["rules", "--mode", "lite", "--players", "1"],
            ["rules", "--mode", "lite", "--players", "6"],
            ["rules", "--mode", "lite", "--rounds", "0"],
            ["rules", "--mode", "lite", "--rounds", "11"],
            ["rules", "--mode", "lite", "--fees", "yes"],
            ["rules", "--mode", "lite", "--split", "on"],
            ["rules", "--mode", "lite", "--majority", "on"],
            ["rules", "--mode", "lite", "--stock-tracks", "on"],
            ["rules", "--mode", "classic", "--impact", "off"],
            ["rules", "--mode", "deluxe", "--investor", "off"],
        )
        for argv in invalid_commands:
            with self.subTest(argv=argv):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
                    terminal.main(argv, output=StringIO())
                self.assertEqual(raised.exception.code, 2)

    def test_defaults_and_explicit_size_arguments_are_equivalent(self):
        default = StringIO()
        explicit = StringIO()
        terminal.main(["rules", "--mode", "lite"], output=default)
        terminal.main(
            [
                "rules",
                "--mode",
                "lite",
                "--players",
                "2",
                "--rounds",
                "6",
            ],
            output=explicit,
        )
        self.assertEqual(default.getvalue(), explicit.getvalue())


if __name__ == "__main__":
    unittest.main()
