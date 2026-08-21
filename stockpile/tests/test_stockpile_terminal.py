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
        for command in ("rules", "complexity", "play", "solve"):
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
        for command in ("rules", "complexity", "play", "solve"):
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
                    "--hand",
                    "--fees",
                    "--dividend",
                    "--split",
                    "--majority",
                    "--stock-tracks",
                    "--sell-order",
                ):
                    self.assertIn(option, rendered)
                self.assertNotIn("--impact", rendered)
                self.assertNotIn("--investor", rendered)

    def test_solve_help_explains_safe_memory_and_output_defaults(self):
        output = StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(output):
            terminal.main(["solve", "--help"])

        self.assertEqual(raised.exception.code, 0)
        rendered = output.getvalue()
        self.assertIn("default: 32", rendered)
        self.assertIn("default: 2000", rendered)
        self.assertIn("three stage-local reservoir", rendered)
        self.assertIn("artifacts/deep_cfr/smoke", rendered)
        self.assertIn("artifacts/deep_cfr/default", rendered)

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
                "--hand",
                "on",
                "--fees",
                "on",
                "--dividend",
                "on",
                "--split",
                "on",
                "--majority",
                "on",
                "--stock-tracks",
                "on",
                "--sell-order",
                "on",
            ],
            output=output,
        )
        rendered = output.getvalue()
        self.assertEqual(status, 0)
        self.assertIn("Players: 3", rendered)
        self.assertIn("Impact: off", rendered)
        self.assertIn("Investor: off", rendered)
        for label in (
            "Hand: on",
            "Fees: on",
            "Dividend: on",
            "Split: on",
            "Majority: on",
            "Stock tracks: on",
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

    def test_play_resolves_before_reporting_unimplemented(self):
        output = StringIO()
        with patch(
            "stockpile.main.interface.resolve_configuration",
            wraps=stockpile.resolve_configuration,
        ) as resolve:
            status = terminal.main(
                ["play", "--mode", "classic", "--players", "5"],
                output=output,
            )
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue(), "Not implemented.\n")
        resolve.assert_called_once()

    def test_solve_rejects_games_outside_the_supported_lite_contract(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            terminal.main(
                ["solve", "--mode", "classic", "--players", "5"],
                output=StringIO(),
            )
        self.assertEqual(raised.exception.code, 2)

    def test_solve_computes_distinct_smoke_and_default_output_paths(self):
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
            (
                ["solve", "--mode", "lite", "--rounds", "1"],
                Path("artifacts/deep_cfr/default"),
                2_000,
            ),
            (
                ["solve", "--mode", "lite", "--rounds", "1", "--smoke"],
                Path("artifacts/deep_cfr/smoke"),
                512,
            ),
        )
        for argv, expected_output, expected_capacity in cases:
            with self.subTest(argv=argv), TemporaryDirectory() as temporary:
                output = StringIO()
                with (
                    chdir(temporary),
                    patch.dict(
                        sys.modules,
                        {"stockpile.training.trainer": trainer_module},
                    ),
                ):
                    status = terminal.main(argv, output=output)
                self.assertEqual(status, 0)
                config = captured.pop(0)
                self.assertEqual(config.output_dir, expected_output)
                self.assertEqual(config.batch_size, 32)
                self.assertEqual(config.memory_capacity, expected_capacity)
                rendered = output.getvalue()
                self.assertIn("Curriculum: 1", rendered)
                self.assertIn("Device: cpu", rendered)
                self.assertIn("(3 reservoirs)", rendered)

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

    def test_mode_is_required_and_invalid_values_fail_in_argparse(self):
        invalid_commands = (
            ["rules"],
            ["complexity"],
            ["play"],
            ["solve"],
            ["rules", "--mode", "lite", "--players", "1"],
            ["rules", "--mode", "lite", "--players", "6"],
            ["rules", "--mode", "lite", "--rounds", "0"],
            ["rules", "--mode", "lite", "--rounds", "11"],
            ["rules", "--mode", "lite", "--fees", "yes"],
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
