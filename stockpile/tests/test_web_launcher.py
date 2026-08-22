"""Process-lifecycle contracts for the one-command local trainer launcher."""

from __future__ import annotations

from contextlib import nullcontext
from io import BytesIO, StringIO
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from stockpile.web import launcher


def _fake_child(name: str, pid: int):
    process = SimpleNamespace(
        pid=pid,
        poll=lambda: None,
        wait=lambda **_kwargs: 0,
    )
    return launcher._ChildProcess(name=name, process=process)


class LauncherTests(unittest.TestCase):
    def test_process_specs_use_the_configured_api_and_fixed_frontend(self):
        root = Path("/repository").resolve()
        with patch.object(launcher.sys, "executable", "/venv/bin/python"):
            backend, frontend = launcher._process_specs(
                root=root,
                npm="/usr/local/bin/npm",
                host="localhost",
                port=8123,
            )

        self.assertEqual(backend[0], "backend")
        self.assertEqual(
            backend[1],
            [
                "/venv/bin/python",
                "-m",
                "uvicorn",
                "stockpile.web.app:app",
                "--host",
                "localhost",
                "--port",
                "8123",
                "--workers",
                "1",
                "--log-level",
                "warning",
                "--no-access-log",
            ],
        )
        self.assertEqual(
            backend[2]["STOCKPILE_API_ORIGIN"], "http://localhost:8123"
        )
        self.assertEqual(frontend[0], "frontend")
        self.assertEqual(
            frontend[1][0:3],
            ["/usr/local/bin/npm", "--prefix", str(root / "frontend")],
        )
        self.assertEqual(
            frontend[1][-6:],
            ["--", "--host", "127.0.0.1", "--port", "5173", "--strictPort"],
        )
        self.assertEqual(
            frontend[2]["STOCKPILE_API_ORIGIN"], "http://localhost:8123"
        )

    def test_ipv6_api_origin_is_safe_for_the_vite_proxy(self):
        self.assertEqual(launcher._api_origin("::1", 8000), "http://[::1]:8000")

    def test_supervisor_rejects_non_loopback_api_hosts(self):
        for host in ("0.0.0.0", "192.0.2.1", "not-a-host"):
            with self.subTest(host=host):
                with self.assertRaisesRegex(ValueError, "loopback"):
                    launcher._api_origin(host, 8000)

    def test_run_trainer_validates_the_host_before_web_prerequisites(self):
        with patch.object(launcher, "_check_prerequisites") as prerequisites:
            with self.assertRaisesRegex(ValueError, "loopback"):
                launcher.run_trainer(host="0.0.0.0", output=StringIO())
        prerequisites.assert_not_called()

    def test_spawn_makes_a_quiet_independent_process_group(self):
        process = SimpleNamespace(stdout=BytesIO(b"vite startup output\n"))
        with (
            patch.object(launcher.os, "name", "posix"),
            patch.object(launcher.subprocess, "Popen", return_value=process) as popen,
        ):
            child = launcher._spawn_child(
                name="frontend",
                command=["npm", "run", "dev"],
                environment={"A": "B"},
                root=Path("/repository"),
            )

        self.assertEqual(child.name, "frontend")
        self.assertIs(child.process, process)
        assert child.output_thread is not None
        child.output_thread.join(timeout=1)
        self.assertEqual(child.output_tail.text(), "vite startup output")
        popen.assert_called_once_with(
            ["npm", "run", "dev"],
            cwd=Path("/repository"),
            env={"A": "B"},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def test_child_output_tail_is_bounded_and_only_attached_to_failures(self):
        tail = launcher._OutputTail(limit=8)
        tail.append(b"012345")
        tail.append(b"abcdef")
        self.assertEqual(tail.text(), "45abcdef")
        failed = SimpleNamespace(pid=10, poll=lambda: 7)
        child = launcher._ChildProcess("frontend", failed, output_tail=tail)
        with self.assertRaisesRegex(
            RuntimeError,
            "(?s)frontend exited before readiness with status 7.*"
            "frontend output:.*45abcdef",
        ):
            launcher._assert_children_running([child], phase="before readiness")

    def test_readiness_fails_immediately_when_either_child_exits(self):
        failed_process = SimpleNamespace(pid=10, poll=lambda: 9)
        children = [launcher._ChildProcess("backend", failed_process)]
        with patch.object(launcher, "_url_ready") as ready:
            with self.assertRaisesRegex(
                RuntimeError, "backend exited before readiness with status 9"
            ):
                launcher._wait_for_readiness(
                    children,
                    backend_url="http://127.0.0.1:8000/api/v2/setup",
                    frontend_url="http://127.0.0.1:5173",
                    timeout=1,
                )
        ready.assert_not_called()

    def test_readiness_timeout_identifies_both_missing_services(self):
        children = [_fake_child("backend", 10), _fake_child("frontend", 20)]
        with patch.object(launcher.time, "monotonic", side_effect=[10.0, 11.0]):
            with self.assertRaisesRegex(
                RuntimeError, "Timed out waiting for Stockpile backend and frontend"
            ):
                launcher._wait_for_readiness(
                    children,
                    backend_url="http://127.0.0.1:8000/api/v2/setup",
                    frontend_url="http://127.0.0.1:5173",
                    timeout=1,
                )

    def test_success_prints_only_after_both_services_are_ready(self):
        output = StringIO()
        children = [_fake_child("backend", 10), _fake_child("frontend", 20)]
        events: list[str] = []

        def ready(*_args, **_kwargs) -> None:
            events.append("ready")
            self.assertEqual(output.getvalue(), "")

        def monitor(_children) -> None:
            events.append("monitor")
            raise KeyboardInterrupt

        with (
            patch.object(launcher, "_repository_root", return_value=Path("/repo")),
            patch.object(launcher, "_check_prerequisites", return_value="npm"),
            patch.object(
                launcher,
                "_process_specs",
                return_value=(
                    ("backend", ["backend"], {}),
                    ("frontend", ["frontend"], {}),
                ),
            ),
            patch.object(launcher, "_spawn_child", side_effect=children),
            patch.object(
                launcher, "_wait_for_readiness", side_effect=ready
            ) as wait_for_readiness,
            patch.object(launcher, "_monitor_children", side_effect=monitor),
            patch.object(launcher, "_shutdown_children") as shutdown,
            patch.object(launcher, "_sigterm_as_interrupt", return_value=nullcontext()),
            patch("webbrowser.open") as open_browser,
        ):
            status = launcher.run_trainer(output=output)

        self.assertEqual(status, 130)
        self.assertEqual(events, ["ready", "monitor"])
        wait_for_readiness.assert_called_once_with(
            children,
            backend_url="http://127.0.0.1:8000/api/v2/setup",
            frontend_url="http://127.0.0.1:5173/api/v2/setup",
            timeout=30.0,
        )
        self.assertEqual(
            output.getvalue(),
            "Starting Stockpile Trainer...\n\nhttp://127.0.0.1:5173\n",
        )
        shutdown.assert_called_once_with(children, timeout=3.0)
        open_browser.assert_not_called()

    def test_startup_failure_is_silent_and_cleans_every_started_child(self):
        output = StringIO()
        children = [_fake_child("backend", 10), _fake_child("frontend", 20)]
        failure = RuntimeError("frontend never became ready")
        with (
            patch.object(launcher, "_repository_root", return_value=Path("/repo")),
            patch.object(launcher, "_check_prerequisites", return_value="npm"),
            patch.object(
                launcher,
                "_process_specs",
                return_value=(
                    ("backend", ["backend"], {}),
                    ("frontend", ["frontend"], {}),
                ),
            ),
            patch.object(launcher, "_spawn_child", side_effect=children),
            patch.object(launcher, "_wait_for_readiness", side_effect=failure),
            patch.object(launcher, "_shutdown_children") as shutdown,
            patch.object(launcher, "_sigterm_as_interrupt", return_value=nullcontext()),
        ):
            with self.assertRaisesRegex(RuntimeError, str(failure)):
                launcher.run_trainer(output=output)

        self.assertEqual(output.getvalue(), "")
        shutdown.assert_called_once_with(children, timeout=3.0)

    def test_child_exit_after_readiness_cleans_both_process_groups(self):
        output = StringIO()
        children = [_fake_child("backend", 10), _fake_child("frontend", 20)]
        failure = RuntimeError("Stockpile frontend exited unexpectedly with status 1")
        with (
            patch.object(launcher, "_repository_root", return_value=Path("/repo")),
            patch.object(launcher, "_check_prerequisites", return_value="npm"),
            patch.object(
                launcher,
                "_process_specs",
                return_value=(
                    ("backend", ["backend"], {}),
                    ("frontend", ["frontend"], {}),
                ),
            ),
            patch.object(launcher, "_spawn_child", side_effect=children),
            patch.object(launcher, "_wait_for_readiness"),
            patch.object(launcher, "_monitor_children", side_effect=failure),
            patch.object(launcher, "_shutdown_children") as shutdown,
            patch.object(launcher, "_sigterm_as_interrupt", return_value=nullcontext()),
        ):
            with self.assertRaisesRegex(RuntimeError, str(failure)):
                launcher.run_trainer(output=output)

        self.assertEqual(
            output.getvalue(),
            "Starting Stockpile Trainer...\n\nhttp://127.0.0.1:5173\n",
        )
        shutdown.assert_called_once_with(children, timeout=3.0)

    def test_shutdown_escalates_each_surviving_group_from_term_to_kill(self):
        first = _fake_child("backend", 10)
        second = _fake_child("frontend", 20)
        children = [first, second]
        with (
            patch.object(launcher, "_signal_child_group") as send,
            patch.object(launcher, "_child_group_alive", return_value=True),
        ):
            launcher._shutdown_children(children, timeout=0)

        self.assertEqual(
            send.call_args_list,
            [
                call(first, force=False),
                call(second, force=False),
                call(first, force=True),
                call(second, force=True),
            ],
        )

    def test_group_permission_failure_falls_back_to_the_direct_child(self):
        process = SimpleNamespace(
            pid=10,
            poll=lambda: None,
            terminate=Mock(),
            kill=Mock(),
        )
        child = launcher._ChildProcess("backend", process)
        with (
            patch.object(launcher.os, "name", "posix"),
            patch.object(launcher.os, "killpg", side_effect=PermissionError),
        ):
            launcher._signal_child_group(child, force=False)
            launcher._signal_child_group(child, force=True)

        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()

    @unittest.skipUnless(os.name == "posix", "POSIX process-group regression")
    def test_short_lived_group_leader_is_reaped_without_full_grace_wait(self):
        process = subprocess.Popen(
            ["/bin/sh", "-c", "exit 0"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        child = launcher._ChildProcess("short-lived", process)
        deadline = launcher.time.monotonic() + 1
        alive = True
        try:
            while alive and launcher.time.monotonic() < deadline:
                alive = launcher._child_group_alive(child)
                if alive:
                    launcher.time.sleep(0.01)
            self.assertFalse(alive)
            self.assertIsNotNone(process.returncode)
        finally:
            process.wait(timeout=1)

    def test_windows_uses_break_then_forced_tree_termination(self):
        process = SimpleNamespace(
            pid=42,
            poll=Mock(return_value=None),
            send_signal=Mock(),
        )
        child = launcher._ChildProcess("frontend", process)
        with (
            patch.object(launcher.os, "name", "nt"),
            patch.object(launcher.signal, "CTRL_BREAK_EVENT", 21, create=True),
            patch.object(launcher.subprocess, "run") as run,
        ):
            launcher._signal_child_group(child, force=False)
            launcher._signal_child_group(child, force=True)

        process.send_signal.assert_called_once_with(21)
        run.assert_called_once_with(
            ["taskkill", "/PID", "42", "/T", "/F"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def test_windows_forces_tree_cleanup_after_group_leader_exits(self):
        process = SimpleNamespace(
            pid=42,
            poll=Mock(return_value=0),
            wait=Mock(return_value=0),
        )
        child = launcher._ChildProcess(
            "frontend", process, force_tree_cleanup=True
        )
        with (
            patch.object(launcher, "_signal_child_group") as send,
            patch.object(launcher, "_child_group_alive", return_value=False),
        ):
            launcher._shutdown_children([child], timeout=0)

        self.assertEqual(
            send.call_args_list,
            [call(child, force=False), call(child, force=True)],
        )

    def test_windows_graceful_signal_race_does_not_abort_cleanup(self):
        process = SimpleNamespace(
            pid=42,
            poll=Mock(return_value=None),
            send_signal=Mock(side_effect=ProcessLookupError),
        )
        child = launcher._ChildProcess(
            "frontend", process, force_tree_cleanup=True
        )
        with (
            patch.object(launcher.os, "name", "nt"),
            patch.object(launcher.signal, "CTRL_BREAK_EVENT", 21, create=True),
        ):
            launcher._signal_child_group(child, force=False)

        process.send_signal.assert_called_once_with(21)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
