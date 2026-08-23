"""Standard-library supervisor for the local Stockpile Trainer workstation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import ipaddress
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import BinaryIO, Iterator, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


FRONTEND_HOST = "127.0.0.1"
FRONTEND_PORT = 5173
FRONTEND_URL = f"http://{FRONTEND_HOST}:{FRONTEND_PORT}"
STARTUP_TIMEOUT_SECONDS = 30.0
SHUTDOWN_TIMEOUT_SECONDS = 3.0
POLL_INTERVAL_SECONDS = 0.05
OUTPUT_TAIL_BYTES = 16_384


class _OutputTail:
    """Thread-safe bounded tail for child output that is shown only on failure."""

    def __init__(self, limit: int = OUTPUT_TAIL_BYTES) -> None:
        self._limit = limit
        self._data = bytearray()
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self._lock:
            self._data.extend(chunk)
            overflow = len(self._data) - self._limit
            if overflow > 0:
                del self._data[:overflow]

    def text(self) -> str:
        with self._lock:
            return bytes(self._data).decode("utf-8", errors="replace").strip()


@dataclass(frozen=True, slots=True)
class _ChildProcess:
    name: str
    process: subprocess.Popen[bytes]
    output_tail: _OutputTail | None = None
    output_thread: threading.Thread | None = None
    force_tree_cleanup: bool = False


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _url_host(host: str) -> str:
    """Validate a loopback bind address and return its URL-safe form."""

    if host.casefold() == "localhost":
        return "localhost"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError(
            "browser play must bind to localhost or a loopback IP address"
        ) from error
    if not address.is_loopback:
        raise ValueError(
            "browser play must bind to localhost or a loopback IP address"
        )
    return f"[{address}]" if address.version == 6 else str(address)


def _api_origin(host: str, port: int) -> str:
    return f"http://{_url_host(host)}:{port}"


def _check_prerequisites(root: Path) -> str:
    missing = [
        package
        for package in ("fastapi", "uvicorn")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        raise RuntimeError(
            "Browser play dependencies are missing; install requirements-web.txt"
        )
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("Stockpile Trainer requires npm on PATH")
    frontend = root / "frontend"
    if not (frontend / "package.json").is_file():
        raise RuntimeError(f"Stockpile frontend was not found at {frontend}")
    if not (frontend / "node_modules" / ".bin" / "vite").exists():
        raise RuntimeError(
            "Frontend dependencies are missing; run npm --prefix frontend install"
        )
    return npm


def _process_specs(
    *,
    root: Path,
    npm: str,
    host: str,
    port: int,
) -> tuple[tuple[str, list[str], dict[str, str]], ...]:
    environment = os.environ.copy()
    environment["STOCKPILE_API_ORIGIN"] = _api_origin(host, port)
    return (
        (
            "backend",
            [
                sys.executable,
                "-m",
                "uvicorn",
                "stockpile.web.app:app",
                "--host",
                host,
                "--port",
                str(port),
                "--workers",
                "1",
                "--log-level",
                "warning",
                "--no-access-log",
            ],
            environment,
        ),
        (
            "frontend",
            [
                npm,
                "--prefix",
                str(root / "frontend"),
                "run",
                "dev",
                "--",
                "--host",
                FRONTEND_HOST,
                "--port",
                str(FRONTEND_PORT),
                "--strictPort",
            ],
            environment,
        ),
    )


def _spawn_child(
    *,
    name: str,
    command: list[str],
    environment: dict[str, str],
    root: Path,
) -> _ChildProcess:
    options: dict[str, object] = {
        "cwd": root,
        "env": environment,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
    }
    if os.name == "posix":
        options["start_new_session"] = True
    elif os.name == "nt":  # pragma: no cover - exercised on Windows only.
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(command, **options)  # type: ignore[arg-type]
    assert process.stdout is not None
    output_tail = _OutputTail()
    output_thread = threading.Thread(
        target=_drain_output,
        args=(process.stdout, output_tail),
        name=f"stockpile-{name}-output",
        daemon=True,
    )
    output_thread.start()
    return _ChildProcess(
        name=name,
        process=process,
        output_tail=output_tail,
        output_thread=output_thread,
        # Windows cannot reliably discover surviving descendants after the
        # ``npm.cmd`` group leader exits.  Retain the cleanup obligation on the
        # handle so shutdown still issues a forced tree kill in that case.
        force_tree_cleanup=os.name == "nt",
    )


def _drain_output(stream: BinaryIO, tail: _OutputTail) -> None:
    try:
        while chunk := stream.read(4096):
            tail.append(chunk)
    finally:
        stream.close()


def _url_ready(url: str) -> bool:
    opener = build_opener(ProxyHandler({}))
    request = Request(url, headers={"Cache-Control": "no-store"})
    try:
        with opener.open(request, timeout=0.4) as response:
            return response.status == 200
    except (HTTPError, URLError, TimeoutError, OSError):
        return False


def _assert_children_running(children: list[_ChildProcess], *, phase: str) -> None:
    for child in children:
        return_code = child.process.poll()
        if return_code is not None:
            if child.output_thread is not None:
                child.output_thread.join(timeout=0.2)
            message = (
                f"Stockpile {child.name} exited {phase} with status {return_code}"
            )
            tail = child.output_tail.text() if child.output_tail is not None else ""
            if tail:
                message += f"\n{child.name} output:\n{tail}"
            raise RuntimeError(message)


def _wait_for_readiness(
    children: list[_ChildProcess],
    *,
    backend_url: str,
    frontend_url: str,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    backend_ready = False
    frontend_ready = False
    while time.monotonic() < deadline:
        _assert_children_running(children, phase="before readiness")
        backend_ready = backend_ready or _url_ready(backend_url)
        frontend_ready = frontend_ready or _url_ready(frontend_url)
        if backend_ready and frontend_ready:
            return
        time.sleep(POLL_INTERVAL_SECONDS)
    missing = []
    if not backend_ready:
        missing.append("backend")
    if not frontend_ready:
        missing.append("frontend")
    message = "Timed out waiting for Stockpile " + " and ".join(missing)
    diagnostics = []
    for child in children:
        tail = child.output_tail.text() if child.output_tail is not None else ""
        if tail:
            diagnostics.append(f"{child.name} output:\n{tail}")
    if diagnostics:
        message += "\n" + "\n".join(diagnostics)
    raise RuntimeError(message)


def _monitor_children(children: list[_ChildProcess]) -> None:
    while True:
        _assert_children_running(children, phase="unexpectedly")
        time.sleep(0.2)


def _signal_child_group(child: _ChildProcess, *, force: bool) -> None:
    if os.name == "posix":
        signum = signal.SIGKILL if force else signal.SIGTERM
        try:
            os.killpg(child.process.pid, signum)
        except ProcessLookupError:
            return
        except PermissionError:
            # Some constrained local runners allow the group signal but deny
            # subsequent group probes/signals.  Retain a direct-child fallback
            # so cleanup stays bounded and never obscures the original exit.
            if child.process.poll() is not None:
                return
            try:
                if force:
                    child.process.kill()
                else:
                    child.process.terminate()
            except ProcessLookupError:
                return
    elif os.name == "nt":  # pragma: no cover - exercised on Windows only.
        if force:
            # Run this even when the direct npm.cmd process has exited: its
            # Vite/Node descendants may still be alive, and Windows' Popen
            # liveness check covers only the group leader.
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(child.process.pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except OSError:
                # ``taskkill`` is part of supported Windows installations, but
                # retain a bounded direct-child fallback for constrained hosts.
                if child.process.poll() is None:
                    try:
                        child.process.kill()
                    except OSError:
                        pass
        elif child.process.poll() is None:
            try:
                child.process.send_signal(signal.CTRL_BREAK_EVENT)
            except OSError:
                # The leader may exit between ``poll`` and signal delivery.
                # Forced tree cleanup still runs from ``_shutdown_children``.
                pass


def _child_group_alive(child: _ChildProcess) -> bool:
    return_code = child.process.poll()
    if return_code is not None:
        try:
            child.process.wait(timeout=0)
        except (subprocess.TimeoutExpired, ChildProcessError):
            pass
    if os.name != "posix":  # pragma: no cover - Windows only.
        return return_code is None
    try:
        os.killpg(child.process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return return_code is None
    return True


def _shutdown_children(
    children: list[_ChildProcess], *, timeout: float = SHUTDOWN_TIMEOUT_SECONDS
) -> None:
    for child in children:
        _signal_child_group(child, force=False)
    deadline = time.monotonic() + max(timeout, 0.0)
    while time.monotonic() < deadline:
        if not any(_child_group_alive(child) for child in children):
            break
        time.sleep(POLL_INTERVAL_SECONDS)
    for child in children:
        if child.force_tree_cleanup or _child_group_alive(child):
            _signal_child_group(child, force=True)
    for child in children:
        try:
            child.process.wait(timeout=0.5)
        except (subprocess.TimeoutExpired, ChildProcessError):
            pass
        if child.output_thread is not None:
            child.output_thread.join(timeout=0.2)


@contextmanager
def _sigterm_as_interrupt() -> Iterator[None]:
    """Give external supervisors the same cleanup path as Ctrl+C."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def stop(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_trainer(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    output: TextIO = sys.stdout,
    startup_timeout: float = STARTUP_TIMEOUT_SECONDS,
    shutdown_timeout: float = SHUTDOWN_TIMEOUT_SECONDS,
    computer_policy: str | None = None,
) -> int:
    """Start, supervise, and cleanly stop the API and Vite development server."""

    _url_host(host)
    root = _repository_root()
    npm = _check_prerequisites(root)
    from .policy import (
        COMPUTER_POLICY_ENV,
        RANDOM_POLICY_TOKEN,
        resolve_computer_policy_path,
    )

    if computer_policy is None:
        configured = os.environ.get(COMPUTER_POLICY_ENV)
        explicit_policy = False
    else:
        configured = computer_policy
        explicit_policy = True
    if configured is None:
        try:
            resolved = resolve_computer_policy_path()
            os.environ[COMPUTER_POLICY_ENV] = str(resolved)
            computer_policy_label = "deep_cfr"
        except Exception:
            os.environ[COMPUTER_POLICY_ENV] = RANDOM_POLICY_TOKEN
            computer_policy_label = "random"
    elif configured.strip().casefold() == RANDOM_POLICY_TOKEN:
        os.environ[COMPUTER_POLICY_ENV] = RANDOM_POLICY_TOKEN
        computer_policy_label = "random"
    else:
        resolved = resolve_computer_policy_path(policy=configured)
        os.environ[COMPUTER_POLICY_ENV] = str(resolved)
        computer_policy_label = (
            str(resolved) if explicit_policy else "deep_cfr"
        )
    children: list[_ChildProcess] = []
    interrupted = False
    with _sigterm_as_interrupt():
        try:
            for name, command, environment in _process_specs(
                root=root, npm=npm, host=host, port=port
            ):
                children.append(
                    _spawn_child(
                        name=name,
                        command=command,
                        environment=environment,
                        root=root,
                    )
                )
            _wait_for_readiness(
                children,
                backend_url=f"{_api_origin(host, port)}/api/v2/setup",
                frontend_url=f"{FRONTEND_URL}/api/v2/setup",
                timeout=startup_timeout,
            )
            print("Starting Stockpile Trainer...", file=output)
            print(f"Computer policy: {computer_policy_label}", file=output)
            print(file=output)
            print(FRONTEND_URL, file=output, flush=True)
            _monitor_children(children)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            _shutdown_children(children, timeout=shutdown_timeout)
    return 130 if interrupted else 0


__all__ = ["FRONTEND_URL", "run_trainer"]
