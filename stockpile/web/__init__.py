"""Lazy web entry point; importing :mod:`stockpile` never requires FastAPI."""

from __future__ import annotations

import ipaddress


def create_app(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Lazily construct the FastAPI application."""

    try:
        from .app import create_app as factory
    except ImportError as error:  # pragma: no cover - core-only installation.
        raise RuntimeError(
            "Browser play dependencies are missing; install requirements-web.txt"
        ) from error
    return factory(*args, **kwargs)


def run_server(*, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run one process-local Stockpile Lite API server."""

    if host.casefold() != "localhost":
        try:
            address = ipaddress.ip_address(host)
        except ValueError as error:
            raise ValueError("browser play must bind to a loopback host") from error
        if not address.is_loopback:
            raise ValueError("browser play must bind to a loopback host")

    try:
        import uvicorn
        from .app import create_app as factory
    except ImportError as error:  # pragma: no cover - exercised by core-only installs.
        raise RuntimeError(
            "Browser play dependencies are missing; install requirements-web.txt"
        ) from error
    uvicorn.run(factory(), host=host, port=port, reload=False, workers=1)


def run_trainer(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Lazily supervise the local API and Vite workstation."""

    from .launcher import run_trainer as supervisor

    return supervisor(*args, **kwargs)


__all__ = ["create_app", "run_server", "run_trainer"]
