"""
Physical AI & Autonomous Robotics Intelligence Platform

Core package.

Provides shared application infrastructure for:
- Configuration and environment settings
- Security and authentication
- Logging and observability
- Exception handling
- Application constants
- Middleware and dependency utilities

Core modules should remain independent of database models,
API routers, and hardware-control implementations.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "get_settings",
    "setup_logging",
]


def get_settings() -> Any:
    """
    Lazily load application settings.

    Expected module:
        app.core.config

    Expected callable:
        get_settings()
    """
    module = import_module("app.core.config")
    return module.get_settings()


def setup_logging(*args: Any, **kwargs: Any) -> Any:
    """
    Lazily initialize application logging.

    Expected module:
        app.core.logging

    Expected callable:
        setup_logging()
    """
    module = import_module("app.core.logging")
    return module.setup_logging(*args, **kwargs)