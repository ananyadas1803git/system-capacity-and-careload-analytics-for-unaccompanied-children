"""Streamlit application package for the HHS UAC analytics dashboard.

This module intentionally does not import :mod:`app.streamlit_app`. Streamlit
entrypoints execute their user-interface code at import time, so eagerly
importing them would make ordinary package discovery and test collection
unsafe.  The constants and helpers below provide a side-effect-free way to
locate the application and its registered pages.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.pages import PageDefinition

APP_TITLE = "System Capacity & Care Load Analytics for Unaccompanied Children"
APP_VERSION = "1.0.0"

APP_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIRECTORY.parent
STREAMLIT_ENTRYPOINT = APP_DIRECTORY / "streamlit_app.py"
PAGES_DIRECTORY = APP_DIRECTORY / "pages"


def get_streamlit_entrypoint() -> Path:
    """Return the absolute path to the main Streamlit application script.

    Raises:
        FileNotFoundError: If the configured entrypoint is not present.
    """

    if not STREAMLIT_ENTRYPOINT.is_file():
        raise FileNotFoundError(
            f"Streamlit entrypoint was not found: {STREAMLIT_ENTRYPOINT}"
        )
    return STREAMLIT_ENTRYPOINT


def available_pages() -> tuple[PageDefinition, ...]:
    """Return the page registry without importing executable page modules."""

    from app.pages import iter_page_definitions

    return iter_page_definitions()


__all__ = [
    "APP_DIRECTORY",
    "APP_TITLE",
    "APP_VERSION",
    "PAGES_DIRECTORY",
    "PROJECT_ROOT",
    "STREAMLIT_ENTRYPOINT",
    "available_pages",
    "get_streamlit_entrypoint",
]
