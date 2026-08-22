"""Metadata registry for the Streamlit multipage dashboard.

Page scripts must not be imported from this module. Each script contains
top-level Streamlit rendering commands and should only be executed by the
Streamlit runtime. The registry provides navigation metadata while keeping
package imports side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PAGE_DIRECTORY = Path(__file__).resolve().parent


class PageRegistryError(LookupError):
    """Raised when dashboard page metadata is invalid or cannot be resolved."""


@dataclass(frozen=True, slots=True)
class PageDefinition:
    """Describes one Streamlit page without importing its Python module."""

    key: str
    title: str
    icon: str
    description: str
    filename: str
    sort_order: int

    @property
    def path(self) -> Path:
        """Return the absolute filesystem path for this page."""

        return PAGE_DIRECTORY / self.filename


PAGE_DEFINITIONS: tuple[PageDefinition, ...] = (
    PageDefinition(
        key="overview",
        title="System Overview",
        icon="🏛️",
        description="Executive view of total care load and system health.",
        filename="overview.py",
        sort_order=10,
    ),
    PageDefinition(
        key="backlog",
        title="Backlog Pressure",
        icon="📈",
        description="Net intake pressure, accumulation streaks, and episodes.",
        filename="backlog.py",
        sort_order=20,
    ),
    PageDefinition(
        key="capacity",
        title="Capacity Planning",
        icon="🏥",
        description="Care-load capacity scenarios and operational thresholds.",
        filename="capacity.py",
        sort_order=30,
    ),
    PageDefinition(
        key="insights",
        title="Analytical Insights",
        icon="💡",
        description="Automated findings and decision-support observations.",
        filename="insights.py",
        sort_order=40,
    ),
    PageDefinition(
        key="kpis",
        title="KPI Performance",
        icon="🎯",
        description="Current performance of core capacity and flow indicators.",
        filename="kpis.py",
        sort_order=50,
    ),
    PageDefinition(
        key="trends",
        title="Longitudinal Trends",
        icon="📊",
        description="Daily, weekly, and monthly movement across the care system.",
        filename="trends.py",
        sort_order=60,
    ),
    PageDefinition(
        key="forecasting",
        title="Forecast Research",
        icon="🔬",
        description="Precomputed seven-day model comparison and diagnostics.",
        filename="forecasting.py",
        sort_order=70,
    ),
)

PAGE_KEYS = tuple(page.key for page in PAGE_DEFINITIONS)
_PAGE_BY_KEY = {page.key: page for page in PAGE_DEFINITIONS}


def iter_page_definitions() -> tuple[PageDefinition, ...]:
    """Return page definitions in deterministic navigation order."""

    return tuple(sorted(PAGE_DEFINITIONS, key=lambda page: page.sort_order))


def get_page_definition(key: str) -> PageDefinition:
    """Return the registered page for ``key``.

    Args:
        key: Case-insensitive page key, such as ``"overview"``.

    Raises:
        PageRegistryError: If ``key`` is blank or is not registered.
    """

    normalized_key = str(key).strip().casefold()
    try:
        return _PAGE_BY_KEY[normalized_key]
    except KeyError as exc:
        choices = ", ".join(PAGE_KEYS)
        raise PageRegistryError(
            f"Unknown dashboard page {key!r}. Expected one of: {choices}."
        ) from exc


def get_page_path(key: str, *, require_exists: bool = True) -> Path:
    """Resolve a registered page to its absolute Python file path."""

    page_path = get_page_definition(key).path
    if require_exists and not page_path.is_file():
        raise PageRegistryError(f"Registered page file does not exist: {page_path}")
    return page_path


def validate_page_registry() -> None:
    """Validate unique metadata and verify that all registered files exist."""

    if len(PAGE_KEYS) != len(set(PAGE_KEYS)):
        raise PageRegistryError("Page registry contains duplicate keys.")

    filenames = tuple(page.filename for page in PAGE_DEFINITIONS)
    if len(filenames) != len(set(filenames)):
        raise PageRegistryError("Page registry contains duplicate filenames.")

    sort_orders = tuple(page.sort_order for page in PAGE_DEFINITIONS)
    if len(sort_orders) != len(set(sort_orders)):
        raise PageRegistryError("Page registry contains duplicate sort orders.")

    missing = [str(page.path) for page in PAGE_DEFINITIONS if not page.path.is_file()]
    if missing:
        raise PageRegistryError("Registered page files are missing: " + ", ".join(missing))


__all__ = [
    "PAGE_DEFINITIONS",
    "PAGE_DIRECTORY",
    "PAGE_KEYS",
    "PageDefinition",
    "PageRegistryError",
    "get_page_definition",
    "get_page_path",
    "iter_page_definitions",
    "validate_page_registry",
]
