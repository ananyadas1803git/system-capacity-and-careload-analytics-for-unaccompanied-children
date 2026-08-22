"""Primary Streamlit entry point for HHS UAC capacity analytics.

Run from the project root with::

    streamlit run app/streamlit_app.py

The files in ``app/pages`` are discovered automatically by Streamlit and remain
independently executable specialist views.
"""

from __future__ import annotations

import html
import sys
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# A script launched from ``app/`` does not reliably receive the project root on
# sys.path.  Insert it before importing the shared backend and source packages.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_utils import (  # noqa: E402
    BACKLOG_STREAK_COLUMN,
    NET_INTAKE_COLUMN,
    DataValidationError,
    generate_mock_data,
)
from backend.analytics import (  # noqa: E402
    AnalysisConfig,
    AnalysisResult,
    AnalyticsError,
    CapacityAnalyticsEngine,
)
from src.kpi import (  # noqa: E402
    BACKLOG_KEY,
    NET_PRESSURE_KEY,
    OFFSET_KEY,
    TOTAL_CARE_KEY,
    VOLATILITY_KEY,
    KPIDashboardResult,
    KPIError,
    calculate_kpi_dashboard,
)
from src.preprocessor import (  # noqa: E402
    PreprocessedDataset,
    PreprocessingConfig,
    PreprocessingError,
    preprocess_data,
)
from src.report_generator import (  # noqa: E402
    ReportConfig,
    ReportGenerationError,
    generate_capacity_report,
)
from src.validation import (  # noqa: E402
    DatasetValidationResult,
    validate_capacity_data,
)
from src.visualisation import (  # noqa: E402
    PLOTLY_RENDER_CONFIG,
    DashboardFigures,
    VisualizationConfig,
    VisualizationError,
    create_dashboard_figures,
    create_validation_summary_chart,
)


NAVY = "#163B65"
SLATE_950 = "#172033"
SLATE_700 = "#3F5063"
SLATE_500 = "#66788A"
SLATE_200 = "#D9E1EA"
WHITE = "#FFFFFF"
BLUE = "#2E75B6"
TEAL = "#167C80"
AMBER = "#D97706"
RED = "#B42318"
GREEN = "#238636"
MAX_UPLOAD_BYTES = PreprocessingConfig().maximum_csv_bytes


st.set_page_config(
    page_title="HHS UAC Capacity Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "System Capacity & Care Load Analytics for Unaccompanied Children. "
            "A decision-support application based on an HHS operational framework."
        )
    },
)


def apply_page_styles() -> None:
    """Apply the shared slate/navy government-analytics visual language."""
    st.markdown(
        f"""
        <style>
            .stApp {{ background: #F5F7FA; color: {SLATE_950}; }}
            [data-testid="stSidebar"] {{
                background: #EDF2F7;
                border-right: 1px solid {SLATE_200};
            }}
            .hero {{
                background: linear-gradient(115deg, {SLATE_950}, {NAVY});
                border-top: 5px solid {BLUE}; border-radius: 10px;
                color: {WHITE}; margin-bottom: 1rem; padding: 1.45rem 1.7rem;
            }}
            .hero h1 {{ color: {WHITE}; font-size: 2rem; margin: 0 0 .35rem; }}
            .hero p {{ color: #DCE7F2; margin: 0; max-width: 920px; }}
            .eyebrow {{
                color: #B9D5F0; font-size: .72rem; font-weight: 700;
                letter-spacing: .11em; margin-bottom: .35rem; text-transform: uppercase;
            }}
            div[data-testid="stMetric"] {{
                background: {WHITE}; border: 1px solid {SLATE_200};
                border-left: 4px solid {BLUE}; border-radius: 8px;
                min-height: 132px; padding: .9rem 1rem;
            }}
            div[data-testid="stMetricLabel"] {{ color: {SLATE_500}; }}
            div[data-testid="stMetricValue"] {{ color: {NAVY}; }}
            .section-title {{
                border-bottom: 2px solid {NAVY}; color: {NAVY};
                font-size: 1.12rem; font-weight: 700; margin: 1.2rem 0 .75rem;
                padding-bottom: .35rem;
            }}
            .status-card {{
                background: {WHITE}; border: 1px solid {SLATE_200};
                border-left: 5px solid {BLUE}; border-radius: 8px;
                min-height: 112px; padding: .85rem 1rem;
            }}
            .status-card h4 {{ color: {NAVY}; margin: 0 0 .35rem; }}
            .status-card p {{ color: {SLATE_700}; margin: 0; }}
            .status-positive {{ border-left-color: {GREEN}; }}
            .status-watch {{ border-left-color: {AMBER}; }}
            .status-critical {{ border-left-color: {RED}; }}
            .source-banner {{
                background: #EFF6FF; border: 1px solid #B9D5F0;
                border-left: 5px solid {BLUE}; border-radius: 7px;
                color: #173A5E; margin: .35rem 0 1rem; padding: .75rem 1rem;
            }}
            .synthetic-banner {{
                background: #FFF8E6; border-color: #EDC46B;
                border-left-color: {AMBER}; color: #684400;
            }}
            .small-note {{ color: {SLATE_500}; font-size: .82rem; }}
            [data-testid="stDownloadButton"] button {{
                border-color: {NAVY}; color: {NAVY}; width: 100%;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_mock_dataset() -> pd.DataFrame:
    """Return a cached defensive copy of the deterministic demo dataset."""
    return generate_mock_data()


@st.cache_data(show_spinner=False)
def read_uploaded_dataset(file_bytes: bytes) -> pd.DataFrame:
    """Read uploaded CSV bytes as strings so validation sees source defects."""
    try:
        return pd.read_csv(
            BytesIO(file_bytes),
            dtype=str,
            encoding="utf-8-sig",
        )
    except (
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise PreprocessingError(f"The uploaded CSV could not be read: {exc}") from exc


def _source_data(uploaded_file: Any) -> tuple[pd.DataFrame, str, bool]:
    """Resolve an uploaded source or the transparent synthetic fallback."""
    if uploaded_file is None:
        return (
            load_mock_dataset().copy(),
            "Synthetic 2023–2025 demonstration dataset",
            True,
        )
    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise PreprocessingError("The uploaded CSV is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise PreprocessingError(
            f"The uploaded CSV is {len(file_bytes):,} bytes; the limit is "
            f"{MAX_UPLOAD_BYTES:,} bytes."
        )
    return read_uploaded_dataset(file_bytes).copy(), uploaded_file.name, False


def _date_range(value: Any) -> tuple[date, date] | None:
    """Normalize Streamlit's date-input return value when both dates exist."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return value[0], value[1]
    return None


def render_sidebar(
    preprocessed: PreprocessedDataset,
    source_name: str,
    synthetic: bool,
) -> tuple[date, date, str, int, str, bool]:
    """Render reporting controls and return validated selections."""
    minimum_date = preprocessed.data.index.min().date()
    maximum_date = preprocessed.data.index.max().date()
    with st.sidebar:
        st.header("Dashboard Controls")
        selected = st.date_input(
            "Reporting period",
            value=(minimum_date, maximum_date),
            min_value=minimum_date,
            max_value=maximum_date,
            help="KPI calculations and charts use this inclusive range.",
        )
        granularity = st.selectbox(
            "Chart granularity",
            options=["Daily", "Weekly", "Monthly"],
            help="Flows are summed; active care loads use period-end values.",
        )
        backlog_threshold = st.slider(
            "Elevated backlog threshold",
            min_value=2,
            max_value=14,
            value=3,
            help="Minimum consecutive positive-net-intake days highlighted in charts.",
        )
        comparison_mode = st.radio(
            "CBP/HHS comparison",
            options=["stacked", "lines"],
            format_func=lambda value: value.title(),
            horizontal=True,
        )
        show_anomalies = st.toggle(
            "Show anomaly markers",
            value=True,
            help="Overlay transfer and discharge constraint exceptions.",
        )
        st.divider()
        st.caption(f"Source: {source_name}")
        st.caption(f"Available: {minimum_date:%d %b %Y}–{maximum_date:%d %b %Y}")
        if synthetic:
            st.warning("Synthetic fallback data is active.")

    normalized = _date_range(selected)
    if normalized is None:
        st.info("Select both a reporting-period start and end date to continue.")
        st.stop()
    start_date, end_date = normalized
    if start_date > end_date:
        st.error("Reporting-period start must be on or before its end.")
        st.stop()
    return (
        start_date,
        end_date,
        granularity,
        backlog_threshold,
        comparison_mode,
        show_anomalies,
    )


def build_analysis(
    preprocessed: PreprocessedDataset,
    *,
    start_date: date,
    end_date: date,
    granularity: str,
    backlog_threshold: int,
) -> AnalysisResult:
    """Run the shared backend engine against the repaired daily dataset."""
    source = preprocessed.data.reset_index()
    engine = CapacityAnalyticsEngine(source)
    return engine.run(
        AnalysisConfig(
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
            backlog_threshold_days=backlog_threshold,
        )
    )


def render_kpi_summary(result: KPIDashboardResult) -> None:
    """Render five classified KPI cards with comparison-window deltas."""
    order = (
        TOTAL_CARE_KEY,
        NET_PRESSURE_KEY,
        VOLATILITY_KEY,
        BACKLOG_KEY,
        OFFSET_KEY,
    )
    columns = st.columns(5)
    for container, key in zip(columns, order, strict=True):
        item = result.kpis[key]
        delta: str | None = None
        if item.delta is not None and pd.notna(item.delta):
            if key == OFFSET_KEY:
                delta = f"{float(item.delta):+.1%} vs prior window"
            elif key == VOLATILITY_KEY:
                delta = f"{float(item.delta):+.2f} pp vs prior window"
            elif key == BACKLOG_KEY:
                delta = f"{float(item.delta):+,.0f} days vs prior window"
            else:
                delta = f"{float(item.delta):+,.0f} vs prior window"
        container.metric(
            item.name,
            item.formatted_value,
            delta=delta,
            delta_color="normal" if key == OFFSET_KEY else "inverse",
            help=f"{item.description} Formula: {item.formula}",
        )


def _current_streak(analysis: AnalysisResult) -> int:
    if analysis.daily_metrics.empty or BACKLOG_STREAK_COLUMN not in analysis.daily_metrics:
        return 0
    return int(
        pd.to_numeric(
            analysis.daily_metrics[BACKLOG_STREAK_COLUMN], errors="coerce"
        ).fillna(0).iloc[-1]
    )


def render_operational_status(
    analysis: AnalysisResult,
    kpis: KPIDashboardResult,
    validation: DatasetValidationResult,
) -> None:
    """Render concise pressure, capacity, and quality status callouts."""
    latest_pressure = int(
        pd.to_numeric(
            analysis.daily_metrics[NET_INTAKE_COLUMN], errors="coerce"
        ).fillna(0).iloc[-1]
    )
    current_streak = _current_streak(analysis)
    critical_alerts = sum(alert.severity == "critical" for alert in kpis.alerts)
    warning_alerts = sum(alert.severity == "watch" for alert in kpis.alerts)
    containers = st.columns(3)

    pressure_class = "status-watch" if latest_pressure > 0 else "status-positive"
    pressure_title = "Positive intake pressure" if latest_pressure > 0 else "Flow relief"
    containers[0].markdown(
        f"""
        <div class="status-card {pressure_class}">
            <h4>{pressure_title}</h4>
            <p>Latest net intake: <strong>{latest_pressure:+,}</strong>;
            active streak: <strong>{current_streak:,} day(s)</strong>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    alert_class = "status-critical" if critical_alerts else (
        "status-watch" if warning_alerts else "status-positive"
    )
    containers[1].markdown(
        f"""
        <div class="status-card {alert_class}">
            <h4>Operational alerts</h4>
            <p><strong>{critical_alerts}</strong> critical and
            <strong>{warning_alerts}</strong> watch-level KPI alert(s).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    validation_class = (
        "status-positive" if validation.report.is_valid else "status-critical"
    )
    containers[2].markdown(
        f"""
        <div class="status-card {validation_class}">
            <h4>Data quality</h4>
            <p>{validation.report.status}; score
            <strong>{validation.report.quality_score}/100</strong> with
            <strong>{validation.report.flagged_row_count:,}</strong> flagged row(s).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_charts(figures: DashboardFigures) -> None:
    """Render the five core charts in compact analytical tabs."""
    overview, comparison, backlog = st.tabs(
        [
            "System Load Overview",
            "CBP vs HHS Comparison",
            "Net Intake & Backlog",
        ]
    )
    with overview:
        st.plotly_chart(
            figures.system_load,
            width="stretch",
            config=PLOTLY_RENDER_CONFIG,
        )
    with comparison:
        st.plotly_chart(
            figures.care_load_comparison,
            width="stretch",
            config=PLOTLY_RENDER_CONFIG,
        )
    with backlog:
        st.plotly_chart(
            figures.net_intake_backlog,
            width="stretch",
            config=PLOTLY_RENDER_CONFIG,
        )

    flow_column, growth_column = st.columns(2)
    with flow_column:
        st.plotly_chart(
            figures.operational_flows,
            width="stretch",
            config=PLOTLY_RENDER_CONFIG,
        )
    with growth_column:
        st.plotly_chart(
            figures.growth_volatility,
            width="stretch",
            config=PLOTLY_RENDER_CONFIG,
        )


def render_page_directory() -> None:
    """Expose the specialist pages discovered by Streamlit."""
    st.markdown(
        '<div class="section-title">Specialist Analysis Pages</div>',
        unsafe_allow_html=True,
    )
    pages = (
        ("📋 Overview", "pages/overview.py", "Executive operational snapshot"),
        ("📈 Backlog", "pages/backlog.py", "Pressure episodes and accumulation"),
        ("🏥 Capacity", "pages/capacity.py", "Planning ceilings and headroom"),
        ("💡 Insights", "pages/insights.py", "Evidence-based analytical findings"),
        ("🎯 KPIs", "pages/kpis.py", "KPI definitions, status, and comparisons"),
        ("📊 Trends", "pages/trends.py", "Longitudinal patterns and volatility"),
    )
    columns = st.columns(3)
    for index, (label, page, description) in enumerate(pages):
        with columns[index % 3]:
            st.page_link(page, label=label, help=description, use_container_width=True)


def render_quality_logs(
    preprocessed: PreprocessedDataset,
    validation: DatasetValidationResult,
    analysis: AnalysisResult,
) -> None:
    """Render preprocessing repairs and independent analytical validation."""
    with st.expander("Data Quality & Validation Logs", expanded=False):
        summary_columns = st.columns(4)
        summary_columns[0].metric("Quality score", f"{validation.report.quality_score}/100")
        summary_columns[1].metric("Error types", validation.report.error_count)
        summary_columns[2].metric("Warning types", validation.report.warning_count)
        summary_columns[3].metric("Logical anomaly rows", len(analysis.anomaly_rows))

        validation_tab, preprocessing_tab, evidence_tab = st.tabs(
            ["Validation findings", "Preprocessing audit", "Flagged rows"]
        )
        with validation_tab:
            chart_column, table_column = st.columns([1, 2])
            with chart_column:
                st.plotly_chart(
                    create_validation_summary_chart(validation, height=330),
                    width="stretch",
                    config=PLOTLY_RENDER_CONFIG,
                )
            with table_column:
                st.dataframe(
                    validation.report.to_frame(),
                    width="stretch",
                    hide_index=True,
                )
        with preprocessing_tab:
            st.dataframe(
                preprocessed.report.to_frame(),
                width="stretch",
                hide_index=True,
            )
        with evidence_tab:
            if validation.flagged_rows.empty:
                st.success("No row-level validation exceptions were identified.")
            else:
                st.dataframe(
                    validation.flagged_rows,
                    width="stretch",
                    hide_index=True,
                )


def render_report_downloads(
    analysis: AnalysisResult,
    *,
    source_name: str,
    synthetic: bool,
) -> None:
    """Generate reproducible report artifacts and expose download buttons."""
    with st.expander("Download Analytical Report", expanded=False):
        st.caption(
            "Exports reflect the selected reporting period and current analytical results."
        )
        try:
            report = generate_capacity_report(
                analysis,
                ReportConfig(
                    source_label=source_name,
                    synthetic_data=synthetic,
                    granularity=analysis.config.granularity,
                    backlog_threshold_days=analysis.config.backlog_threshold_days,
                    include_daily_appendix=True,
                ),
            )
        except ReportGenerationError as exc:
            st.error(f"Unable to generate report exports: {exc}")
            return

        html_column, json_column, csv_column = st.columns(3)
        html_column.download_button(
            "Download HTML report",
            data=report.to_html_bytes(),
            file_name=f"{report.filename_stem}.html",
            mime="text/html",
        )
        json_column.download_button(
            "Download JSON report",
            data=report.to_json_bytes(),
            file_name=f"{report.filename_stem}.json",
            mime="application/json",
        )
        csv_column.download_button(
            "Download KPI table",
            data=report.table_to_csv_bytes("kpis"),
            file_name=f"{report.filename_stem}_kpis.csv",
            mime="text/csv",
        )


def main() -> None:
    """Load source data, run analytics, and render the home dashboard."""
    apply_page_styles()
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Operational Decision Support</div>
            <h1>System Capacity &amp; Care Load Analytics</h1>
            <p>Monitoring active CBP and HHS care loads, intake pressure,
            discharge performance, backlog accumulation, and data-quality risk
            for unaccompanied children.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Upload capacity CSV",
            type=["csv"],
            help="Expected fields follow the six-column HHS UAC project schema.",
        )

    try:
        raw_data, source_name, synthetic = _source_data(uploaded_file)
        preprocessed = preprocess_data(raw_data)
    except (PreprocessingError, DataValidationError, TypeError, ValueError) as exc:
        st.error(f"Unable to prepare the selected dataset: {exc}")
        st.info(
            "Verify the Date field and the five required CBP/HHS count columns, "
            "then upload the CSV again."
        )
        st.stop()

    if synthetic:
        st.markdown(
            """
            <div class="source-banner synthetic-banner">
                <strong>Synthetic demonstration mode.</strong> No CSV was uploaded,
                so the deterministic 2023–2025 fallback dataset is active. These
                values are not official HHS or CBP statistics.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        escaped_source_name = html.escape(source_name)
        st.markdown(
            f"""
            <div class="source-banner"><strong>Uploaded source:</strong>
            {escaped_source_name} · {len(raw_data):,} source row(s)</div>
            """,
            unsafe_allow_html=True,
        )

    (
        start_date,
        end_date,
        granularity,
        backlog_threshold,
        comparison_mode,
        show_anomalies,
    ) = render_sidebar(preprocessed, source_name, synthetic)

    try:
        analysis = build_analysis(
            preprocessed,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
            backlog_threshold=backlog_threshold,
        )
        kpis = calculate_kpi_dashboard(analysis.daily_metrics)
        validation = validate_capacity_data(analysis.daily_metrics)
        figures = create_dashboard_figures(
            analysis.daily_metrics,
            VisualizationConfig(
                granularity=granularity,
                backlog_threshold_days=backlog_threshold,
                comparison_mode=comparison_mode,
                show_anomalies=show_anomalies,
            ),
        )
    except (
        AnalyticsError,
        KPIError,
        VisualizationError,
        DataValidationError,
        TypeError,
        ValueError,
    ) as exc:
        st.error(f"Unable to calculate the selected analytical view: {exc}")
        st.stop()

    st.caption(
        f"Reporting period: {start_date:%d %b %Y}–{end_date:%d %b %Y} · "
        f"{len(analysis.daily_metrics):,} daily observations · {granularity} charts"
    )
    st.markdown(
        '<div class="section-title">Key Performance Indicators</div>',
        unsafe_allow_html=True,
    )
    render_kpi_summary(kpis)
    render_operational_status(analysis, kpis, validation)

    st.markdown(
        '<div class="section-title">Capacity and Care-Load Analysis</div>',
        unsafe_allow_html=True,
    )
    render_charts(figures)
    render_page_directory()
    render_quality_logs(preprocessed, validation, analysis)
    render_report_downloads(
        analysis,
        source_name=source_name,
        synthetic=synthetic,
    )
    st.divider()
    st.markdown(
        "<span class='small-note'>Decision-support analytics only. Validate "
        "material findings against authoritative HHS and CBP sources before "
        "operational use.</span>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
