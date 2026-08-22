"""Core analytics library for the HHS UAC capacity analytics system.

The package exposes a stable, documented import surface while loading modules
only when their symbols are requested. Lazy loading keeps lightweight commands
from paying the import cost of pandas, Plotly, and report-generation modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "1.0.0"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Feature engineering.
    "FeatureEngineeringError": (
        "src.feature_engineering",
        "FeatureEngineeringError",
    ),
    "FeatureEngineeringConfig": (
        "src.feature_engineering",
        "FeatureEngineeringConfig",
    ),
    "FeatureEngineeringResult": (
        "src.feature_engineering",
        "FeatureEngineeringResult",
    ),
    "CapacityFeatureEngineer": (
        "src.feature_engineering",
        "CapacityFeatureEngineer",
    ),
    "build_feature_matrix": ("src.feature_engineering", "build_feature_matrix"),
    "split_features_and_target": (
        "src.feature_engineering",
        "split_features_and_target",
    ),
    "chronological_train_test_split": (
        "src.feature_engineering",
        "chronological_train_test_split",
    ),
    "feature_manifest": ("src.feature_engineering", "feature_manifest"),
    # KPI calculation.
    "TOTAL_CARE_KEY": ("src.kpi", "TOTAL_CARE_KEY"),
    "NET_PRESSURE_KEY": ("src.kpi", "NET_PRESSURE_KEY"),
    "VOLATILITY_KEY": ("src.kpi", "VOLATILITY_KEY"),
    "BACKLOG_KEY": ("src.kpi", "BACKLOG_KEY"),
    "OFFSET_KEY": ("src.kpi", "OFFSET_KEY"),
    "KPI_KEYS": ("src.kpi", "KPI_KEYS"),
    "KPI_DEFINITIONS": ("src.kpi", "KPI_DEFINITIONS"),
    "KPIError": ("src.kpi", "KPIError"),
    "KPIStatus": ("src.kpi", "KPIStatus"),
    "KPIDirection": ("src.kpi", "KPIDirection"),
    "KPIConfig": ("src.kpi", "KPIConfig"),
    "KPIResult": ("src.kpi", "KPIResult"),
    "KPIAlert": ("src.kpi", "KPIAlert"),
    "KPIDashboardResult": ("src.kpi", "KPIDashboardResult"),
    "CapacityKPICalculator": ("src.kpi", "CapacityKPICalculator"),
    "calculate_kpi_trends": ("src.kpi", "calculate_kpi_trends"),
    "calculate_kpi_dashboard": ("src.kpi", "calculate_kpi_dashboard"),
    "kpi_results_to_dict": ("src.kpi", "kpi_results_to_dict"),
    # Structured logging.
    "ROOT_LOGGER_NAME": ("src.logger", "ROOT_LOGGER_NAME"),
    "REDACTED_VALUE": ("src.logger", "REDACTED_VALUE"),
    "VALID_LEVELS": ("src.logger", "VALID_LEVELS"),
    "LoggerConfigurationError": ("src.logger", "LoggerConfigurationError"),
    "LoggingConfig": ("src.logger", "LoggingConfig"),
    "ContextFilter": ("src.logger", "ContextFilter"),
    "SensitiveDataFilter": ("src.logger", "SensitiveDataFilter"),
    "JSONLogFormatter": ("src.logger", "JSONLogFormatter"),
    "HumanReadableFormatter": ("src.logger", "HumanReadableFormatter"),
    "configure_logging": ("src.logger", "configure_logging"),
    "ContextLoggerAdapter": ("src.logger", "ContextLoggerAdapter"),
    "get_logger": ("src.logger", "get_logger"),
    "new_context_id": ("src.logger", "new_context_id"),
    "log_context": ("src.logger", "log_context"),
    "current_log_context": ("src.logger", "current_log_context"),
    "PerformanceTimer": ("src.logger", "PerformanceTimer"),
    "log_execution": ("src.logger", "log_execution"),
    "log_audit_event": ("src.logger", "log_audit_event"),
    "log_dataframe_profile": ("src.logger", "log_dataframe_profile"),
    # Preprocessing.
    "IS_IMPUTED_DATE_COLUMN": ("src.preprocessor", "IS_IMPUTED_DATE_COLUMN"),
    "HAS_IMPUTED_VALUES_COLUMN": (
        "src.preprocessor",
        "HAS_IMPUTED_VALUES_COLUMN",
    ),
    "IMPUTED_VALUE_COUNT_COLUMN": (
        "src.preprocessor",
        "IMPUTED_VALUE_COUNT_COLUMN",
    ),
    "NEGATIVE_COUNT_ANOMALY_COLUMN": (
        "src.preprocessor",
        "NEGATIVE_COUNT_ANOMALY_COLUMN",
    ),
    "ANY_ANOMALY_COLUMN": ("src.preprocessor", "ANY_ANOMALY_COLUMN"),
    "PreprocessingError": ("src.preprocessor", "PreprocessingError"),
    "IssueSeverity": ("src.preprocessor", "IssueSeverity"),
    "PreprocessingIssue": ("src.preprocessor", "PreprocessingIssue"),
    "PreprocessingReport": ("src.preprocessor", "PreprocessingReport"),
    "PreprocessingConfig": ("src.preprocessor", "PreprocessingConfig"),
    "PreprocessedDataset": ("src.preprocessor", "PreprocessedDataset"),
    "DataSource": ("src.preprocessor", "DataSource"),
    "HHSDataPreprocessor": ("src.preprocessor", "HHSDataPreprocessor"),
    "preprocess_data": ("src.preprocessor", "preprocess_data"),
    "validate_preprocessed_data": (
        "src.preprocessor",
        "validate_preprocessed_data",
    ),
    "preprocessed_to_csv_bytes": (
        "src.preprocessor",
        "preprocessed_to_csv_bytes",
    ),
    # Report generation.
    "ReportGenerationError": (
        "src.report_generator",
        "ReportGenerationError",
    ),
    "ReportFormat": ("src.report_generator", "ReportFormat"),
    "InsightSeverity": ("src.report_generator", "InsightSeverity"),
    "ReportConfig": ("src.report_generator", "ReportConfig"),
    "ReportInsight": ("src.report_generator", "ReportInsight"),
    "CapacityReport": ("src.report_generator", "CapacityReport"),
    "CapacityReportGenerator": (
        "src.report_generator",
        "CapacityReportGenerator",
    ),
    "generate_capacity_report": (
        "src.report_generator",
        "generate_capacity_report",
    ),
    "export_report": ("src.report_generator", "export_report"),
    # Dataset validation.
    "ROW_NUMBER_COLUMN": ("src.validation", "ROW_NUMBER_COLUMN"),
    "ISSUE_CODES_COLUMN": ("src.validation", "ISSUE_CODES_COLUMN"),
    "DatasetValidationError": (
        "src.validation",
        "DatasetValidationError",
    ),
    "ValidationSeverity": ("src.validation", "ValidationSeverity"),
    "ValidationConfig": ("src.validation", "ValidationConfig"),
    "ValidationFinding": ("src.validation", "ValidationFinding"),
    "DatasetValidationReport": (
        "src.validation",
        "DatasetValidationReport",
    ),
    "DatasetValidationResult": (
        "src.validation",
        "DatasetValidationResult",
    ),
    "CapacityDataValidator": ("src.validation", "CapacityDataValidator"),
    "validate_capacity_data": ("src.validation", "validate_capacity_data"),
    "validate_or_raise": ("src.validation", "validate_or_raise"),
    # Plotly visualisation.
    "PLOTLY_RENDER_CONFIG": ("src.visualisation", "PLOTLY_RENDER_CONFIG"),
    "VisualizationError": ("src.visualisation", "VisualizationError"),
    "GovernmentChartTheme": (
        "src.visualisation",
        "GovernmentChartTheme",
    ),
    "VisualizationConfig": ("src.visualisation", "VisualizationConfig"),
    "VisualizationData": ("src.visualisation", "VisualizationData"),
    "DashboardFigures": ("src.visualisation", "DashboardFigures"),
    "prepare_visualization_data": (
        "src.visualisation",
        "prepare_visualization_data",
    ),
    "apply_government_theme": (
        "src.visualisation",
        "apply_government_theme",
    ),
    "create_system_load_chart": (
        "src.visualisation",
        "create_system_load_chart",
    ),
    "create_care_load_comparison_chart": (
        "src.visualisation",
        "create_care_load_comparison_chart",
    ),
    "create_net_intake_backlog_chart": (
        "src.visualisation",
        "create_net_intake_backlog_chart",
    ),
    "create_operational_flow_chart": (
        "src.visualisation",
        "create_operational_flow_chart",
    ),
    "create_growth_volatility_chart": (
        "src.visualisation",
        "create_growth_volatility_chart",
    ),
    "create_capacity_utilization_chart": (
        "src.visualisation",
        "create_capacity_utilization_chart",
    ),
    "create_validation_summary_chart": (
        "src.visualisation",
        "create_validation_summary_chart",
    ),
    "create_dashboard_figures": (
        "src.visualisation",
        "create_dashboard_figures",
    ),
    "figure_to_html_bytes": ("src.visualisation", "figure_to_html_bytes"),
}


def __getattr__(name: str) -> Any:
    """Load and cache a declared public export on first access."""

    export = _LAZY_EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = export
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return package attributes, including lazily exposed names."""

    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = ["__version__", *_LAZY_EXPORTS]
