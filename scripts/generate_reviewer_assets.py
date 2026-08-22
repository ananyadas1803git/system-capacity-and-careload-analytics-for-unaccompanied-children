"""Generate deterministic, evidence-backed SVG charts for repository reviewers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "output" / "forecasting"
ASSET_ROOT = PROJECT_ROOT / "docs" / "assets"
NAVY = "#163B65"
BLUE = "#2E75B6"
AMBER = "#D97706"
SLATE = "#66788A"


def _save(figure: plt.Figure, name: str) -> None:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        ASSET_ROOT / name,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "HHS UAC Capacity Analytics"},
    )
    plt.close(figure)


def model_comparison() -> None:
    """Render development and holdout MAE from the frozen metric artifact."""

    payload = json.loads(
        (ARTIFACT_ROOT / "metrics" / "model_comparison_metrics.json").read_text(encoding="utf-8")
    )
    rows = []
    for name, values in payload["models"].items():
        rows.append(
            {
                "model": name.replace("_", " ").title(),
                "cv": values["walk_forward"]["mean_mae"],
                "holdout": values["holdout"]["mae"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("cv", ascending=True)
    positions = range(len(frame))
    figure, axis = plt.subplots(figsize=(11, 6.8))
    axis.barh(
        [value - 0.2 for value in positions],
        frame["cv"],
        0.38,
        label="Walk-forward MAE",
        color=BLUE,
    )
    axis.barh(
        [value + 0.2 for value in positions],
        frame["holdout"],
        0.38,
        label="Frozen holdout MAE",
        color=AMBER,
    )
    axis.set_yticks(list(positions), frame["model"])
    axis.set_xlabel("Mean absolute error (children)")
    axis.set_title("Model comparison from frozen research artifacts", color=NAVY, weight="bold")
    axis.grid(axis="x", color="#D9E1EA", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="lower right")
    figure.text(
        0.01,
        0.005,
        "Selection used development OOF MAE; holdout was a promotion gate.",
        color=SLATE,
        fontsize=9,
    )
    _save(figure, "model_comparison.svg")


def holdout_forecast() -> None:
    """Render actual, champion, baseline, and interval on the frozen holdout."""

    predictions = pd.read_csv(
        ARTIFACT_ROOT / "predictions" / "final_holdout_predictions.csv",
        parse_dates=["target_date"],
    )
    champion = predictions.loc[
        predictions["model_name"].eq("validation_weighted_ensemble")
    ].sort_values("target_date")
    persistence = predictions.loc[predictions["model_name"].eq("persistence")].sort_values(
        "target_date"
    )
    figure, axis = plt.subplots(figsize=(12, 6.2))
    axis.fill_between(
        champion["target_date"],
        champion["lower_interval"],
        champion["upper_interval"],
        color=BLUE,
        alpha=0.14,
        label="Nominal 80% quantile interval",
    )
    axis.plot(
        champion["target_date"], champion["actual_value"], color=NAVY, linewidth=2.2, label="Actual"
    )
    axis.plot(
        champion["target_date"],
        champion["reconstructed_absolute_prediction"],
        color=AMBER,
        linewidth=1.8,
        label="Validation-weighted ensemble",
    )
    axis.plot(
        persistence["target_date"],
        persistence["reconstructed_absolute_prediction"],
        color=SLATE,
        linewidth=1.2,
        linestyle="--",
        label="Persistence",
    )
    axis.set_ylabel("Children under care")
    axis.set_title("Frozen holdout: actual versus seven-day forecast", color=NAVY, weight="bold")
    axis.grid(color="#D9E1EA", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncol=2)
    figure.autofmt_xdate()
    figure.text(
        0.01,
        0.005,
        "Research prototype; source provenance is unknown/unverified.",
        color=SLATE,
        fontsize=9,
    )
    _save(figure, "holdout_actual_vs_predicted.svg")


def main() -> None:
    """Regenerate every reviewer chart from approved artifacts."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "hhs-uac-reviewer-assets-v1",
        }
    )
    model_comparison()
    holdout_forecast()


if __name__ == "__main__":
    main()
