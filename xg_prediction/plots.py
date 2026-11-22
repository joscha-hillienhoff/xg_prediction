"""Figures for the README: calibration of the out-of-fold xG and a shot map coloured by xG."""

from pathlib import Path

from loguru import logger
import matplotlib

matplotlib.use("Agg")
from matplotlib import patches
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
import typer

from xg_prediction.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

TARGET = "is_goal"
OOF_COLUMN = "xg"
CALIBRATION_MAX = 0.6  # axis limit; almost all shots have a predicted xG below this


def plot_calibration(y_true: pd.Series, xg: pd.Series, path: Path, n_bins: int = 8) -> None:
    """Reliability diagram (quantile bins) with the distribution of predicted xG below it."""
    frac_goals, mean_xg = calibration_curve(y_true, xg, n_bins=n_bins, strategy="quantile")

    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(5, 6), height_ratios=[3, 1], sharex=True, constrained_layout=True
    )
    ax.plot(
        [0, CALIBRATION_MAX],
        [0, CALIBRATION_MAX],
        linestyle="--",
        color="grey",
        label="perfectly calibrated",
    )
    ax.plot(mean_xg, frac_goals, marker="o", color="tab:blue", label="out-of-fold xG")
    ax.set_ylabel("Observed goal rate")
    ax.set_title("Calibration of out-of-fold xG")
    ax.legend(loc="upper left")
    ax.set_xlim(0, CALIBRATION_MAX)
    ax.set_ylim(0, CALIBRATION_MAX)

    ax_hist.hist(xg.clip(upper=CALIBRATION_MAX), bins=30, range=(0, CALIBRATION_MAX))
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel(f"Predicted xG (values above {CALIBRATION_MAX} in the last bin)")
    ax_hist.set_ylabel("Shots")

    fig.savefig(path, dpi=110)
    plt.close(fig)


def _draw_half_pitch(ax) -> None:
    """Draw the attacking half of a StatsBomb pitch (120 x 80, attacking towards x = 120)."""
    line = {"fill": False, "edgecolor": "black", "linewidth": 1}
    ax.add_patch(patches.Rectangle((60, 0), 60, 80, **line))
    ax.add_patch(patches.Rectangle((102, 18), 18, 44, **line))  # penalty area
    ax.add_patch(patches.Rectangle((114, 30), 6, 20, **line))  # six-yard box
    ax.add_patch(patches.Rectangle((120, 36), 1.5, 8, **line))  # goal
    ax.plot(108, 40, marker="o", markersize=2, color="black")  # penalty spot
    ax.set_xlim(58, 123)
    ax.set_ylim(-2, 82)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_shot_map(shots: pd.DataFrame, xg: pd.Series, path: Path) -> None:
    """Shot map on the attacking half, coloured by predicted xG; goals are ringed."""
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    _draw_half_pitch(ax)

    points = ax.scatter(
        shots["x"], shots["y"], c=xg, cmap="viridis", vmin=0, vmax=0.5, s=22, alpha=0.85, zorder=3
    )
    goals = shots[shots[TARGET] == 1]
    ax.scatter(
        goals["x"],
        goals["y"],
        facecolors="none",
        edgecolors="red",
        s=70,
        linewidths=1.2,
        label="goal",
        zorder=4,
    )
    ax.invert_yaxis()  # StatsBomb y runs top to bottom
    ax.legend(loc="lower left")
    ax.set_title("Shot map coloured by out-of-fold xG")
    fig.colorbar(points, ax=ax, shrink=0.7, extend="max", label="Predicted xG")

    fig.savefig(path, dpi=110)
    plt.close(fig)


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "train_feat.csv",
    oof_path: Path = PROCESSED_DATA_DIR / "oof_predictions.csv",
    output_dir: Path = FIGURES_DIR,
):
    """Write `calibration.png` and `shot_map.png` from the out-of-fold CV predictions."""
    shots = pd.read_csv(features_path)
    xg = pd.read_csv(oof_path, index_col=0)[OOF_COLUMN].reindex(shots.index)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_calibration(shots[TARGET], xg, output_dir / "calibration.png")
    plot_shot_map(shots, xg, output_dir / "shot_map.png")
    logger.success(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    app()
