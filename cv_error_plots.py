"""Visualizations for the 5-fold CV per-class error analysis.

Reads CSVs produced by cv_error_analysis.py and writes PNGs to the same summary folder.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cv_exp_num = "01"  # part of the CV_ROOT path; change to load a different run's results
project = "gg"  # project name either "atto" or "gg"

REPO_DIR = Path(__file__).resolve().parent
METRICS = ["recall", "precision", "iou", "f1"]
DEFAULT_COLOR = "#4C78A8"


def configure(project_name: str, exp_num: str, summary_name: str = "summary"):
    """Point the module at a different project / CV run (used by cv_complete_analysis).

    `summary_name` selects the output subfolder — e.g. "summary_merged" to plot the
    post-hoc merged-class CSVs written by cv_merge_classes.py.
    """
    global project, cv_exp_num, SUMMARY_DIR
    project = project_name
    cv_exp_num = exp_num
    SUMMARY_DIR = Path(
        r"C:\Users\faulhamm\Documents\Philipp\Code\cc-machine-learning\results\{0}\cross_val_{1}\{2}".format(project, cv_exp_num, summary_name)
    )


configure(project, cv_exp_num)  # initialize module-level paths


def load_class_colors() -> dict:
    """Map class name -> hex color from ontology_{project}.json."""
    with open(REPO_DIR / f"ontology_{project}.json") as f:
        ontology = json.load(f)["ontology"]
    return {name: entry["color"] for name, entry in ontology.items()}


def colors_for(classes, class_colors: dict) -> list:
    """Color per class in order, falling back to the default for unknown names."""
    return [class_colors.get(cls, DEFAULT_COLOR) for cls in classes]


def load():
    per_class = pd.read_csv(SUMMARY_DIR / "per_class_metrics.csv")
    per_fold = pd.read_csv(SUMMARY_DIR / "per_fold_metrics.csv")
    cm_mean = pd.read_csv(SUMMARY_DIR / "confusion_matrix_normalized_mean.csv", index_col=0)
    cm_std = pd.read_csv(SUMMARY_DIR / "confusion_matrix_normalized_std.csv", index_col=0)
    return per_class, per_fold, cm_mean, cm_std


def plot_metric_bars(per_class: pd.DataFrame, metric: str, out_path: Path,
                     class_colors: dict):
    classes = per_class["class"].tolist()
    means = per_class[f"{metric}_mean"].to_numpy()
    stds = per_class[f"{metric}_std"].to_numpy()

    is_iou = metric == "iou"
    if is_iou:
        means = means * 100
        stds = stds * 100
        ylabel = "IoU (%)"
        ylim = (0, 105)
        label_fmt = "{:.1f}"
        label_offset = 0.5
    else:
        ylabel = metric.capitalize()
        ylim = (0, 1.05)
        label_fmt = "{:.3f}"
        label_offset = 0.01

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(classes))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors_for(classes, class_colors),
                  edgecolor="black", linewidth=0.5, error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_title(f"Per-class {ylabel if is_iou else metric}  (mean ± SD, 5-fold CV)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + label_offset, label_fmt.format(m),
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_iou_with_folds(per_class: pd.DataFrame, per_fold: pd.DataFrame, out_path: Path,
                        class_colors: dict):
    """IoU bars with individual fold values overlaid as dots."""
    classes = per_class["class"].tolist()
    means = per_class["iou_mean"].to_numpy() * 100
    stds = per_class["iou_std"].to_numpy() * 100
    pooled = per_class["iou_pooled"].to_numpy() * 100

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(classes))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors_for(classes, class_colors), alpha=0.8,
           edgecolor="black", linewidth=0.5, label="mean ± SD (5 folds)",
           error_kw={"elinewidth": 1.2})

    for i, cls in enumerate(classes):
        vals = per_fold.loc[per_fold["class"] == cls, "iou"].to_numpy() * 100
        jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full_like(vals, i) + jitter, vals, s=24, color="black",
                   zorder=3, alpha=0.7, label="per-fold" if i == 0 else None)

    ax.scatter(x, pooled, marker="D", s=55, color="#E45756", zorder=4,
               edgecolor="black", linewidth=0.6, label="pooled")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("IoU (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-class IoU across 5-fold CV")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_all_metrics_grid(per_class: pd.DataFrame, out_path: Path, class_colors: dict):
    classes = per_class["class"].tolist()
    x = np.arange(len(classes))
    bar_colors = colors_for(classes, class_colors)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, metric in zip(axes.flat, METRICS):
        means = per_class[f"{metric}_mean"].to_numpy()
        stds = per_class[f"{metric}_std"].to_numpy()
        if metric == "iou":
            means = means * 100
            stds = stds * 100
            ax.set_title("IoU (%)")
            ax.set_ylim(0, 105)
        else:
            ax.set_title(metric.capitalize())
            ax.set_ylim(0, 1.05)
        ax.bar(x, means, yerr=stds, capsize=3, color=bar_colors,
               edgecolor="black", linewidth=0.4, error_kw={"elinewidth": 1.0})
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=30, ha="right")
    fig.suptitle("Per-class metrics (mean ± SD, 5-fold CV)", y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_heatmap(cm_mean: pd.DataFrame, cm_std: pd.DataFrame, out_path: Path):
    classes = cm_mean.index.tolist()
    M = cm_mean.to_numpy()
    S = cm_std.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Mean normalized confusion matrix (± SD across 5 folds)")

    threshold = 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if M[i, j] > threshold else "black"
            ax.text(j, i, f"{M[i, j]:.3f}\n±{S[i, j]:.3f}",
                    ha="center", va="center", color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (row-normalized)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    np.random.seed(0)
    per_class, per_fold, cm_mean, cm_std = load()
    class_colors = load_class_colors()

    for metric in METRICS:
        plot_metric_bars(per_class, metric, SUMMARY_DIR / f"bars_{metric}.png", class_colors)

    plot_iou_with_folds(per_class, per_fold, SUMMARY_DIR / "iou_with_folds.png", class_colors)
    plot_all_metrics_grid(per_class, SUMMARY_DIR / "all_metrics_grid.png", class_colors)
    plot_confusion_heatmap(cm_mean, cm_std, SUMMARY_DIR / "confusion_matrix_mean_std.png")

    print(f"Wrote plots to: {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
