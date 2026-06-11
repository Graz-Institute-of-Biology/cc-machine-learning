"""Visualizations for the 5-fold CV per-class error analysis.

Reads CSVs produced by cv_error_analysis.py and writes PNGs to the same summary folder.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SUMMARY_DIR = Path(
    r"C:\Users\faulhamm\Documents\Philipp\Code\cc-machine-learning\results\02-cc-gg\cross_val\summary"
)
METRICS = ["recall", "precision", "iou", "f1"]


def load():
    per_class = pd.read_csv(SUMMARY_DIR / "per_class_metrics.csv")
    per_fold = pd.read_csv(SUMMARY_DIR / "per_fold_metrics.csv")
    cm_mean = pd.read_csv(SUMMARY_DIR / "confusion_matrix_normalized_mean.csv", index_col=0)
    cm_std = pd.read_csv(SUMMARY_DIR / "confusion_matrix_normalized_std.csv", index_col=0)
    return per_class, per_fold, cm_mean, cm_std


def plot_metric_bars(per_class: pd.DataFrame, metric: str, out_path: Path):
    classes = per_class["class"].tolist()
    means = per_class[f"{metric}_mean"].to_numpy()
    stds = per_class[f"{metric}_std"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(classes))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="#4C78A8",
                  edgecolor="black", linewidth=0.5, error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel(metric.upper() if metric == "iou" else metric.capitalize())
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-class {metric.upper() if metric == 'iou' else metric}  (mean ± SD, 5-fold CV)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.01, f"{m:.3f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_iou_with_folds(per_class: pd.DataFrame, per_fold: pd.DataFrame, out_path: Path):
    """IoU bars with individual fold values overlaid as dots."""
    classes = per_class["class"].tolist()
    means = per_class["iou_mean"].to_numpy()
    stds = per_class["iou_std"].to_numpy()
    pooled = per_class["iou_pooled"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(classes))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4C78A8", alpha=0.8,
           edgecolor="black", linewidth=0.5, label="mean ± SD (5 folds)",
           error_kw={"elinewidth": 1.2})

    for i, cls in enumerate(classes):
        vals = per_fold.loc[per_fold["class"] == cls, "iou"].to_numpy()
        jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full_like(vals, i) + jitter, vals, s=24, color="black",
                   zorder=3, alpha=0.7, label="per-fold" if i == 0 else None)

    ax.scatter(x, pooled, marker="D", s=55, color="#E45756", zorder=4,
               edgecolor="black", linewidth=0.6, label="pooled")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class IoU across 5-fold CV")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_all_metrics_grid(per_class: pd.DataFrame, out_path: Path):
    classes = per_class["class"].tolist()
    x = np.arange(len(classes))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, metric in zip(axes.flat, METRICS):
        means = per_class[f"{metric}_mean"].to_numpy()
        stds = per_class[f"{metric}_std"].to_numpy()
        ax.bar(x, means, yerr=stds, capsize=3, color="#4C78A8",
               edgecolor="black", linewidth=0.4, error_kw={"elinewidth": 1.0})
        ax.set_title(metric.upper() if metric == "iou" else metric.capitalize())
        ax.set_ylim(0, 1.05)
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

    for metric in METRICS:
        plot_metric_bars(per_class, metric, SUMMARY_DIR / f"bars_{metric}.png")

    plot_iou_with_folds(per_class, per_fold, SUMMARY_DIR / "iou_with_folds.png")
    plot_all_metrics_grid(per_class, SUMMARY_DIR / "all_metrics_grid.png")
    plot_confusion_heatmap(cm_mean, cm_std, SUMMARY_DIR / "confusion_matrix_mean_std.png")

    print(f"Wrote plots to: {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
