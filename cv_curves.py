"""Plot training/validation loss & IoU curves across the 5 CV folds.

Loads train_log.csv and valid_log.csv per fold, then produces convergence plots:
  - individual fold curves (thin) + mean across folds (thick) + SD band
  - one figure for loss, one for IoU, plus a combined 2-panel figure
  - best-epoch markers per fold on the IoU plot
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cv_exp_num = "01"  # part of the CV_ROOT path; change to load a different run's results
project = "gg"  # project name either "atto" or "gg"


def configure(project_name: str, exp_num: str):
    """Point the module at a different project / CV run (used by cv_complete_analysis)."""
    global project, cv_exp_num, CV_ROOT, OUT_DIR
    project = project_name
    cv_exp_num = exp_num
    CV_ROOT = Path(
        r"C:\Users\faulhamm\Documents\Philipp\Code\cc-machine-learning\results\02-cc-{0}\cross_val_{1}".format(project, cv_exp_num)
    )
    OUT_DIR = CV_ROOT / "summary"


configure(project, cv_exp_num)  # initialize module-level paths


def selection_column(df: pd.DataFrame) -> str:
    """Validation column the training run used to pick the best model.

    Newer runs select on corpus-wide foreground IoU (`iou_corpus_fg`); older
    runs (no such column) selected on `iou_score`. Best-epoch markers use this
    so they land on the epoch that actually produced best_model.pth.
    """
    return "iou_corpus_fg" if "iou_corpus_fg" in df.columns else "iou_score"


def load_fold_logs():
    folds = []
    for d in sorted(CV_ROOT.iterdir()):
        if not (d.is_dir() and re.search(r"_cv\d+_", d.name)):
            continue
        cv_id = int(re.search(r"_cv(\d+)_", d.name).group(1))
        train = pd.read_csv(d / "train_log.csv", index_col=0)
        valid = pd.read_csv(d / "valid_log.csv", index_col=0)
        folds.append({"cv": cv_id, "train": train, "valid": valid})
    return folds


def stack(folds, split: str, col: str) -> np.ndarray:
    """Returns array of shape (n_folds, n_epochs). Truncated to shortest fold length."""
    arrays = [f[split][col].to_numpy() for f in folds]
    n_min = min(len(a) for a in arrays)
    return np.stack([a[:n_min] for a in arrays], axis=0)


def plot_curves(folds, col: str, ylabel: str, title: str, out_path: Path,
                invert_y: bool = False, mark_best: bool = False):
    train_arr = stack(folds, "train", col)
    valid_arr = stack(folds, "valid", col)
    epochs = np.arange(train_arr.shape[1])

    train_mean = train_arr.mean(axis=0)
    train_std = train_arr.std(axis=0, ddof=1)
    valid_mean = valid_arr.mean(axis=0)
    valid_std = valid_arr.std(axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, f in enumerate(folds):
        ax.plot(f["train"].index, f["train"][col], color="#4C78A8",
                alpha=0.25, linewidth=1,
                label="train (per fold)" if i == 0 else None)
        ax.plot(f["valid"].index, f["valid"][col], color="#E45756",
                alpha=0.25, linewidth=1,
                label="valid (per fold)" if i == 0 else None)

    ax.plot(epochs, train_mean, color="#4C78A8", linewidth=2.2, label="train (mean)")
    ax.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                    color="#4C78A8", alpha=0.18, label="train ±SD")
    ax.plot(epochs, valid_mean, color="#E45756", linewidth=2.2, label="valid (mean)")
    ax.fill_between(epochs, valid_mean - valid_std, valid_mean + valid_std,
                    color="#E45756", alpha=0.18, label="valid ±SD")

    if mark_best:
        for f in folds:
            # Best epoch = where the model-selection signal peaked (matches
            # best_model.pth), marked on whichever curve is being plotted.
            best_ep = int(f["valid"][selection_column(f["valid"])].idxmax())
            best_val = f["valid"][col].loc[best_ep]
            ax.scatter([best_ep], [best_val], marker="*", s=110, color="black",
                       zorder=5, edgecolor="white", linewidth=0.6)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_combined(folds, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    specs = [
        ("DiceCE_loss", "Dice + CE loss", "Loss convergence", False),
        ("iou_score", "IoU", "IoU convergence", True),
    ]
    for ax, (col, ylabel, title, mark_best) in zip(axes, specs):
        train_arr = stack(folds, "train", col)
        valid_arr = stack(folds, "valid", col)
        epochs = np.arange(train_arr.shape[1])

        for i, f in enumerate(folds):
            ax.plot(f["train"].index, f["train"][col], color="#4C78A8",
                    alpha=0.25, linewidth=1,
                    label="train (per fold)" if i == 0 else None)
            ax.plot(f["valid"].index, f["valid"][col], color="#E45756",
                    alpha=0.25, linewidth=1,
                    label="valid (per fold)" if i == 0 else None)

        ax.plot(epochs, train_arr.mean(axis=0), color="#4C78A8",
                linewidth=2.2, label="train (mean)")
        ax.fill_between(epochs,
                        train_arr.mean(axis=0) - train_arr.std(axis=0, ddof=1),
                        train_arr.mean(axis=0) + train_arr.std(axis=0, ddof=1),
                        color="#4C78A8", alpha=0.18)
        ax.plot(epochs, valid_arr.mean(axis=0), color="#E45756",
                linewidth=2.2, label="valid (mean)")
        ax.fill_between(epochs,
                        valid_arr.mean(axis=0) - valid_arr.std(axis=0, ddof=1),
                        valid_arr.mean(axis=0) + valid_arr.std(axis=0, ddof=1),
                        color="#E45756", alpha=0.18)

        if mark_best:
            for f in folds:
                best_ep = int(f["valid"][selection_column(f["valid"])].idxmax())
                best_val = f["valid"][col].loc[best_ep]
                ax.scatter([best_ep], [best_val], marker="*", s=110,
                           color="black", zorder=5, edgecolor="white", linewidth=0.6,
                           label="best epoch" if f["cv"] == 0 else None)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("5-fold CV convergence", y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_convergence_summary(folds, out_path: Path):
    """Quick numeric snapshot of how converged each fold looks."""
    rows = []
    for f in folds:
        v_iou = f["valid"]["iou_score"]
        t_iou = f["train"]["iou_score"]
        best_ep = int(v_iou.idxmax())
        last10_v = v_iou.iloc[-10:]
        last10_t = t_iou.iloc[-10:]
        rows.append({
            "fold": f["cv"],
            "epochs": len(v_iou),
            "best_epoch": best_ep,
            "best_val_iou": v_iou.iloc[best_ep],
            "final_val_iou": v_iou.iloc[-1],
            "val_iou_last10_mean": last10_v.mean(),
            "val_iou_last10_std": last10_v.std(ddof=1),
            "final_train_iou": t_iou.iloc[-1],
            "train_val_gap_final": t_iou.iloc[-1] - v_iou.iloc[-1],
            "train_val_gap_last10": last10_t.mean() - last10_v.mean(),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds = load_fold_logs()
    print(f"Loaded {len(folds)} folds")

    plot_curves(folds, "DiceCE_loss", "Dice + CE loss",
                "Loss convergence (5-fold CV)",
                OUT_DIR / "curves_loss.png")
    plot_curves(folds, "iou_score", "IoU",
                "IoU convergence (5-fold CV)",
                OUT_DIR / "curves_iou.png", mark_best=True)
    plot_combined(folds, OUT_DIR / "curves_combined.png")
    write_convergence_summary(folds, OUT_DIR / "convergence_summary.csv")

    print(f"Wrote curves and summary to: {OUT_DIR}")


if __name__ == "__main__":
    main()
