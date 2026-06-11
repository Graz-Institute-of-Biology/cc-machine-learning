"""Aggregate per-class error metrics across 5-fold CV runs for the GG biocrust project.

For each fold, picks the epoch with max validation IoU (matches the saved best_model.pth),
loads that epoch's raw confusion matrix, and computes per-class recall, precision, IoU, F1.
Across folds: mean, std, and 95% CI (t-distribution, df=n-1).
Also produces a pooled CV confusion matrix (sum of raw matrices across folds).
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy import stats

CV_ROOT = Path(r"C:\Users\faulhamm\Documents\Philipp\Code\cc-machine-learning\results\02-cc-gg\cross_val")
OUT_DIR = CV_ROOT / "summary"


def find_fold_dirs(root: Path):
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and re.search(r"_cv\d+_", d.name))
    return dirs


def selection_column(df: pd.DataFrame) -> str:
    """The validation column the training run used to pick the best model / save CMs.

    Newer runs select the best model on corpus-wide foreground IoU
    (`iou_corpus_fg`) and only save a confusion matrix at those best epochs.
    Older runs have no such column and selected on `iou_score` (and saved a CM
    every epoch). Using the matching column guarantees the chosen epoch's
    confusion_matrix_raw_ep*.csv exists and lines up with the saved best_model.pth.
    """
    return "iou_corpus_fg" if "iou_corpus_fg" in df.columns else "iou_score"


def best_epoch(fold_dir: Path) -> int:
    df = pd.read_csv(fold_dir / "valid_log.csv", index_col=0)
    return int(df[selection_column(df)].idxmax())


def load_raw_cm(fold_dir: Path, epoch: int):
    df = pd.read_csv(fold_dir / f"confusion_matrix_raw_ep{epoch}.csv", index_col=0)
    classes = list(df.index)
    cm = df.to_numpy(dtype=np.float64)
    return classes, cm


def per_class_metrics(cm: np.ndarray):
    """Rows = true class, cols = predicted class."""
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)
        precision = np.where((tp + fp) > 0, tp / (tp + fp), np.nan)
        iou = np.where((tp + fn + fp) > 0, tp / (tp + fn + fp), np.nan)
        f1 = np.where((2 * tp + fn + fp) > 0, 2 * tp / (2 * tp + fn + fp), np.nan)

    return {"recall": recall, "precision": precision, "iou": iou, "f1": f1}


def ci95(values: np.ndarray):
    """t-based 95% CI half-width; returns (low, high). NaN if <2 valid values."""
    vals = values[~np.isnan(values)]
    n = len(vals)
    if n < 2:
        return np.nan, np.nan
    mean = vals.mean()
    sem = vals.std(ddof=1) / np.sqrt(n)
    h = sem * stats.t.ppf(0.975, df=n - 1)
    return mean - h, mean + h


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_dirs = find_fold_dirs(CV_ROOT)
    print(f"Found {len(fold_dirs)} fold(s):")
    for d in fold_dirs:
        print(f"  {d.name}")

    classes_ref = None
    pooled_cm = None
    per_fold_metrics = {"recall": [], "precision": [], "iou": [], "f1": []}
    per_fold_rows = []

    for d in fold_dirs:
        ep = best_epoch(d)
        classes, cm = load_raw_cm(d, ep)
        if classes_ref is None:
            classes_ref = classes
            pooled_cm = np.zeros_like(cm)
        else:
            assert classes == classes_ref, f"Class order mismatch in {d.name}"
        pooled_cm += cm

        m = per_class_metrics(cm)
        for k, v in m.items():
            per_fold_metrics[k].append(v)

        fold_id = re.search(r"_cv(\d+)_", d.name).group(1)
        vdf = pd.read_csv(d / "valid_log.csv", index_col=0)
        val_iou = float(vdf[selection_column(vdf)].max())
        print(f"  cv{fold_id}: best_epoch={ep}, val_iou={val_iou:.4f}")
        for ci, cls in enumerate(classes):
            per_fold_rows.append({
                "fold": int(fold_id),
                "best_epoch": ep,
                "val_iou": val_iou,
                "class": cls,
                "recall": m["recall"][ci],
                "precision": m["precision"][ci],
                "iou": m["iou"][ci],
                "f1": m["f1"][ci],
            })

    # Per-fold raw values
    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_df.to_csv(OUT_DIR / "per_fold_metrics.csv", index=False)

    # Pooled CM metrics (treating all folds as one big confusion matrix)
    pooled_metrics = per_class_metrics(pooled_cm)

    # Summary across folds (mean / std / 95% CI / pooled)
    summary_rows = []
    for ci, cls in enumerate(classes_ref):
        row = {"class": cls}
        for metric in ["recall", "precision", "iou", "f1"]:
            vals = np.array([fm[ci] for fm in per_fold_metrics[metric]], dtype=np.float64)
            valid = vals[~np.isnan(vals)]
            row[f"{metric}_mean"] = valid.mean() if len(valid) else np.nan
            row[f"{metric}_std"] = valid.std(ddof=1) if len(valid) > 1 else np.nan
            lo, hi = ci95(vals)
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
            row[f"{metric}_pooled"] = pooled_metrics[metric][ci]
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "per_class_metrics.csv", index=False)

    # Headline IoU table (compact, paper-ready)
    headline = summary_df[[
        "class", "iou_mean", "iou_std",
        "iou_ci95_low", "iou_ci95_high", "iou_pooled",
    ]].rename(columns={
        "iou_mean": "mean",
        "iou_std": "std",
        "iou_ci95_low": "ci95_low",
        "iou_ci95_high": "ci95_high",
        "iou_pooled": "pooled",
    })
    headline.to_csv(OUT_DIR / "headline_iou.csv", index=False)

    # Pooled confusion matrix (raw counts and row-normalized)
    pooled_raw_df = pd.DataFrame(pooled_cm, index=classes_ref, columns=classes_ref)
    pooled_raw_df.to_csv(OUT_DIR / "pooled_confusion_matrix_raw.csv")

    row_sums = pooled_cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled_norm = np.where(row_sums > 0, pooled_cm / row_sums, 0.0)
    pooled_norm_df = pd.DataFrame(pooled_norm, index=classes_ref, columns=classes_ref)
    pooled_norm_df.to_csv(OUT_DIR / "pooled_confusion_matrix_normalized.csv")

    # Per-cell std across folds for normalized matrices (each fold normalized first)
    norm_stack = []
    for fm_idx, d in enumerate(fold_dirs):
        ep = best_epoch(d)
        _, cm = load_raw_cm(d, ep)
        rs = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm_stack.append(np.where(rs > 0, cm / rs, 0.0))
    norm_stack = np.stack(norm_stack, axis=0)
    cell_mean = norm_stack.mean(axis=0)
    cell_std = norm_stack.std(axis=0, ddof=1)
    pd.DataFrame(cell_mean, index=classes_ref, columns=classes_ref).to_csv(
        OUT_DIR / "confusion_matrix_normalized_mean.csv"
    )
    pd.DataFrame(cell_std, index=classes_ref, columns=classes_ref).to_csv(
        OUT_DIR / "confusion_matrix_normalized_std.csv"
    )

    print(f"\nWrote outputs to: {OUT_DIR}")
    print("\nPer-class summary (mean ± std | 95% CI | pooled):")
    for _, r in summary_df.iterrows():
        print(
            f"  {r['class']:20s}  "
            f"IoU={r['iou_mean']:.4f} ± {r['iou_std']:.4f}  "
            f"CI95=[{r['iou_ci95_low']:.4f}, {r['iou_ci95_high']:.4f}]  "
            f"pooled={r['iou_pooled']:.4f}"
        )


if __name__ == "__main__":
    main()
