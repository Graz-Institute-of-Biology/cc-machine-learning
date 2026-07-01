"""Post-hoc class-merge metrics for a finished CV run — no retraining.

Collapses chosen classes together in each fold's best-epoch raw confusion matrix
(A<->B confusions become internal, i.e. correct), then recomputes the same
per-class / per-fold / pooled metrics and confusion matrices as cv_error_analysis,
writing them to a separate `summary_merged/` folder.

IMPORTANT: this is a *lower bound* on what a model trained on the merged labels
would score. It reuses the predictions the model already made, so it cannot show
the gain from no longer having to draw the (unlearnable) within-group boundary,
nor the reweighting of sampler/loss when class frequencies change. Use it to decide
whether a merge is worth a rerun; report the rerun number, not this one.

Edit MERGE below (new_name -> members). Classes not listed are kept unchanged.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

import cv_error_analysis as base

project = "atto"     # "atto" or "gg"
cv_exp_num = "01"    # which CV run to analyze

# new merged-class name -> original classes folded into it.
# Anything not mentioned here passes through unchanged.
MERGE = {
    "cyanos": ["cyanosliverwort", "cyanosmoss", "cyanosbark"],
}


def configure(project_name: str, exp_num: str):
    """Point at a project / CV run and align cv_error_analysis's paths with ours."""
    global project, cv_exp_num, CV_ROOT, OUT_DIR
    project = project_name
    cv_exp_num = exp_num
    base.configure(project_name, exp_num)  # sets base.CV_ROOT / base.OUT_DIR
    CV_ROOT = base.CV_ROOT
    OUT_DIR = CV_ROOT / "summary_merged"


configure(project, cv_exp_num)


def build_grouping(classes, merge):
    """Return (new_classes, G) where G is (n_old x n_new) 0/1 group-membership.

    A merged class takes the position of its first-listed member so column order
    stays stable and readable. Members must exist in `classes`.
    """
    member_to_group = {}
    for new_name, members in merge.items():
        for m in members:
            if m not in classes:
                raise ValueError(f"merge member {m!r} not in classes {classes}")
            member_to_group[m] = new_name

    new_classes = []
    for c in classes:                       # preserve original ordering
        label = member_to_group.get(c, c)
        if label not in new_classes:
            new_classes.append(label)

    new_index = {name: i for i, name in enumerate(new_classes)}
    G = np.zeros((len(classes), len(new_classes)), dtype=np.float64)
    for i, c in enumerate(classes):
        G[i, new_index[member_to_group.get(c, c)]] = 1.0
    return new_classes, G


def collapse(cm, G):
    """new_cm[a,b] = sum of cm[i,j] over old i in group a, old j in group b."""
    return G.T @ cm @ G


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_dirs = base.find_fold_dirs(CV_ROOT)
    print(f"Found {len(fold_dirs)} fold(s); merging: {MERGE}")

    new_classes = None
    G = None
    pooled_cm = None
    per_fold_metrics = {"recall": [], "precision": [], "iou": [], "f1": []}
    per_fold_rows = []
    norm_stack = []

    for d in fold_dirs:
        ep = base.best_epoch(d)
        classes, cm = base.load_raw_cm(d, ep)
        if G is None:
            new_classes, G = build_grouping(classes, MERGE)
            pooled_cm = np.zeros((len(new_classes), len(new_classes)))
        cm = collapse(cm, G)
        pooled_cm += cm

        m = base.per_class_metrics(cm)
        for k, v in m.items():
            per_fold_metrics[k].append(v)

        rs = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm_stack.append(np.where(rs > 0, cm / rs, 0.0))

        fold_id = int(re.search(r"_cv(\d+)_", d.name).group(1))
        vdf = pd.read_csv(d / "valid_log.csv", index_col=0)
        val_iou = float(vdf[base.selection_column(vdf)].max())
        print(f"  cv{fold_id}: best_epoch={ep}, val_iou={val_iou:.4f}")
        for ci, cls in enumerate(new_classes):
            per_fold_rows.append({
                "fold": fold_id, "best_epoch": ep, "val_iou": val_iou, "class": cls,
                "recall": m["recall"][ci], "precision": m["precision"][ci],
                "iou": m["iou"][ci], "f1": m["f1"][ci],
            })

    pd.DataFrame(per_fold_rows).to_csv(OUT_DIR / "per_fold_metrics.csv", index=False)

    pooled_metrics = base.per_class_metrics(pooled_cm)
    summary_rows = []
    for ci, cls in enumerate(new_classes):
        row = {"class": cls}
        for metric in ["recall", "precision", "iou", "f1"]:
            vals = np.array([fm[ci] for fm in per_fold_metrics[metric]], dtype=np.float64)
            valid = vals[~np.isnan(vals)]
            row[f"{metric}_mean"] = valid.mean() if len(valid) else np.nan
            row[f"{metric}_std"] = valid.std(ddof=1) if len(valid) > 1 else np.nan
            lo, hi = base.ci95(vals)
            row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"] = lo, hi
            row[f"{metric}_pooled"] = pooled_metrics[metric][ci]
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "per_class_metrics.csv", index=False)

    summary_df[[
        "class", "iou_mean", "iou_std", "iou_ci95_low", "iou_ci95_high", "iou_pooled",
    ]].rename(columns={
        "iou_mean": "mean", "iou_std": "std",
        "iou_ci95_low": "ci95_low", "iou_ci95_high": "ci95_high", "iou_pooled": "pooled",
    }).to_csv(OUT_DIR / "headline_iou.csv", index=False)

    pd.DataFrame(pooled_cm, index=new_classes, columns=new_classes).to_csv(
        OUT_DIR / "pooled_confusion_matrix_raw.csv")
    rs = pooled_cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled_norm = np.where(rs > 0, pooled_cm / rs, 0.0)
    pd.DataFrame(pooled_norm, index=new_classes, columns=new_classes).to_csv(
        OUT_DIR / "pooled_confusion_matrix_normalized.csv")

    norm_stack = np.stack(norm_stack, axis=0)
    pd.DataFrame(norm_stack.mean(axis=0), index=new_classes, columns=new_classes).to_csv(
        OUT_DIR / "confusion_matrix_normalized_mean.csv")
    pd.DataFrame(norm_stack.std(axis=0, ddof=1), index=new_classes, columns=new_classes).to_csv(
        OUT_DIR / "confusion_matrix_normalized_std.csv")

    print(f"\nWrote merged outputs to: {OUT_DIR}")
    print("\nMerged per-class summary (mean ± std | 95% CI | pooled):")
    for _, r in summary_df.iterrows():
        print(
            f"  {r['class']:20s}  IoU={r['iou_mean']:.4f} ± {r['iou_std']:.4f}  "
            f"CI95=[{r['iou_ci95_low']:.4f}, {r['iou_ci95_high']:.4f}]  "
            f"pooled={r['iou_pooled']:.4f}"
        )


if __name__ == "__main__":
    main()
