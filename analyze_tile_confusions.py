"""
Quick analysis of a tile_confusions_ep{N}.csv file.

CSV schema:
    tile, true_class, pred_class, err_pixels, true_pixels_in_tile,
    err_fraction, image_path, mask_path

A row is one tile listed under a (true_class -> pred_class) confusion pair.
err_pixels        = pixels in this tile whose true label is `true_class`
                    but the model predicted `pred_class`.
true_pixels_in_tile = total pixels of `true_class` in this tile.
err_fraction      = err_pixels / true_pixels_in_tile.
                    1.0 means the entire labeled region for that class was
                    classified as the other class -> "catastrophic" tile,
                    a prime audit target.

Each (true, pred) pair is reported with at most 20 tiles (worst-N export).

Usage:
    python analyze_tile_confusions.py tile_confusions_ep42.csv
    python analyze_tile_confusions.py tile_confusions_ep42.csv \
        --true cyanosbark --pred cyanosmoss
    python analyze_tile_confusions.py tile_confusions_ep42.csv \
        --true cyanosbark --pred cyanosmoss,cyanosliverwort \
        --min-fraction 0.9 --min-pixels 20000 --out audit_targets.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {
        "tile", "true_class", "pred_class", "err_pixels",
        "true_pixels_in_tile", "err_fraction", "image_path", "mask_path",
    }
    missing = expected - set(df.columns)
    if missing:
        sys.exit(f"CSV missing columns: {missing}")
    return df


def pair_summary(df: pd.DataFrame, top: int = 25) -> pd.DataFrame:
    agg = (
        df.groupby(["true_class", "pred_class"])
          .agg(
              n_tiles=("tile", "count"),
              err_px_sum=("err_pixels", "sum"),
              err_frac_mean=("err_fraction", "mean"),
              err_frac_max=("err_fraction", "max"),
              n_catastrophic=("err_fraction", lambda s: (s >= 0.9).sum()),
          )
          .reset_index()
          .sort_values("err_px_sum", ascending=False)
    )
    return agg.head(top)


def filter_confusion(
    df: pd.DataFrame,
    true_class: str,
    pred_classes: list[str],
    min_fraction: float = 0.0,
    min_pixels: int = 0,
) -> pd.DataFrame:
    mask = (
        (df.true_class == true_class)
        & (df.pred_class.isin(pred_classes))
        & (df.err_fraction >= min_fraction)
        & (df.err_pixels >= min_pixels)
    )
    return (
        df.loc[mask]
          .sort_values(["err_fraction", "err_pixels"], ascending=False)
          .reset_index(drop=True)
    )


def image_stem_counts(rows: pd.DataFrame) -> pd.DataFrame:
    """Group flagged tiles by source image stem (drops `_part_N.JPG`)."""
    stems = rows.tile.str.replace(r"_part_\d+\.JPG$", "", regex=True)
    return (
        stems.value_counts()
             .rename_axis("image_stem")
             .reset_index(name="n_flagged_tiles")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--true", dest="true_class", default=None,
                    help="filter on this true_class")
    ap.add_argument("--pred", dest="pred_classes", default=None,
                    help="comma-separated pred_class values to filter on")
    ap.add_argument("--min-fraction", type=float, default=0.0)
    ap.add_argument("--min-pixels", type=int, default=0)
    ap.add_argument("--top-pairs", type=int, default=25)
    ap.add_argument("--audit-top", type=int, default=0,
                    help="for the top-N (true->pred) pairs (ranked by "
                         "catastrophic-tile count, tiebreak err_px_sum), "
                         "print the filtered tile list and source-image "
                         "stem counts. Honors --min-fraction / --min-pixels.")
    ap.add_argument("--audit-min-catastrophic", type=int, default=0,
                    help="alternative to --audit-top: audit every pair that "
                         "has at least this many catastrophic tiles "
                         "(err_fraction >= 0.9).")
    ap.add_argument("--out", type=Path, default=None,
                    help="write filtered tiles to this CSV (single --true/--pred "
                         "mode), or all audited tiles concatenated (audit mode).")
    args = ap.parse_args()

    df = load(args.csv)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)

    print(f"loaded {len(df)} rows from {args.csv}")
    print(f"true_class values: {sorted(df.true_class.unique())}")
    print(f"pred_class values: {sorted(df.pred_class.unique())}\n")

    print(f"=== top {args.top_pairs} (true -> pred) pairs by summed err_pixels ===")
    print(pair_summary(df, args.top_pairs).to_string(index=False))
    print()

    if args.audit_top or args.audit_min_catastrophic:
        full = (
            df.groupby(["true_class", "pred_class"])
              .agg(
                  n_tiles=("tile", "count"),
                  err_px_sum=("err_pixels", "sum"),
                  n_catastrophic=("err_fraction", lambda s: (s >= 0.9).sum()),
              )
              .reset_index()
              .sort_values(["n_catastrophic", "err_px_sum"], ascending=False)
        )
        if args.audit_min_catastrophic:
            ranked = full[full.n_catastrophic >= args.audit_min_catastrophic]
        else:
            ranked = full.head(args.audit_top)

        print(f"=== auditing {len(ranked)} pair(s) "
              f"(ranked by n_catastrophic, then err_px_sum) ===\n")
        collected = []
        for _, r in ranked.iterrows():
            t, p = r.true_class, r.pred_class
            sub = filter_confusion(
                df, t, [p],
                min_fraction=args.min_fraction, min_pixels=args.min_pixels,
            )
            header = (f"### {t} -> {p}  "
                      f"(n_catastrophic={r.n_catastrophic}, "
                      f"err_px_sum={int(r.err_px_sum):,}, "
                      f"matching_tiles={len(sub)})")
            print(header)
            if sub.empty:
                print("  (no tiles meet the filters)\n")
                continue
            cols = ["tile", "err_pixels", "true_pixels_in_tile", "err_fraction"]
            print(sub[cols].to_string(index=False))
            print()
            print("  source-image stems:")
            stems = image_stem_counts(sub)
            print(stems.to_string(index=False))
            print()
            collected.append(sub.assign(true_class=t, pred_class=p))

        if args.out and collected:
            pd.concat(collected, ignore_index=True).to_csv(args.out, index=False)
            print(f"wrote {sum(len(s) for s in collected)} rows -> {args.out}")
        return

    if args.true_class and args.pred_classes:
        preds = [p.strip() for p in args.pred_classes.split(",")]
        sub = filter_confusion(
            df, args.true_class, preds,
            min_fraction=args.min_fraction, min_pixels=args.min_pixels,
        )
        print(f"=== {args.true_class} -> {preds} "
              f"(err_fraction >= {args.min_fraction}, "
              f"err_pixels >= {args.min_pixels}) ===")
        cols = ["tile", "pred_class", "err_pixels",
                "true_pixels_in_tile", "err_fraction"]
        print(sub[cols].to_string(index=False))
        print()

        if not sub.empty:
            print("=== source-image stems contributing these tiles ===")
            print(image_stem_counts(sub).to_string(index=False))
            print()

        if args.out:
            sub.to_csv(args.out, index=False)
            print(f"wrote {len(sub)} rows -> {args.out}")


if __name__ == "__main__":
    main()
