"""One-stop CV analysis: aggregate metrics, then plot curves and per-class errors.

Runs, in order:
  1. cv_error_analysis.main()  -> writes the summary CSVs (per-class / per-fold / CMs)
  2. cv_curves.main()          -> convergence curves + convergence_summary.csv
  3. cv_error_plots.main()     -> per-class error bars (class-colored) + CM heatmap

Set `project` and `cv_exp_num` below; they are pushed into every sub-module before
its main() runs. The plots depend on the CSVs, so cv_error_analysis must run first.
"""

import cv_error_analysis
import cv_curves
import cv_error_plots

project = "atto"        # "atto" or "cc"
cv_exp_num = "01"     # which CV run to analyze (part of the results path)


def main():
    for mod in (cv_error_analysis, cv_curves, cv_error_plots):
        print(f"\n=== {mod.__name__} ===")
        mod.configure(project, cv_exp_num)
        mod.main()


if __name__ == "__main__":
    main()
