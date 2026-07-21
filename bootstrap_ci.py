"""
Bootstrap confidence intervals for test-set classification metrics.

Expects a CSV with columns:
    labels        - ground-truth class index
    predictions   - predicted class index
    score_0..N    - per-class scores, needed for AUC-ROC

Usage:
    python bootstrap_ci.py --csv results.csv --n_boot 1000
    python bootstrap_ci.py --csv results.csv --n_boot 1000 --classmap classmap.json
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def bootstrap_ci(arrays, metric_fn, n_boot=1000, ci=95, seed=0):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    arrays = [np.asarray(a) for a in arrays]

    scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        resampled = [a[idx] for a in arrays]
        scores[b] = metric_fn(*resampled)

    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    point = metric_fn(*arrays)
    return point, lower, upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to predictions CSV")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labels_col", default="labels")
    parser.add_argument("--preds_col", default="predictions")
    parser.add_argument("--score_prefix", default="score_")
    args = parser.parse_args()

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log(f"csv: {args.csv}")

    df = pd.read_csv(args.csv)
    y_true = df[args.labels_col].to_numpy()
    y_pred = df[args.preds_col].to_numpy()
    classes = sorted(pd.unique(y_true))

    score_cols = [f"{args.score_prefix}{c}" for c in classes]
    missing = [c for c in score_cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing score columns for AUC-ROC: {missing}")
    y_score = df[score_cols].to_numpy()

    classmap = None
    if args.classmap is not None:
        with open(args.classmap, "r") as f:
            raw_map = json.load(f)
        classmap = {int(k): v for k, v in raw_map.items()}

    def acc_macro(yt, yp, ys):
        return balanced_accuracy_score(yt, yp)

    def acc_micro(yt, yp, ys):
        return accuracy_score(yt, yp)

    def f1_macro(yt, yp, ys):
        return f1_score(yt, yp, labels=classes, average="macro", zero_division=0)

    def auc_macro(yt, yp, ys):
        return roc_auc_score(yt, ys, multi_class="ovr", average="macro", labels=classes)

    overall_metrics = {
        "Accuracy (Macro)": acc_macro,
        "Accuracy (Micro)": acc_micro,
        "F1 Score (Macro)": f1_macro,
        "AUC-ROC (Macro)": auc_macro,
    }

    log(f"n = {len(y_true)}, bootstrap iters = {args.n_boot}, CI = {args.ci}%")

    log("")
    results = {}
    for name, fn in overall_metrics.items():
        point, lo, hi = bootstrap_ci((y_true, y_pred, y_score), fn, n_boot=args.n_boot, ci=args.ci, seed=args.seed)
        results[name] = (point, lo, hi)
        log(f"{name}: {point:.4f} ({lo:.4f} - {hi:.4f})")

    out_dir = os.path.dirname(os.path.abspath(args.csv))
    csv_stem = os.path.splitext(os.path.basename(args.csv))[0]

    summary_csv_path = os.path.join(out_dir, f"{csv_stem}_bootstrap_ci.csv")
    rows = [{"metric": k, "point": v[0], "ci_lower": v[1], "ci_upper": v[2]} for k, v in results.items()]
    pd.DataFrame(rows).to_csv(summary_csv_path, index=False)
    log(f"saved metric summary to {summary_csv_path}")

    txt_path = os.path.join(out_dir, f"{csv_stem}_bootstrap_results.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"saved full log to {txt_path}")


if __name__ == "__main__":
    main()