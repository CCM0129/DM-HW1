"""Coffee-quality feature ablations for the team's Ridge / fixed-GroupKFold pipeline.

Install at: DM-HW1-model/src/ablation.py

In WSL, from DM-HW1-model root:
    export DM_FEATURE_REPO=/mnt/c/Users/iweiw/DM-HW1-feature
    python src/ablation.py
    python src/ablation.py --retune-k

Outputs go to results/ablation/. These experiments never open the held-out test set.

NOTE: This script is prepared against the APIs shown in the team's README:
    evaluate.load_train_folds() -> (train, y, folds)
    models.tune(name, X, y, folds, feature_set=..., drop_group=..., k=...)
                          -> (curve, best_params, per_fold_dataframe)
Please run once in the team repository before committing/handing in.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from evaluate import load_train_folds
from models import tune

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "ablation"
K_GRID = (5, 10, 20, 33, 50, 75, 97)


def evaluate_setting(train, y, folds, label, feature_set, drop_group=None, k=None):
    """Run team's unchanged tuning routine for a single feature configuration."""
    _, best, per_fold = tune(
        "ridge", train, y, folds,
        feature_set=feature_set, drop_group=drop_group, k=k,
    )
    pf = per_fold.copy()
    required = {"fold", "MAE", "RMSE", "R2"}
    missing = required - set(pf.columns)
    if missing:
        raise ValueError(f"Unexpected per-fold output; missing {sorted(missing)}")
    pf["setting"] = label
    pf["best_alpha"] = float(best["alpha"])
    summary = {"setting": label, "best_alpha": float(best["alpha"])}
    for metric in ("MAE", "RMSE", "R2"):
        summary[f"{metric}_mean"] = float(pf[metric].mean())
        summary[f"{metric}_sd_sample"] = float(pf[metric].std(ddof=1))
    return summary, pf


def choose_k(train, y, folds):
    """Same CV criterion previously used to select Set B's k; exploratory choice."""
    rows = []
    for k in K_GRID:
        s, _ = evaluate_setting(train, y, folds, f"Set B k={k}", "B", k=k)
        s["k"] = k
        rows.append(s)
        print(f"k={k:>2}  alpha={s['best_alpha']:>9.3f}  MAE={s['MAE_mean']:.6f}", flush=True)
    k_table = pd.DataFrame(rows)
    k_table.sort_values(["MAE_mean", "k"], inplace=True)
    k_table.to_csv(OUTPUT / "set_b_k_search.csv", index=False, encoding="utf-8-sig")
    return int(k_table.iloc[0]["k"])


def main():
    parser = argparse.ArgumentParser(description="Seven Ridge feature-set ablations")
    parser.add_argument(
        "--retune-k", action="store_true",
        help="rerun the Set B k grid (5,10,20,33,50,75,97) before the seven comparisons",
    )
    parser.add_argument("--k", type=int, default=33, help="preselected Set B k, default 33")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    train, y, folds = load_train_folds()
    print(f"Train rows: {len(train)}, folds: {len(np.unique(folds))}")
    selected_k = choose_k(train, y, folds) if args.retune_k else args.k
    print(f"Using Set B k={selected_k}")

    configurations = [
        ("Set A", "A", None, None),
        (f"Set B (k={selected_k})", "B", None, selected_k),
        ("Set C", "C", None, None),
        ("C - polynomial", "C", "polynomial", None),
        ("C - interactions", "C", "interactions", None),
        ("C - transforms", "C", "transforms", None),
        ("C - origin", "C", "origin", None),
    ]

    summaries = []
    per_fold_tables = []
    for label, fs, drop, k in configurations:
        s, pf = evaluate_setting(train, y, folds, label, fs, drop, k)
        summaries.append(s)
        per_fold_tables.append(pf)
        print(f"{label:19} MAE={s['MAE_mean']:.4f} ± {s['MAE_sd_sample']:.4f}", flush=True)

    summary = pd.DataFrame(summaries)
    full_mae = float(summary.loc[summary.setting.eq("Set C"), "MAE_mean"].iloc[0])
    summary["delta_MAE_vs_C"] = summary["MAE_mean"] - full_mae
    summary.to_csv(OUTPUT / "ablation_summary.csv", index=False, encoding="utf-8-sig")

    all_folds = pd.concat(per_fold_tables, ignore_index=True)
    all_folds.to_csv(OUTPUT / "ablation_per_fold.csv", index=False, encoding="utf-8-sig")

    full = all_folds.loc[all_folds.setting.eq("Set C"), ["fold", "MAE"]].set_index("fold")["MAE"]
    tests = []
    for label, _, _, _ in configurations:
        if label == "Set C":
            continue
        current = all_folds.loc[all_folds.setting.eq(label), ["fold", "MAE"]].set_index("fold")["MAE"]
        pair = pd.concat([current.rename("other"), full.rename("full_C")], axis=1).dropna()
        if len(pair) != 5:
            raise ValueError(f"Not five matched folds for {label}")
        diff = pair["other"] - pair["full_C"]
        t_stat, p_value = ttest_rel(pair["other"], pair["full_C"])
        tests.append({
            "setting_vs_C": label,
            "mean_delta_MAE": float(diff.mean()),
            "std_delta_MAE": float(diff.std(ddof=1)),
            "C_better_folds": int((diff > 0).sum()),
            "paired_t_stat": float(t_stat),
            "p_two_sided_exploratory": float(p_value),
        })
    pd.DataFrame(tests).to_csv(OUTPUT / "ablation_paired_tests.csv", index=False, encoding="utf-8-sig")

    print("\nSaved to", OUTPUT)
    print("NOTE: p-values are exploratory; folds share training data and tune/k reuse CV.")
    print("NOTE: C - origin changes encoding to one-hot; it does NOT remove all origin information.")
    print("NOTE: Confirm quakers/cert_body availability and documentation before publication.")


if __name__ == "__main__":
    main()
