"""主模型(§5.2):Ridge 與 Elastic Net 搭配 Set C,在固定折上調 α 與 L1 比例,輸出調參曲線。
組員 4 的消融請呼叫 tune(),只換 feature_set / drop_group,調參流程即與主模型完全相同。
直接執行輸出 results/tuning_*.csv、best_params.json 與 figures/tuning_*.png。"""
import itertools
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline

from evaluate import FIGURES, RANDOM_SEED, RESULTS, cross_validate, load_train_folds, summarize
from features import build_features

GRIDS = {   # 搜尋範圍(寫進論文 §5.2)
    "ridge": {"alpha": np.logspace(-2, 4, 25)},
    "enet": {"alpha": np.logspace(-4, 1, 21), "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]},
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # 固定順序的類別色


def make_model(name, feature_set="C", drop_group=None, k=None, **params):
    """回傳 特徵 Pipeline + 線性模型;name 為 'ridge' 或 'enet'。"""
    fe = build_features(feature_set, drop_group=drop_group, k=k)
    if name == "ridge":
        return make_pipeline(fe, Ridge(**params))
    if name == "enet":
        return make_pipeline(fe, ElasticNet(max_iter=50_000, random_state=RANDOM_SEED, **params))
    raise ValueError(f"未知模型 {name!r}")


def _param_list(grid):
    return [dict(zip(grid, v)) for v in itertools.product(*grid.values())]


def tune(name, X, y, folds, feature_set="C", drop_group=None, k=None, grid=None):
    """在固定折上做網格搜尋,以 CV 平均 MAE 選最佳參數。回傳 (曲線表, 最佳參數 dict, 最佳參數的每折指標表)。"""
    rows, tables = [], {}
    for p in _param_list(grid or GRIDS[name]):
        p = {key: float(v) for key, v in p.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            t = cross_validate(make_model(name, feature_set, drop_group, k, **p), X, y, folds)
        key = tuple(sorted(p.items()))
        tables[key] = t
        rows.append({**p, "MAE_mean": t.MAE.mean(), "MAE_std": t.MAE.std(ddof=1), "RMSE_mean": t.RMSE.mean()})
    curve = pd.DataFrame(rows)
    best = curve.loc[curve["MAE_mean"].idxmin(), list((grid or GRIDS[name]).keys())].astype(float).to_dict()
    return curve, best, tables[tuple(sorted(best.items()))]


def plot_ridge(curve, best):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    a, m, s = curve["alpha"], curve["MAE_mean"], curve["MAE_std"]
    ax.fill_between(a, m - s, m + s, color=COLORS[0], alpha=0.15, linewidth=0, label="± 1 SD across folds")
    ax.plot(a, m, color=COLORS[0], linewidth=2, marker="o", markersize=4, label="CV MAE (mean of 5 folds)")
    ax.axvline(best["alpha"], color="#888888", linestyle="--", linewidth=1)
    ax.annotate(f"best α = {best['alpha']:.3g}\nMAE = {m.min():.3f}", (best["alpha"], m.min()),
                xytext=(-12, -42), textcoords="offset points", ha="right", fontsize=8, color="#333333")
    _style(ax, "Ridge (Set C): tuning α", "α (log scale)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _save(fig, "tuning_ridge.png")


def plot_enet(curve, best):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for c, (r, d) in zip(COLORS, curve.groupby("l1_ratio")):
        ax.plot(d["alpha"], d["MAE_mean"], color=c, linewidth=2, label=f"L1 ratio = {r:g}")
    ax.scatter([best["alpha"]], [curve["MAE_mean"].min()], s=60, color="black", zorder=5,
               edgecolors="white", linewidths=1.5)
    ax.annotate(f"best α = {best['alpha']:.3g}, L1 = {best['l1_ratio']:g}\nMAE = {curve['MAE_mean'].min():.3f}",
                (best["alpha"], curve["MAE_mean"].min()), xytext=(0.3, 0.55), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8), fontsize=8, color="#333333")
    _style(ax, "Elastic Net (Set C): tuning α and L1 ratio", "α (log scale)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _save(fig, "tuning_enet.png")


def _style(ax, title, xlabel):
    ax.set_xscale("log")
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("CV MAE (points)", fontsize=9)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run():
    train, y, folds = load_train_folds()
    RESULTS.mkdir(parents=True, exist_ok=True)
    best_all, per_fold, summary = {}, [], []
    for name, plot in [("ridge", plot_ridge), ("enet", plot_enet)]:
        curve, best, t = tune(name, train, y, folds)
        curve.to_csv(RESULTS / f"tuning_{name}.csv", index=False)
        plot(curve, best)
        best_all[f"{name}_setC"] = best
        per_fold.append(t.assign(model=f"{name}_setC"))
        summary.append(summarize(t, f"{name}_setC"))
        print(f"- {name}_setC 最佳參數 {best}  CV MAE {t.MAE.mean():.3f} ± {t.MAE.std(ddof=1):.3f}")
    (RESULTS / "best_params.json").write_text(json.dumps(best_all, indent=2), encoding="utf-8")
    pd.concat(per_fold).to_csv(RESULTS / "cv_models_per_fold.csv", index=False)
    pd.DataFrame(summary).to_csv(RESULTS / "cv_models_summary.csv", index=False)


if __name__ == "__main__":
    run()
