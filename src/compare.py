"""主結果表(§7.1,只用 CV,不碰測試集):所有 baseline 與調好的線性模型在相同折上比較,
跨折 paired t-test,以及「低於多少分就不送杯測」的篩選門檻分析(用 out-of-fold 預測)。
需先執行 baselines.py 與 models.py(用到 best_params.json)。"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from baselines import baseline_models
from evaluate import FIGURES, RESULTS, THRESHOLD, cross_validate, load_train_folds, summarize
from models import COLORS, make_model

FINAL = "ridge_setC"      # 主模型:與 enet 的 CV MAE 只差約 0.001 分,Ridge 較簡單且保留全部係數給 §8 分析
CUTOFFS = np.round(np.arange(79.0, 83.01, 0.25), 2)
MAX_GOOD_KILLED = 0.05    # 選門檻的規則:錯殺的好豆不超過 5%


def all_models():
    best = json.loads((RESULTS / "best_params.json").read_text(encoding="utf-8"))
    models = baseline_models()
    for key, params in best.items():
        models[key] = make_model(key.split("_")[0], "C", **params)
    return models


def screening_table(y, pred):
    """預測分數 < cutoff 就不送杯測;每個 cutoff 的節省比例、擋下的壞豆、錯殺的好豆與留下批次的達標率。"""
    y, pred = np.asarray(y), np.asarray(pred)
    good = y >= THRESHOLD
    rows = []
    for c in CUTOFFS:
        skip = pred < c
        rows.append({"cutoff": c, "skipped": skip.mean(), "bad_catch": skip[~good].mean(),
                     "good_killed": skip[good].mean(),
                     "precision_kept": good[~skip].mean() if (~skip).any() else np.nan})
    return pd.DataFrame(rows)


def plot_screening(curves):
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for c, (name, t) in zip(COLORS, curves.items()):
        ax.plot(t["good_killed"], t["bad_catch"], color=c, linewidth=2, marker="o", markersize=3, label=name)
    ax.axvline(MAX_GOOD_KILLED, color="#888888", linestyle="--", linewidth=1)
    ax.text(MAX_GOOD_KILLED + 0.005, 0.02, f"{MAX_GOOD_KILLED:.0%} good lots lost", fontsize=8, color="#555555")
    ax.set_xlabel("Good lots (≥ 80) wrongly skipped", fontsize=9)
    ax.set_ylabel("Sub-80 lots correctly skipped", fontsize=9)
    ax.set_title(f"Screening trade-off, cutoffs {CUTOFFS[0]:g}–{CUTOFFS[-1]:g} (CV, out-of-fold)",
                 fontsize=10, loc="left")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(color="#e5e5e5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "screening_tradeoff.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run():
    train, y, folds = load_train_folds()
    FIGURES.mkdir(parents=True, exist_ok=True)
    tables, oof = {}, {}
    for name, model in all_models().items():
        tables[name], oof[name] = cross_validate(model, train, y, folds, return_pred=True)

    # 主結果表 + 偵測低分批次的 AUC(只看排序,與門檻無關)
    main = pd.DataFrame([summarize(t, n) for n, t in tables.items()])
    main["AUC_sub80"] = [roc_auc_score(y < THRESHOLD, -oof[n]) for n in main["model"]]
    main.to_csv(RESULTS / "main_cv_table.csv", index=False)

    # 跨折 paired t-test:主模型 vs 其他每個模型(每折 MAE 配對;只有 5 折,檢定力有限)
    rows = []
    for name, t in tables.items():
        if name == FINAL:
            continue
        diff = tables[FINAL]["MAE"].to_numpy() - t["MAE"].to_numpy()
        res = stats.ttest_rel(tables[FINAL]["MAE"], t["MAE"])
        rows.append({"vs": name, "MAE_diff_mean": diff.mean(), "MAE_diff_std": diff.std(ddof=1),
                     "folds_better": int((diff < 0).sum()), "t": res.statistic, "p": res.pvalue})
    pd.DataFrame(rows).to_csv(RESULTS / "paired_ttest_cv.csv", index=False)

    # 篩選門檻:用主模型 OOF 預測選 cutoff,存檔給 final_test.py 用
    curves = {n: screening_table(y, oof[n]) for n in ["country_mean", "rf_setA", FINAL]}
    for n, t in curves.items():
        t.to_csv(RESULTS / f"screening_{n}.csv", index=False)
    ok = curves[FINAL][curves[FINAL]["good_killed"] <= MAX_GOOD_KILLED]
    chosen = float(ok["cutoff"].max())
    (RESULTS / "screening_cutoff.json").write_text(
        json.dumps({"model": FINAL, "cutoff": chosen, "rule": f"good_killed <= {MAX_GOOD_KILLED}"}, indent=2),
        encoding="utf-8")
    plot_screening(curves)
    pd.DataFrame({"row_id": train["row_id"], "fold": folds, "y": y, **{f"pred_{n}": p for n, p in oof.items()}}) \
        .to_csv(RESULTS / "oof_predictions.csv", index=False)

    show = main.set_index("model")[["MAE_mean", "MAE_std", "RMSE_mean", "R2_mean", "AUC_sub80"]].round(3)
    print(show.sort_values("MAE_mean").to_string())
    print("\n主模型 vs 其他(MAE 差 < 0 表示主模型較好):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"\n篩選門檻:{FINAL} 預測 < {chosen:g} 分就不送測(規則:錯殺好豆 ≤ {MAX_GOOD_KILLED:.0%})")
    print(curves[FINAL][curves[FINAL]["cutoff"] == chosen].round(3).to_string(index=False))


if __name__ == "__main__":
    run()
