"""測試集評估(§7.1):所有設計定案後只跑一次。用全部訓練集重訓每個模型,在測試集算指標,
以 group_id 為單位的自助法(群組 bootstrap)給 MAE 與「主模型 − baseline」的 95% 信賴區間。
  python src/final_test.py --dry-run   用訓練集最後 20%(依評鑑日)當假測試集,只檢查程式,不碰測試集
  python src/final_test.py --confirm   正式評估測試集(只跑一次!結果寫入 results/test_*)"""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import clone

from compare import FINAL, all_models
from evaluate import RANDOM_SEED, RESULTS, metrics
from features import TARGET, load_data

N_BOOT = 2000


def group_bootstrap(y, preds, groups, final, n_boot=N_BOOT, seed=RANDOM_SEED):
    """整組重抽(同農場的批次一起抽),回傳每個模型 MAE 與「主模型 − 該模型」MAE 差的 95% CI。"""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    err = {n: np.abs(p - y) for n, p in preds.items()}
    boot = {n: [] for n in preds}
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_g[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        for n in preds:
            boot[n].append(err[n][idx].mean())
    boot = {n: np.array(v) for n, v in boot.items()}
    rows = []
    for n in preds:
        d = boot[final] - boot[n]
        rows.append({"model": n, "MAE_lo": np.percentile(boot[n], 2.5), "MAE_hi": np.percentile(boot[n], 97.5),
                     "diff_vs_final_lo": np.percentile(d, 2.5), "diff_vs_final_hi": np.percentile(d, 97.5),
                     "p_final_not_better": (d >= 0).mean()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--confirm", action="store_true")
    args = ap.parse_args()

    train, test = load_data()
    prefix = "test"
    if args.dry_run:      # 假測試集:訓練集依 row_id(評鑑日)排序後的最後 20%
        train = train.sort_values("row_id").reset_index(drop=True)
        cut = int(len(train) * 0.8)
        train, test, prefix = train.iloc[:cut], train.iloc[cut:].reset_index(drop=True), "dryrun"
    elif (RESULTS / "test_metrics.csv").exists():
        raise SystemExit("results/test_metrics.csv 已存在:測試集已評估過。若確定要重跑,請先手動刪除並在報告中說明原因。")

    y_tr, y_te = train[TARGET].to_numpy(float), test[TARGET].to_numpy(float)
    cutoff = json.loads((RESULTS / "screening_cutoff.json").read_text(encoding="utf-8"))["cutoff"]
    preds, rows = {}, []
    for name, model in all_models().items():
        preds[name] = clone(model).fit(train, y_tr).predict(test)
        skip = preds[name] < cutoff
        good = y_te >= 80
        rows.append({"model": name, **metrics(y_te, preds[name]),
                     "skipped@cutoff": skip.mean(), "bad_catch@cutoff": skip[~good].mean() if (~good).any() else np.nan,
                     "good_killed@cutoff": skip[good].mean()})
    table = pd.DataFrame(rows).merge(group_bootstrap(y_te, preds, test["group_id"].to_numpy(), FINAL), on="model")
    table.to_csv(RESULTS / f"{prefix}_metrics.csv", index=False)
    pd.DataFrame({"row_id": test["row_id"], "y": y_te, **{f"pred_{n}": p for n, p in preds.items()}}) \
        .to_csv(RESULTS / f"{prefix}_predictions.csv", index=False)
    if args.confirm:
        (RESULTS / "test_run_log.txt").write_text(
            f"測試集評估時間:{datetime.now().isoformat(timespec='seconds')}\n主模型:{FINAL}\n篩選門檻:{cutoff}\n",
            encoding="utf-8")

    print(f"{'假測試集(dry run)' if args.dry_run else '測試集'}:{len(test)} 列,低於 80 分 {int((y_te < 80).sum())} 列;篩選門檻 {cutoff:g}")
    cols = ["model", "MAE", "MAE_lo", "MAE_hi", "RMSE", "R2", "diff_vs_final_lo", "diff_vs_final_hi",
            "bad_catch@cutoff", "good_killed@cutoff"]
    print(table[cols].sort_values("MAE").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
