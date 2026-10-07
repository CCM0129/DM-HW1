"""共用評估(§6):固定的 GroupKFold 折、指標計算與交叉驗證。
全組所有實驗(baseline、調參、組員 4 的消融)都必須透過 cross_validate() 跑,確保切分與指標一致。
直接執行 python src/evaluate.py 會產生並檢查 folds.csv。
組員 1、2 的資料與特徵模組在另一個資料夾,預設為同層的 DM-HW1-feature-feature-engineering;
放在別處時設環境變數 DM_FEATURE_REPO 指向它。"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]          # 本資料夾(DM-HW1-model)
FEATURE_REPO = Path(os.environ.get("DM_FEATURE_REPO", ROOT.parent / "DM-HW1-feature-feature-engineering")).resolve()
assert (FEATURE_REPO / "src" / "features.py").exists(), f"找不到組員 2 的 features.py:{FEATURE_REPO},請設定 DM_FEATURE_REPO"
sys.path.insert(0, str(FEATURE_REPO / "src"))
import features  # noqa: E402  組員 2 的模組
from features import TARGET, load_data  # noqa: E402

features.MASTER = FEATURE_REPO / features.MASTER     # features.py 用相對路徑讀資料;改成絕對路徑,從任何目錄執行都可以
features.SPLIT = FEATURE_REPO / features.SPLIT

FOLDS = ROOT / "folds.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
N_FOLDS = 5
THRESHOLD = 80.0          # 精品級門檻(分)
RANDOM_SEED = 42

if hasattr(sys.stdout, "reconfigure"):    # Windows 中文主控台(cp950)印不出部分符號
    sys.stdout.reconfigure(encoding="utf-8")


def make_folds(train):
    """依 group_id 做 GroupKFold,同一農場 / 公司的批次落在同一折;結果只由資料決定,可重現。"""
    fold = np.full(len(train), -1)
    for k, (_, va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(train, groups=train["group_id"])):
        fold[va] = k
    return pd.DataFrame({"row_id": train["row_id"], "fold": fold})


def load_folds(train):
    """讀 folds.csv(不存在就產生),回傳與 train 列順序對齊的 fold 陣列。"""
    if not FOLDS.exists():
        make_folds(train).to_csv(FOLDS, index=False)
    f = train[["row_id"]].merge(pd.read_csv(FOLDS), on="row_id", how="left", validate="one_to_one")
    assert f["fold"].notna().all(), "folds.csv 與訓練集 row_id 對不上(切分改過?請刪掉 folds.csv 重產)"
    return f["fold"].to_numpy(dtype=int)


def load_train_folds():
    """常用組合:回傳 (train, y, folds)。"""
    train, _ = load_data()
    return train, train[TARGET], load_folds(train)


def metrics(y_true, y_pred, threshold=THRESHOLD):
    """回歸指標 + 以 threshold 篩選的指標。「預測 ≥ 門檻」= 模型建議送杯測。
    precision:建議送測的批次中真正達標的比例;recall:達標好豆中被建議送測的比例(1 − 錯殺率);
    bad_catch:未達標批次中被正確擋下的比例(篩選真正省錢的部分)。"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    good, keep = y_true >= threshold, y_pred >= threshold
    return {
        "MAE": np.abs(err).mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "R2": 1 - (err ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum(),
        "precision": good[keep].mean() if keep.any() else np.nan,
        "recall": keep[good].mean() if good.any() else np.nan,
        "good_killed": (~keep)[good].mean() if good.any() else np.nan,
        "bad_catch": (~keep)[~good].mean() if (~good).any() else np.nan,
        "n_bad": int((~good).sum()),
    }


def cross_validate(pipeline, X, y, folds, return_pred=False):
    """對任何 sklearn 模型 / Pipeline 跑固定折的交叉驗證。回傳每折一列的指標表;
    return_pred=True 時另外回傳 out-of-fold 預測(給誤差分析用)。"""
    y = np.asarray(y, float)
    rows, oof = [], np.full(len(y), np.nan)
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        model = clone(pipeline).fit(X[tr], y[tr])
        oof[va] = model.predict(X[va])
        rows.append({"fold": int(k), "n_val": int(va.sum()), **metrics(y[va], oof[va])})
    table = pd.DataFrame(rows)
    return (table, oof) if return_pred else table


def summarize(table, name):
    """每折指標表 → 一列「平均 ± 標準差」。"""
    cols = ["MAE", "RMSE", "R2", "precision", "recall", "good_killed", "bad_catch"]
    out = {"model": name}
    for c in cols:
        out[f"{c}_mean"], out[f"{c}_std"] = table[c].mean(), table[c].std(ddof=1)
    return out


if __name__ == "__main__":
    train, y, folds = load_train_folds()
    print(f"折檔:{FOLDS}")
    g = pd.DataFrame({"fold": folds, "group": train["group_id"], "y": y})
    print(g.groupby("fold").agg(列數=("y", "size"), 組數=("group", "nunique"), 平均分=("y", "mean"),
                                低於80比例=("y", lambda s: (s < THRESHOLD).mean())).round(3))
    leak = g.groupby("group")["fold"].nunique().max()
    assert leak == 1, "有 group 跨折,GroupKFold 失效"
    print("檢查通過:每個 group_id 只出現在一折。")
