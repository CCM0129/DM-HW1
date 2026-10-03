"""特徵工程(§5):Set A(原始欄位)與 Set C(原始 + 工程特徵)的前處理 Pipeline。
用法:make_pipeline(build_features("C"), Ridge());輸入是主檔的列(DataFrame),所有需要學習的參數只在 fit 時從訓練資料學。
直接執行 python src/features.py 會在訓練集上做煙霧測試(非正式結果)。"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

MASTER = Path("data/processed/master_public.csv")
SPLIT = Path("data/processed/split.csv")

TARGET = "total_cup_points"
NUM_RAW = ["altitude_m", "moisture_pct", "moisture_suspect", "category_one_defects", "category_two_defects",
           "quakers", "n_bags", "bag_weight_kg"]
CAT_LOW = ["processing_group", "color"]          # 每個 Set 都 one-hot
CAT_HIGH = ["country", "region", "variety", "cert_body"]   # Set A one-hot;Set C 改用目標編碼
FEATURE_GROUPS = {   # 依 §4 的觀察設計(見 docs/feature_engineering.md);有沒有用交給 §7 ablation
    "polynomial": ["altitude_m^2"],
    "interactions": ["alt_x_washed", "cat2_x_sample"],
    "transforms": ["log_cat1", "small_bag", "specialty_grade"],
    "origin": ["te_country_region_variety_cert_body", "alt_rel_country"],
}
ENGINEERED = FEATURE_GROUPS["transforms"] + FEATURE_GROUPS["interactions"]   # 逐列算出、不需學習的欄位


def load_data():
    """回傳 (train, test);依 data/processed/split.csv 切分。"""
    assert SPLIT.exists(), "找不到 data/processed/split.csv,請先執行 src/make_split.py"
    df = pd.read_csv(MASTER, keep_default_na=False, na_values=[""])
    df = df.merge(pd.read_csv(SPLIT), on="row_id", how="left", validate="one_to_one")
    assert df["split"].notna().all(), "split.csv 與主檔的 row_id 對不上"
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return train, test


def add_engineered(df):
    """逐列計算的工程特徵(不從資料學參數);缺失輸入維持缺失,交給後面的補值。"""
    d = df.copy()
    washed = (d["processing_group"] == "washed").astype(float)       # 處理法缺失視為非水洗
    # 轉換
    d["log_cat1"] = np.log1p(d["category_one_defects"])              # 第一個第一類瑕疵扣最多(H2)
    d["small_bag"] = (d["bag_weight_kg"] <= 2.5).astype(float).where(d["bag_weight_kg"].notna())   # 袋重雙峰(H3)
    d["specialty_grade"] = ((d["category_one_defects"] == 0) & (d["category_two_defects"] <= 5)).astype(float)  # 近似 SCA 精品生豆分級
    # 交互作用:處理法改變海拔的斜率(H5)、袋型改變瑕疵的斜率(H6)
    d["alt_x_washed"] = d["altitude_m"] / 100 * washed
    d["cat2_x_sample"] = d["category_two_defects"] * d["small_bag"]
    return d


class CountryRelativeAltitude(BaseEstimator, TransformerMixin):
    """海拔減去該產國的訓練集海拔中位數;樣本數 < min_count 或沒看過的產國用整體中位數。"""

    def __init__(self, min_count=5):
        self.min_count = min_count

    def fit(self, X, y=None):
        alt = X["altitude_m"]
        self.overall_ = float(alt.median())
        stats = alt.groupby(X["country"]).agg(["count", "median"])
        self.medians_ = stats.loc[stats["count"] >= self.min_count, "median"].to_dict()
        return self

    def transform(self, X):
        ref = X["country"].map(self.medians_).fillna(self.overall_)
        return (X["altitude_m"] - ref).to_numpy(dtype=float).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["alt_rel_country"], dtype=object)


def _numeric(indicator=True):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=indicator), StandardScaler())


def _onehot():
    return make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                         OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False))


def build_features(feature_set="C", drop_group=None):
    """回傳前處理 Pipeline。feature_set:"A"(原始)或 "C"(我們的);drop_group 只用於 C,可為 FEATURE_GROUPS 的鍵。"""
    if feature_set not in ("A", "C"):
        raise ValueError(f"feature_set 須為 'A' 或 'C',收到 {feature_set!r}")
    if drop_group is not None and (feature_set != "C" or drop_group not in FEATURE_GROUPS):
        raise ValueError(f"drop_group 只用於 Set C,且須為 {list(FEATURE_GROUPS)} 之一")

    if feature_set == "A":
        blocks = [("num", _numeric(), NUM_RAW), ("onehot", _onehot(), CAT_LOW + CAT_HIGH)]
    else:
        eng = [c for c in ENGINEERED if drop_group is None or c not in FEATURE_GROUPS[drop_group]]
        blocks = [("num", _numeric(), NUM_RAW),
                  ("eng", _numeric(indicator=False), eng)]     # 缺失指標已由原始欄位提供,不重複
        if drop_group != "polynomial":       # 海拔先在訓練折內置中、標準化再平方(H1)
            blocks.append(("poly", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                 FunctionTransformer(np.square, feature_names_out=lambda t, n: ["altitude_m^2"]),
                                                 StandardScaler()), ["altitude_m"]))
        if drop_group == "origin":           # 拿掉產地特徵時,產國 / 產區 / 品種 / 評鑑機構退回 one-hot,不完全丟掉資訊
            blocks.append(("onehot", _onehot(), CAT_LOW + CAT_HIGH))
        else:
            blocks += [
                ("onehot", _onehot(), CAT_LOW),
                ("te", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                                     TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=0)),
                                     StandardScaler()), CAT_HIGH),     # fit_transform 內部交叉擬合,不洩漏
                ("altrel", make_pipeline(CountryRelativeAltitude(), SimpleImputer(strategy="median"),
                                         StandardScaler()), ["country", "altitude_m"]),
            ]

    engineer = FunctionTransformer(add_engineered, validate=False,
                                   feature_names_out=lambda self, names: list(names) + ENGINEERED)
    return Pipeline([("eng", engineer), ("pre", ColumnTransformer(blocks, remainder="drop"))])


if __name__ == "__main__":
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, cross_val_score

    train, _ = load_data()           # 只用訓練集
    y, groups = train[TARGET], train["group_id"]
    cv = GroupKFold(n_splits=5)
    print(f"煙霧測試(非正式結果,正式調參見 §5.2/§6):訓練集 {len(train)} 列,Ridge(alpha=1),5 折 GroupKFold\n")
    settings = [("A", None), ("C", None)] + [("C", g) for g in FEATURE_GROUPS]
    for s, g in settings:
        fe = build_features(s, g).fit(train, y)
        width = len(fe.get_feature_names_out())
        mae = -cross_val_score(make_pipeline(build_features(s, g), Ridge(alpha=1.0)), train, y,
                               groups=groups, cv=cv, scoring="neg_mean_absolute_error")
        label = f"Set {s}" + (f" − {g}" if g else "")
        print(f"- {label:<20s} 特徵數 {width:3d}  CV MAE {mae.mean():.3f} ± {mae.std():.3f} 分")
