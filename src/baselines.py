"""Baseline(§6):全體平均、產國平均、只用海拔的 OLS、隨機森林 / 梯度提升(Set A 原始特徵)。
直接執行會在固定折上跑 CV,輸出 results/cv_baselines_*.csv。"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from evaluate import RANDOM_SEED, RESULTS, cross_validate, load_train_folds, summarize
from features import build_features


class GroupMeanRegressor(BaseEstimator, RegressorMixin):
    """「只看產地名聲」:預測 = 訓練資料中同產國的平均分數;樣本數 < min_count 或沒看過的產國用全體平均。"""

    def __init__(self, col="country", min_count=5):
        self.col = col
        self.min_count = min_count

    def fit(self, X, y):
        y = pd.Series(np.asarray(y, float), index=X.index)
        stats = y.groupby(X[self.col]).agg(["count", "mean"])
        self.means_ = stats.loc[stats["count"] >= self.min_count, "mean"].to_dict()
        self.overall_ = float(y.mean())
        return self

    def predict(self, X):
        return X[self.col].map(self.means_).fillna(self.overall_).to_numpy(dtype=float)


def baseline_models():
    """名稱 → 未訓練的模型;輸入都是主檔的 DataFrame。"""
    altitude_only = make_pipeline(
        ColumnTransformer([("alt", SimpleImputer(strategy="median"), ["altitude_m"])]), LinearRegression())
    return {
        "mean": DummyRegressor(strategy="mean"),
        "country_mean": GroupMeanRegressor("country", min_count=5),
        "ols_altitude": altitude_only,
        "rf_setA": make_pipeline(build_features("A"), RandomForestRegressor(
            n_estimators=500, min_samples_leaf=5, max_features=0.33, n_jobs=-1, random_state=RANDOM_SEED)),
        "hgb_setA": make_pipeline(build_features("A"), HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=300, min_samples_leaf=20, l2_regularization=1.0, random_state=RANDOM_SEED)),
    }


def run():
    train, y, folds = load_train_folds()
    RESULTS.mkdir(parents=True, exist_ok=True)
    per_fold, summary = [], []
    for name, model in baseline_models().items():
        t = cross_validate(model, train, y, folds)
        per_fold.append(t.assign(model=name))
        summary.append(summarize(t, name))
        print(f"- {name:<13s} MAE {t.MAE.mean():.3f} ± {t.MAE.std(ddof=1):.3f}   RMSE {t.RMSE.mean():.3f}")
    pd.concat(per_fold).to_csv(RESULTS / "cv_baselines_per_fold.csv", index=False)
    pd.DataFrame(summary).to_csv(RESULTS / "cv_baselines_summary.csv", index=False)


if __name__ == "__main__":
    run()
