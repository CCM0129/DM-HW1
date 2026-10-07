"""§5.2 選 Ridge 的證據:Set C 的 VIF 與高相關特徵對、OLS 與 Ridge 係數比較、各線性變體的 CV MAE。只用訓練集。
輸出 results/vif_setC.csv、corr_pairs_setC.csv、coef_ols_vs_ridge.csv、cv_linear_variants.csv。"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from evaluate import RESULTS, cross_validate, load_train_folds, summarize
from features import build_features
from models import make_model, tune

REFERENCE = ["processing_group_missing", "color_missing"]   # one-hot 各去一欄當參照,否則必然完全共線


def short(names):
    return [n.split("__", 1)[1] for n in names]


def vif_and_pairs(X, y, top=10):
    """回傳 (VIF 表, 相關係數絕對值最高的 top 對特徵)。"""
    fe = build_features("C").fit(X, y)
    Z = pd.DataFrame(fe.transform(X), columns=short(fe.get_feature_names_out())).drop(columns=REFERENCE)
    R = np.corrcoef(Z.to_numpy(), rowvar=False)
    vif = np.diag(np.linalg.pinv(R))   # VIF_j = (R^-1)_jj
    vif = pd.Series(vif, index=Z.columns, name="VIF").sort_values(ascending=False).rename_axis("feature").reset_index()
    i, j = np.triu_indices_from(R, k=1)
    pairs = pd.DataFrame({"feature_1": Z.columns[i], "feature_2": Z.columns[j], "corr": R[i, j]})
    pairs = pairs.reindex(pairs["corr"].abs().sort_values(ascending=False).index).head(top)
    return vif, pairs


def coef_table(X, y, alpha):
    ols = make_pipeline(build_features("C"), LinearRegression()).fit(X, y)
    ridge = make_model("ridge", alpha=alpha).fit(X, y)
    names = short(ols[0].get_feature_names_out())
    t = pd.DataFrame({"feature": names, "OLS": ols[-1].coef_, "Ridge": ridge[-1].coef_})
    t["sign_flip"] = np.sign(t["OLS"]) != np.sign(t["Ridge"])
    return t


def run():
    X, y, folds = load_train_folds()
    best = json.loads((RESULTS / "best_params.json").read_text(encoding="utf-8"))
    alpha = best["ridge_setC"]["alpha"]

    vif, pairs = vif_and_pairs(X, y)
    vif.to_csv(RESULTS / "vif_setC.csv", index=False)
    pairs.to_csv(RESULTS / "corr_pairs_setC.csv", index=False)
    print(f"VIF 最高 {vif.VIF.max():.1f}({vif.feature[0]});VIF > 10 有 {(vif.VIF > 10).sum()} / {len(vif)} 欄")
    print(vif[vif.VIF > 5].round(1).to_string(index=False))
    print("\n相關係數最高的特徵對:")
    print(pairs.round(2).to_string(index=False))

    coef = coef_table(X, y, alpha)
    coef.to_csv(RESULTS / "coef_ols_vs_ridge.csv", index=False)
    print(f"\nOLS 與 Ridge(α = {alpha:g})正負號相反的特徵:")
    print(coef[coef.sign_flip].round(3).to_string(index=False))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        _, lasso_best, lasso_t = tune("enet", X, y, folds,
                                      grid={"alpha": np.logspace(-4, 0, 17), "l1_ratio": [1.0]})
    rows = [summarize(cross_validate(make_pipeline(build_features("C"), LinearRegression()), X, y, folds),
                      "OLS (no penalty)"),
            summarize(lasso_t, f"Lasso (L1, alpha={lasso_best['alpha']:.3g})")]
    cv = pd.read_csv(RESULTS / "cv_models_summary.csv")
    for key, label in [("enet_setC", "Elastic Net (L1+L2, alpha={alpha:.3g}, l1_ratio={l1_ratio:g})"),
                       ("ridge_setC", "Ridge (L2, alpha={alpha:g})")]:
        rows.append({**cv[cv.model == key].iloc[0].to_dict(), "model": label.format(**best[key])})
    out = pd.DataFrame(rows)[["model", "MAE_mean", "MAE_std", "RMSE_mean", "R2_mean"]]
    out.to_csv(RESULTS / "cv_linear_variants.csv", index=False)
    print("\n各線性變體(Set C,相同 5 折):")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
