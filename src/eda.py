"""探索性特徵分析(§4):只用訓練集,測試集完全不讀。
圖輸出到 docs/eda_figures/,關鍵數字印在終端。需先執行 src/make_split.py。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

sys.path.insert(0, str(Path(__file__).parent))
from features import CAT_HIGH, CAT_LOW, ENGINEERED, NUM_RAW, TARGET, CountryRelativeAltitude, add_engineered, load_data

FIG = Path("docs/eda_figures")
YLIM = (74, 91)
YLABEL = "Total cup points (points)"
SHORT = {"Tanzania, United Republic Of": "Tanzania", "United States (Hawaii)": "Hawaii (US)"}

FIG.mkdir(parents=True, exist_ok=True)
train, _ = load_data()           # 測試集丟掉不用
tr = add_engineered(train)
tr["alt_rel_country"] = CountryRelativeAltitude().fit(tr).transform(tr)[:, 0]
z = (tr["altitude_m"] - tr["altitude_m"].mean()) / tr["altitude_m"].std()
tr["altitude_m^2"] = z ** 2      # 與 features.py 相同:先置中、標準化再平方
y = tr[TARGET]
num_cols = NUM_RAW + ENGINEERED + ["altitude_m^2", "alt_rel_country"]
print(f"訓練集 {len(tr)} 列;目標平均 {y.mean():.2f}、標準差 {y.std():.2f}、範圍 {y.min():.2f}~{y.max():.2f} 分")
print(f"低於 {YLIM[0]} 分(圖中被裁掉)的點:{int((y < YLIM[0]).sum())} 個\n")


def note(tag, claim, evidence, feature):
    print(f"- {tag} {claim}:{evidence} → {feature}")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


JITTER = {"altitude_m": 25}   # 海拔多為整數百公尺,畫圖時左右微抖避免點疊在一起(不影響計算)
RNG = np.random.default_rng(0)


def points(ax, x, yy, color, label=None):
    """散佈點:放大、加白邊,並對重複值微抖。"""
    w = JITTER.get(x.name, 0)
    ax.scatter(x + RNG.uniform(-w, w, len(x)), yy, s=16, alpha=0.5, color=color,
               edgecolors="white", linewidths=0.3, label=label)


def scatter_binned(ax, col, bins):
    """散佈 + 分箱平均 ± 95% CI;回傳分箱平均。"""
    d = tr[[col, TARGET]].dropna()
    points(ax, d[col], d[TARGET], "tab:blue", label="lot (x jittered)")
    g = d.groupby(pd.cut(d[col], bins), observed=True)
    ax.errorbar(g[col].mean(), g[TARGET].mean(), yerr=1.96 * g[TARGET].sem(), fmt="o", color="black",
                ms=7, capsize=4, lw=1.5, label="binned mean ± 95% CI", zorder=3)
    return g[TARGET].mean()


def slope_ci(d, x, scale=1):
    r = stats.linregress(d[x], d[TARGET])
    return r.slope * scale, 1.96 * r.stderr * scale, r


print("觀察與假設:")
# ---- H1:海拔的曲線關係 → 二次項 ----
fig, ax = plt.subplots(figsize=(6.5, 4.3))
bm = scatter_binned(ax, "altitude_m", [0, 600, 900, 1100, 1300, 1500, 1700, 1900, 2600])
d = tr[["altitude_m", TARGET]].dropna()
zz = (d["altitude_m"] - d["altitude_m"].mean()) / d["altitude_m"].std()
b = np.polyfit(zz, d[TARGET], 2)
xs = np.linspace(d["altitude_m"].min(), d["altitude_m"].max(), 100)
ax.plot(xs, np.polyval(b, (xs - d["altitude_m"].mean()) / d["altitude_m"].std()), color="tab:red", lw=2.5,
        label=f"quadratic fit (z² coef {b[0]:+.2f})")
rho = stats.spearmanr(d["altitude_m"], d[TARGET])[0]
ax.set(xlabel="Altitude (m)", ylabel=YLABEL, ylim=YLIM, title=f"(H1) Altitude: convex, Spearman ρ = {rho:+.2f}")
ax.legend(loc="lower left", fontsize=8)
save(fig, "h1_altitude.png")
note("H1", "海拔與分數是曲線(凸)關係",
     f"分箱平均 1,100–1,500 m 約 {bm.iloc[3:5].mean():.1f} 持平,>1,900 m 升到 {bm.iloc[-1]:.1f};二次項係數 {b[0]:+.2f}(z²)",
     "多項式:altitude_m^2")

# ---- H2:第一個瑕疵扣最多(log)+ SCA 分級門檻 ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={"width_ratios": [1.4, 1]})
g = tr.groupby(pd.cut(tr["category_one_defects"], [-1, 0, 1, 2, 4, 40], labels=["0", "1", "2", "3–4", "≥5"]),
               observed=True)[TARGET]
a1.errorbar(range(len(g.mean())), g.mean(), yerr=1.96 * g.sem(), fmt="o-", color="black", capsize=3)
a1.set_xticks(range(len(g.mean())), [f"{k}\n(n={n})" for k, n in g.size().items()])
a1.set(xlabel="Category-1 defects (count)", ylabel=YLABEL, title="(a) Mean ± 95% CI by category-1 defects")
spec, below = y[tr["specialty_grade"] == 1], y[tr["specialty_grade"] == 0]
a2.boxplot([spec, below], tick_labels=[f"specialty grade\nn={len(spec)}", f"below grade\nn={len(below)}"], showmeans=True)
a2.set(ylim=YLIM, title=f"(b) SCA grade: diff {spec.mean() - below.mean():+.2f} pts")
save(fig, "h2_defects.png")
first = g.mean().iloc[0] - g.mean().iloc[1]
note("H2", "第一類瑕疵的邊際傷害遞減,且 SCA 精品等級是品質分界",
     f"0→1 個瑕疵掉 {first:.2f} 分,之後趨緩;84% 為 0、偏態 {tr['category_one_defects'].skew():.1f};"
     f"精品等級高 {spec.mean() - below.mean():.2f} 分", "轉換:log_cat1、specialty_grade")

# ---- H3:袋重雙峰(樣品袋 vs 出口袋)→ 二元化 ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.3))
bw = tr["bag_weight_kg"].dropna()
a1.hist(bw, bins=np.logspace(np.log10(0.4), np.log10(200), 40), color="tab:blue")
a1.set_xscale("log")
a1.axvline(2.5, ls="--", color="grey")
a1.set(xlabel="Bag weight (kg)", ylabel="Lots", title="(a) Bag weight is bimodal")
small, big = y[tr["small_bag"] == 1], y[tr["small_bag"] == 0]
diff = small.mean() - big.mean()
ci = 1.96 * np.sqrt(small.var() / len(small) + big.var() / len(big))
a2.boxplot([small, big], tick_labels=[f"≤ 2.5 kg (sample)\nn={len(small)}", f"> 2.5 kg\nn={len(big)}"], showmeans=True)
a2.set(ylabel=YLABEL, ylim=YLIM, title=f"(b) diff {diff:+.2f} [{diff - ci:+.2f}, {diff + ci:+.2f}] pts")
save(fig, "h3_bag_weight.png")
note("H3", "袋重分成樣品袋與出口袋兩群,兩群分數不同",
     f"≤ 2.5 kg 佔 {len(small) / len(bw):.0%};樣品袋低 {-diff:.2f} 分 [{-diff - ci:.2f}, {-diff + ci:.2f}]", "轉換:small_bag")

# ---- H4:產地(產國差異 + 產國內的海拔效果)----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})
top = tr["country"].value_counts().head(12).index
order = tr[tr["country"].isin(top)].groupby("country")[TARGET].median().sort_values().index
kw = stats.kruskal(*[y[tr["country"] == c] for c, n in tr["country"].value_counts().items() if n >= 20])
a1.boxplot([y[tr["country"] == c] for c in order], tick_labels=[SHORT.get(c, c) for c in order])
a1.tick_params(axis="x", rotation=45)
a1.set(ylabel=YLABEL, ylim=YLIM, title=f"(a) Country (12 most frequent); Kruskal–Wallis p = {kw.pvalue:.1g}")
d = tr.dropna(subset=["altitude_m"])
within = {c: (float(stats.spearmanr(g["altitude_m"], g[TARGET])[0]), len(g))
          for c, g in d.groupby("country") if len(g) >= 30}
within = dict(sorted(within.items(), key=lambda kv: kv[1][0]))
a2.barh([f"{SHORT.get(c, c)} (n={n})" for c, (_, n) in within.items()], [r for r, _ in within.values()], color="tab:blue")
a2.axvline(0, color="black", lw=0.8)
a2.set(xlabel="Spearman ρ (altitude vs points)", title="(b) Altitude effect within each country")
save(fig, "h4_origin.png")
cm = y.groupby(tr["country"]).agg(["size", "mean"]).query("size >= 20")["mean"]
note("H4", "產地本身影響分數,且同樣的海拔在不同產國意義不同",
     f"產國平均 {cm.min():.1f}–{cm.max():.1f}(Kruskal–Wallis p = {kw.pvalue:.1g});產國內海拔 ρ 從 "
     f"{min(r for r, _ in within.values()):+.2f} 到 {max(r for r, _ in within.values()):+.2f};"
     f"region 互資訊最高但有 {tr['region'].nunique()} 類", "轉換:產國 / 產區 / 品種目標編碼、alt_rel_country")


# ---- H5、H6:某個分組改變斜率 → 交互作用 ----
def interaction_plot(name, tag, x, group, levels, bins, labels, scale, unit, xlabel, title):
    """分組分箱平均線:x 切段,每段畫兩組平均 ± 95% CI;兩條線不平行就是交互作用。回傳各組斜率。"""
    d = tr[[x, TARGET]].assign(g=group).dropna()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    out, counts = {}, []
    for (lvl, color), off in zip(zip(levels, ["tab:blue", "tab:orange"]), [-0.08, 0.08]):
        dd = d[d["g"] == lvl]
        grp = dd.groupby(pd.cut(dd[x], bins, labels=labels), observed=False)[TARGET]
        sl, ci, _ = slope_ci(dd, x, scale)
        ax.errorbar(np.arange(len(labels)) + off, grp.mean(), yerr=1.96 * grp.sem(), fmt="o-", color=color,
                    capsize=4, lw=2.2, ms=7, label=f"{lvl} (n={len(dd)}): slope {sl:+.2f} ± {ci:.2f} pts per {unit}")
        out[lvl] = (sl, ci, len(dd))
        counts.append(grp.size())
    ax.set_xticks(range(len(labels)), [f"{lab}\n({a} / {b})" for lab, a, b in zip(labels, *counts)])
    ax.set(xlabel=f"{xlabel}   (n per bin: {levels[0].split(' (')[0]} / {levels[1].split(' (')[0]})",
           ylabel=YLABEL, title=f"({tag}) {title}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    save(fig, name)
    return out


s5 = interaction_plot("h5_altitude_by_process.png", "H5", "altitude_m",
                      tr["processing_group"].where(tr["processing_group"].isin(["washed", "natural"])), ["washed", "natural"],
                      [0, 900, 1100, 1300, 1500, 1700, 2600], ["<900", "900–1.1k", "1.1–1.3k", "1.3–1.5k", "1.5–1.7k", "≥1.7k"],
                      100, "100 m", "Altitude (m)", "Altitude × processing: mean ± 95% CI per bin")
note("H5", "水洗豆的海拔效果比日曬豆強",
     f"每 100 m:水洗 {s5['washed'][0]:+.2f} ± {s5['washed'][1]:.2f}、日曬 {s5['natural'][0]:+.2f} ± {s5['natural'][1]:.2f}",
     "交互作用:alt_x_washed")
bag = tr["small_bag"].map({1.0: "sample bag (≤ 2.5 kg)", 0.0: "export bag (> 2.5 kg)"})
s6 = interaction_plot("h6_defects_by_bag.png", "H6", "category_two_defects", bag,
                      ["sample bag (≤ 2.5 kg)", "export bag (> 2.5 kg)"],
                      [-1, 0, 2, 5, 10, 60], ["0", "1–2", "3–5", "6–10", ">10"],
                      1, "defect", "Category-2 defects (count)", "Category-2 defects × bag type: mean ± 95% CI per bin")
sm, ex = s6["sample bag (≤ 2.5 kg)"], s6["export bag (> 2.5 kg)"]
note("H6", "樣品袋批次的第二類瑕疵扣分比出口袋重",
     f"每多 1 顆:樣品袋 {sm[0]:+.3f} ± {sm[1]:.3f}、出口袋 {ex[0]:+.3f} ± {ex[1]:.3f}", "交互作用:cat2_x_sample")

# ---- 關聯:Pearson / Spearman(數值)與互資訊(原始欄位)----
rows = []
for c in num_cols:
    m = tr[c].notna()
    pr, pp = stats.pearsonr(tr.loc[m, c], y[m])
    sr, sp = stats.spearmanr(tr.loc[m, c], y[m])
    rows.append({"feature": c, "kind": "raw" if c in NUM_RAW else "engineered", "dtype_kind": "numeric",
                 "n_nonmissing": int(m.sum()), "n_unique": int(tr[c].nunique()),
                 "pearson_r": pr, "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp})
cat_cols = CAT_LOW + CAT_HIGH
X_mi = pd.concat([tr[NUM_RAW].fillna(tr[NUM_RAW].median()),
                  tr[cat_cols].fillna("missing").apply(lambda s: pd.factorize(s)[0])], axis=1)
mi = mutual_info_regression(X_mi, y, discrete_features=[c in cat_cols for c in X_mi.columns],
                            n_neighbors=3, random_state=0)
mi = pd.Series(mi, index=X_mi.columns)
for c in cat_cols:
    rows.append({"feature": c, "kind": "raw", "dtype_kind": "categorical",
                 "n_nonmissing": int(tr[c].notna().sum()), "n_unique": int(tr[c].nunique())})
assoc = pd.DataFrame(rows)
assoc["mutual_info"] = assoc["feature"].map(mi)
assoc = (assoc.assign(_r=assoc["spearman_rho"].abs()).sort_values(["_r", "mutual_info"], ascending=False)
         .drop(columns="_r").round(4))

# ---- 圖 3:關聯排名 ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
s = assoc[assoc["dtype_kind"] == "numeric"].iloc[::-1]
colors = ["tab:blue" if k == "raw" else "tab:orange" for k in s["kind"]]
a1.barh(s["feature"], s["spearman_rho"], color=colors)
a1.axvline(0, color="black", lw=0.8)
a1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="tab:blue"), plt.Rectangle((0, 0), 1, 1, color="tab:orange")],
          labels=["raw", "engineered"], loc="lower right")
a1.set(xlabel="Spearman ρ with total cup points", title="(a) Rank correlation (numeric features)")
s = mi.sort_values()
a2.barh(s.index, s.values, color=["tab:green" if c in cat_cols else "tab:blue" for c in s.index])
for i, c in enumerate(s.index):
    if c in cat_cols:   # 高基數類別的互資訊會偏高,標出類別數供判讀
        a2.text(s[c], i, f" {tr[c].nunique()} levels", va="center", fontsize=8)
a2.set(xlabel="Mutual information (nats)", title="(b) Mutual information (raw columns)")
a2.set_xlim(0, s.max() * 1.3)
fig.tight_layout()
fig.savefig(FIG / "eda_association.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---- 圖 4:特徵之間的冗餘 + VIF ----
corr = tr[num_cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols)), num_cols, rotation=60, ha="right")
ax.set_yticks(range(len(num_cols)), num_cols)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(corr.iat[i, j]) > 0.6 else "black")
fig.colorbar(im, ax=ax, label="Spearman ρ")
fig.tight_layout()
fig.savefig(FIG / "eda_redundancy.png", dpi=200, bbox_inches="tight")
plt.close(fig)

Z = tr[num_cols].fillna(tr[num_cols].median())
Z = ((Z - Z.mean()) / Z.std()).to_numpy()
vif = []
for j, c in enumerate(num_cols):
    others = np.column_stack([np.ones(len(Z)), np.delete(Z, j, axis=1)])
    resid = Z[:, j] - others @ np.linalg.lstsq(others, Z[:, j], rcond=None)[0]
    r2 = 1 - resid.var() / Z[:, j].var()
    vif.append({"feature": c, "vif": round(1 / (1 - r2), 2)})
vif = pd.DataFrame(vif).sort_values("vif", ascending=False)

# ---- 類別平均 ----
gm = []
for c in ["country", "processing_group", "variety", "color"]:
    g = y.groupby(tr[c].fillna("missing")).agg(["size", "mean", "median", "std"]).reset_index()
    g.columns = ["level", "n", "mean", "median", "std"]
    gm.append(g[g["n"] >= 5].sort_values("mean", ascending=False).assign(column=c))
gm = pd.concat(gm)[["column", "level", "n", "mean", "median", "std"]].round(2)

# ---- 摘要 ----
pd.set_option("display.width", 160)
print("\n關聯排名前 8(依 |Spearman ρ|):")
print(assoc.head(8)[["feature", "kind", "n_nonmissing", "pearson_r", "spearman_rho", "spearman_p"]].to_string(index=False))
print("\n原始欄位互資訊(nats;region 等高基數類別會被高估):", mi.sort_values(ascending=False).round(3).to_dict())
off = corr.where(~np.eye(len(num_cols), dtype=bool)).abs().stack()
print(f"\n特徵間最大 |Spearman ρ|:{off.max():.2f}({' vs '.join(off.idxmax())});最大 VIF:{vif['vif'].max()}({vif.iloc[0]['feature']})")
rank_b = assoc[(assoc["kind"] == "raw") & (assoc["dtype_kind"] == "numeric")]["feature"].tolist()
print("\n原始數值欄位依 |Spearman ρ| 排名(給 Set B 參考;正式的 Set B 必須在每個 CV 訓練折內重新挑):", rank_b)
print("\n產國平均(n >= 20):", gm[(gm["column"] == "country") & (gm["n"] >= 20)].set_index("level")["mean"].to_dict())
print(f"\n輸出:{sorted(p.name for p in FIG.glob('*.png'))} → {FIG}")
