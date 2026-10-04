"""畫 §5 要求的管線圖(每個階段的輸入與輸出),輸出 docs/pipeline.png 與 docs/pipeline.svg。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("docs")
STAGES = [  # 標題、內容、寬度、底色
    ("Raw data", "CQI arabica reviews\n(Kaggle, May 2023)\n1,509 lots × 42 columns\ngraded 2010–2018\nand 2022–2023",
     2.7, "#e8eef7"),
    ("Preprocessing", "drop leaky / personal columns\nfix units, normalise labels\nfarm / company group key\ndate split: last 20% = test",
     2.85, "#e9f4ec"),
    ("Feature engineering", "impute + missing flags, standardise\npolynomial: altitude²\n"
     "interactions: altitude × washed,\n  defects × sample bag\ntransforms: log(cat-1 defects),\n  sample-bag & specialty-grade flags,\n"
     "  origin & cert-body target encoding,\n  altitude vs. country median",
     3.9, "#fdf1e3"),
    ("Linear model", "Linear Regression,\nLasso, Ridge\nhyperparameters by\nGroupKFold CV\non the training split only",
     2.4, "#f3eaf6"),
    ("Prediction &\nevaluation", "total cup points (points)\nMAE, RMSE vs. baselines\ntest split used once",
     2.4, "#f6e9e9"),
]
GAP, H, Y0 = 0.55, 2.4, 0.0

fig, ax = plt.subplots(figsize=(15, 4.4))
x, boxes = 0.0, []
for title, body, w, color in STAGES:
    ax.add_patch(FancyBboxPatch((x, Y0), w, H, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec="#4a4a4a", lw=1.2))
    ax.text(x + w / 2, Y0 + H - 0.18, title, ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(x + w / 2, Y0 + H - 0.78, body, ha="center", va="top", fontsize=8.5, linespacing=1.4)
    boxes.append((x, w))
    x += w + GAP
for (x1, w1), (x2, _) in zip(boxes, boxes[1:]):
    ax.add_patch(FancyArrowPatch((x1 + w1 + 0.04, Y0 + H / 2), (x2 - 0.04, Y0 + H / 2),
                                 arrowstyle="-|>", mutation_scale=16, lw=1.4, color="#4a4a4a"))

# 特徵工程與模型:只在訓練折上 fit
(fx, _), (mx, mw) = boxes[2], boxes[3]
ax.add_patch(FancyBboxPatch((fx - 0.15, Y0 - 0.15), mx + mw - fx + 0.3, H + 0.58,
                            boxstyle="round,pad=0.02,rounding_size=0.15", fc="none", ec="#c0392b", lw=1.2, ls="--"))
ax.text((fx + mx + mw) / 2, Y0 + H + 0.33, "fit on training folds only (no test leakage)",
        ha="center", va="center", fontsize=9, color="#c0392b")

ax.set_xlim(-0.2, x - GAP + 0.2)
ax.set_ylim(Y0 - 0.3, Y0 + H + 0.6)
ax.axis("off")
OUT.mkdir(exist_ok=True)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"pipeline.{ext}", dpi=200, bbox_inches="tight")
print("輸出:", OUT / "pipeline.png", OUT / "pipeline.svg")
