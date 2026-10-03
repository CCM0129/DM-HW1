"""切分訓練集與測試集:依評鑑日取最後 20% 當測試集,切點落在日與日之間(同一天不拆開)。
讀 data/processed/master_public.csv,輸出 data/processed/split.csv(row_id, split)。"""
from pathlib import Path

import pandas as pd

IN = Path("data/processed/master_public.csv")
OUT = Path("data/processed/split.csv")
TEST_FRAC = 0.2

df = pd.read_csv(IN, keep_default_na=False, na_values=[""])
assert df["row_id"].is_unique and df["row_id"].is_monotonic_increasing, "row_id 不唯一或未依評鑑日排序"

n = len(df)
cut = df["grading_date"].iloc[round((1 - TEST_FRAC) * n)]
is_test = df["grading_date"] >= cut            # 切點那天整天都歸測試集
df["split"] = is_test.map({True: "test", False: "train"})
tr, te = df[~is_test], df[is_test]
assert len(tr) == 1203 and len(te) == 305, "切分筆數與文件記載不符"

df[["row_id", "split"]].to_csv(OUT, index=False)

print(f"切點:{cut}(評鑑日 >= 切點者為測試集)→ {OUT}")
for name, s in [("訓練集", tr), ("測試集", te)]:
    print(f"- {name}:{len(s)} 列({len(s) / n:.1%}),評鑑日 {s['grading_date'].min()} 到 {s['grading_date'].max()}")
print("測試集各年筆數:", te["grading_year"].value_counts().sort_index().to_dict())
shared = te["group_id"].isin(tr["group_id"])
print(f"測試集中,分組鍵也出現在訓練集的列:{int(shared.sum())} 列({shared.mean():.1%})")
