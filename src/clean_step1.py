"""Step 1:移除明顯錯誤、統一單位、建立穩定列編號。只讀原始檔,輸出到 data/interim/(不進 git)。"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/tang/arabica_coffee_full_table.csv")
OUT = Path("data/interim/step1.csv")
LB_TO_KG = 0.45359237
log = []


def note(rule, n, why):
    log.append((rule, int(n), why))


def parse_bag_kg(x):
    """'69 kg' -> 69.0;'100 lbs' -> 45.36;沒有單位(如 '6')或空白 -> NaN"""
    if pd.isna(x):
        return np.nan
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(kg|kgs|lbs|lb)?\s*$", str(x).lower())
    if not m or m.group(2) is None:
        return np.nan
    v = float(m.group(1))
    return v * LB_TO_KG if m.group(2).startswith("lb") else v


df = pd.read_csv(RAW)
n0 = len(df)

# 1. 總分 <= 0
bad = df["Total_Cup_Points"].fillna(0) <= 0
note("移除總分 <= 0 的列", bad.sum(), "該列所有感官分數也是 0,是輸入錯誤")
df = df[~bad].copy()

# 2. 日期與穩定列編號
df["grading_date"] = pd.to_datetime(df["parsed_grading_date"], errors="coerce")
assert df["grading_date"].notna().all(), "有無法解析的評鑑日"
df = df.sort_values(["grading_date", "coffee_id"], kind="mergesort").reset_index(drop=True)
df["row_id"] = np.arange(len(df))
df["grading_year"] = df["grading_date"].dt.year

# 3. 海拔
alt = df["Altitude"].astype(float)
bad_alt = (alt < 10) | (alt > 3000)
note("海拔 < 10 或 > 3000 公尺 → 缺失", bad_alt.sum(), "不是合理的咖啡種植海拔;1~2 的值疑似以公里填寫")
df["altitude_m"] = alt.where(~bad_alt)

# 4. 含水率(原始是小數,如 0.117)
m = df["Moisture"].astype(float)
zero = m <= 0
note("含水率 = 0 → 缺失", zero.sum(), "0 不可能存在,是隱藏缺失值")
df["moisture_pct"] = (m.where(~zero) * 100).round(2)
df["moisture_suspect"] = ((df["moisture_pct"] < 5) | (df["moisture_pct"] > 20)).astype(int)
note("含水率 < 5% 或 > 20%(只標記,數值保留)", df["moisture_suspect"].sum(), "偏離常見範圍但無文獻依據,不確定是錯誤,留給後續分析判斷")

# 5. 袋數與袋重
nb = df["Number_of_Bags"].astype(float)
bad_nb = nb <= 0
note("袋數 = 0 → 缺失", bad_nb.sum(), "無意義")
df["n_bags"] = nb.where(~bad_nb)

bw = df["Bag_Weight"].map(parse_bag_kg)
note("袋重無法解析 → 缺失", bw.isna().sum(), "原始為空白或沒有單位(如 '6'),無法判斷是 kg 還是 lbs")
bad_bw = (bw <= 0) | (bw > 200)
note("袋重 <= 0 或 > 200 kg → 缺失", bad_bw.sum(), "多為把整批總重(如 19250 kg)填成袋重")
df["bag_weight_kg"] = bw.where(~bad_bw)

# 輸出
OUT.parent.mkdir(parents=True, exist_ok=True)
assert df["row_id"].is_unique
df.to_csv(OUT, index=False)

print(f"原始 {n0} 列 → 輸出 {len(df)} 列,{df.shape[1]} 欄 → {OUT}\n")
for rule, n, why in log:
    print(f"- {rule}:{n} 列({why})")
print("\n評鑑日範圍:", df["grading_date"].min().date(), "到", df["grading_date"].max().date())
print("row_id 範圍:", int(df["row_id"].min()), "到", int(df["row_id"].max()))
print("含水率缺失 %.1f%% | 海拔缺失 %.1f%% | 袋重缺失 %.1f%%"
      % (df["moisture_pct"].isna().mean() * 100, df["altitude_m"].isna().mean() * 100,
         df["bag_weight_kg"].isna().mean() * 100))
