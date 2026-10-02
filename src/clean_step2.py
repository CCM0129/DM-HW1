"""Step 2:統一文字寫法(國名、產區、品種、處理法、顏色、評鑑機構、收成年)。
讀 data/interim/step1.csv,輸出 data/interim/step2.csv(不進 git)。原始欄位保留,新欄位另外加。"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

IN = Path("data/interim/step1.csv")
OUT = Path("data/interim/step2.csv")

PROCESS_MAP = {
    "washed / wet": "washed",
    "natural / dry": "natural",
    "pulped natural / honey": "honey",
    "honey,mossto": "honey",
    "semi-washed / semi-pulped": "semi_washed",
    "semi washed": "semi_washed",
}
COLOR_MAP = {
    "green": "green",
    "bluishgreen": "blue_green", "bluegreen": "blue_green",
    "greenish": "greenish",
    "yellowgreen": "yellow_green", "yellogreen": "yellow_green",
    "paleyellow": "yellow", "yellowish": "yellow",
    "brownish": "brownish", "browishgreen": "brownish",
}


def norm_text(s):
    """小寫、去重音、去標點、壓縮空白;保留中日韓文字。"""
    if pd.isna(s):
        return np.nan
    t = unicodedata.normalize("NFKD", str(s))
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if t else np.nan


def clean_country(s):
    if pd.isna(s):
        return np.nan
    t = re.sub(r"\s+", " ", str(s)).strip()
    return t.replace("Cote d?Ivoire", "Cote d'Ivoire")      # 編碼錯誤


def parse_harvest_year(raw):
    """取收成年(季)的起始西元年;無法判斷回傳 NaN。"""
    if pd.isna(raw):
        return np.nan
    s = str(raw).strip()
    m = re.search(r"(19|20)\d{2}", s)                              # 2021 / 2022、2013/2014、2010-2011
    if m:
        return float(m.group(0))
    m = re.search(r"\b(\d{2})\s*/\s*(\d{2})\s*crop", s, flags=re.I)  # 08/09 crop
    if m:
        return 2000.0 + int(m.group(1))
    m = re.search(r"[A-Za-z]{3}-(\d{2})\b", s)                     # Mar-10
    if m:
        return 2000.0 + int(m.group(1))
    m = re.search(r"\dT\s*/\s*(\d{2})\b", s)                       # 4T/10
    if m:
        return 2000.0 + int(m.group(1))
    return np.nan                                                  # Mayo a Julio 等


df = pd.read_csv(IN)
print(f"讀入 {len(df)} 列\n")

# 國名、產區、品種、評鑑機構
df["country"] = df["Country_of_Origin"].map(clean_country)
df["region_clean"] = df["Region"].map(norm_text)
df["variety_clean"] = df["Variety"].map(norm_text)
df["cert_body_clean"] = df["Certification_Body"].map(lambda x: np.nan if pd.isna(x) else re.sub(r"\s+", " ", str(x)).strip())
for label, raw, new in [("國名", "Country_of_Origin", "country"), ("產區", "Region", "region_clean"),
                        ("品種", "Variety", "variety_clean"), ("評鑑機構", "Certification_Body", "cert_body_clean")]:
    print(f"- {label}:不同值 {df[raw].nunique()} → {df[new].nunique()}(缺失 {int(df[new].isna().sum())} 列)")

# 處理法
pm = df["Processing_Method"].map(lambda x: np.nan if pd.isna(x) else str(x).strip().lower())
df["processing_method_clean"] = pm
df["processing_group"] = pm.map(lambda x: np.nan if pd.isna(x) else PROCESS_MAP.get(x, "other"))
print("\n處理法合併後:", df["processing_group"].value_counts(dropna=False).to_dict())
print("  被歸入 other 的原標籤:", sorted(pm[df["processing_group"] == "other"].dropna().unique()))

# 顏色
key = df["Color"].map(lambda x: np.nan if pd.isna(x) else re.sub(r"[^a-z]", "", str(x).lower()))
df["color_clean"] = key.map(lambda x: np.nan if pd.isna(x) else COLOR_MAP.get(x, "other"))
print("\n顏色合併後:", df["color_clean"].value_counts(dropna=False).to_dict())
print("  原始不同寫法 %d 種 → %d 種" % (df["Color"].nunique(), df["color_clean"].nunique()))

# 收成年
df["harvest_year_start"] = df["Harvest_Year"].map(parse_harvest_year)
unparsed = df.loc[df["Harvest_Year"].notna() & df["harvest_year_start"].isna(), "Harvest_Year"]
suspect = (df["harvest_year_start"] > df["grading_year"]) | (df["harvest_year_start"] < df["grading_year"] - 5)
df["harvest_year_suspect"] = suspect.astype(int)
df.loc[suspect, "harvest_year_start"] = np.nan
print("\n收成年:原始 %d 種寫法;原始缺失 %d 列;有值但無法解析 %d 列 %s"
      % (df["Harvest_Year"].nunique(), int(df["Harvest_Year"].isna().sum()), len(unparsed), sorted(unparsed.unique())))
print("  晚於評鑑年或早於評鑑年 5 年以上(設為缺失):", int(suspect.sum()), "列")
print("  最後缺失 %d 列(%.1f%%)" % (df["harvest_year_start"].isna().sum(), df["harvest_year_start"].isna().mean() * 100))

df.to_csv(OUT, index=False)
print(f"\n輸出 {len(df)} 列、{df.shape[1]} 欄 → {OUT}")
