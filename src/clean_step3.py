"""Step 3:建立分組鍵、去掉個資與感官分項、輸出可進 repo 的主檔與資料字典。
讀 data/interim/step2.csv,輸出 data/processed/master_public.csv 與 data/processed/data_dictionary.csv。"""
import re
import hashlib
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

IN = Path("data/interim/step2.csv")
OUT_DIR = Path("data/processed")
KEEP_ORG_NAMES = False      # True:主檔多保留農場 / 公司 / 磨坊名稱(仍不含生產者、擁有者、聯絡資料)

PLACEHOLDER_RE = re.compile(r"^(various|varios|varias|vario|multiple|several|many|mixed|mix|n a|na|none|"
                            r"unknown|unspecified|not specified|no data|other|others)\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
FORBIDDEN = {"Producer", "Owner", "Certification_Contact", "Certification_Address", "In_Country_Partner",
             "Lot_Number", "ICO_Number", "Status", "Expiration", "Aroma", "Flavor", "Aftertaste", "Acidity",
             "Body", "Balance", "Uniformity", "Clean_Cup", "Sweetness", "Overall", "Defects",
             "farm_name_raw", "producer", "owner"}


def norm_name(s):
    """農場 / 公司 / 磨坊名稱正規化;占位文字(various、varios 等)視為缺失。"""
    if pd.isna(s):
        return np.nan
    t = unicodedata.normalize("NFKD", str(s))
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t or PLACEHOLDER_RE.match(t):
        return np.nan
    return t


assert IN.exists(), "找不到 data/interim/step2.csv,請先執行 src/clean_step2.py"
# 只把空字串當缺失:避免泰國的 Nan 府(小寫成 "nan")被誤判成缺失值
df = pd.read_csv(IN, keep_default_na=False, na_values=[""])
print(f"讀入 {len(df)} 列\n")

# ---- 分組鍵:農場 > 公司 > 磨坊 > 產區 > 產國;名稱只用來算雜湊,不寫進主檔 ----
farm, comp, mill = (df[c].map(norm_name) for c in ["Farm_Name", "Company", "Mill"])
level, name = [], []
for f, c, m, r in zip(farm, comp, mill, df["region_clean"]):
    if pd.notna(f):
        level.append("farm"); name.append(f)
    elif pd.notna(c):
        level.append("company"); name.append(c)
    elif pd.notna(m):
        level.append("mill"); name.append(m)
    elif pd.notna(r):
        level.append("region"); name.append(r)
    else:
        level.append("country"); name.append("")
df["group_level"] = level
key = df["country"].fillna("unknown").astype(str) + "|" + pd.Series(level, index=df.index) + ":" + pd.Series(name, index=df.index)
hashed = key.map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())
df["group_id"] = pd.factorize(hashed, sort=True)[0]

# ---- 組出主檔 ----
rename = {
    "coffee_id": "source_id", "region_clean": "region", "variety_clean": "variety",
    "processing_method_clean": "processing_method", "color_clean": "color", "cert_body_clean": "cert_body",
    "Harvest_Year": "harvest_year_raw", "Category_One_Defects": "category_one_defects",
    "Category_Two_Defects": "category_two_defects", "Quakers": "quakers", "Total_Cup_Points": "total_cup_points",
}
d = df.rename(columns=rename)
d["grading_date"] = pd.to_datetime(d["grading_date"]).dt.strftime("%Y-%m-%d")
cols = ["row_id", "source_id", "grading_date", "grading_year", "group_id", "group_level",
        "country", "region", "altitude_m", "variety", "processing_method", "processing_group", "color",
        "harvest_year_raw", "harvest_year_start", "moisture_pct", "moisture_suspect",
        "category_one_defects", "category_two_defects", "quakers", "n_bags", "bag_weight_kg",
        "cert_body", "total_cup_points"]
if KEEP_ORG_NAMES:
    d["farm_name"], d["company"], d["mill"] = df["Farm_Name"], df["Company"], df["Mill"]
    cols += ["farm_name", "company", "mill"]
master = d[cols].copy()

# ---- 自我檢查:任何一項失敗就中止,不輸出檔案 ----
assert master["row_id"].is_unique and master["row_id"].is_monotonic_increasing, "row_id 不唯一或未遞增"
assert master["total_cup_points"].between(50, 100).all(), "總分超出 50~100"
assert master["grading_date"].notna().all() and master["group_id"].notna().all(), "日期或分組鍵有缺失"
assert not (FORBIDDEN & set(master.columns)), "主檔含禁止欄位:" + str(FORBIDDEN & set(master.columns))
for c in master.columns:
    s = master[c]
    if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
        v = s.dropna().astype(str)
        has_phone = v.map(lambda t: any(len(re.sub(r"\D", "", m.group())) >= 9 for m in PHONE.finditer(t)))
        assert not v.str.contains(EMAIL).any(), f"{c} 疑似含信箱"
        assert not has_phone.any(), f"{c} 疑似含電話"
print("自我檢查通過:列編號唯一、分數範圍合理、無禁止欄位、無信箱與電話\n")

# ---- 資料字典 ----
meta = {
    "row_id": ("id", "-", "no", "穩定列編號(依評鑑日、來源編號排序);切分檔一律用它對應"),
    "source_id": ("id", "-", "no", "Tang 資料集的原始編號,僅供追溯"),
    "grading_date": ("date", "-", "maybe", "杯測評鑑日;用於時間切分,是否可當特徵須團隊決定(評鑑當下才產生)"),
    "grading_year": ("date", "-", "maybe", "評鑑年;用於切分與切片,不可直接當數值特徵(會外插)"),
    "group_id": ("group", "-", "no", "分組鍵(雜湊後的整數),給 GroupKFold 與群組自助法"),
    "group_level": ("group", "-", "no", "分組鍵使用的層級:farm/company/mill/region/country"),
    "country": ("candidate", "-", "yes", "產國"),
    "region": ("candidate", "-", "yes", "產區(小寫、去重音;稀有值多)"),
    "altitude_m": ("candidate", "m", "yes", "海拔;<=0 或 >3000 已設為缺失"),
    "variety": ("candidate", "-", "yes", "品種(約 60 種標籤;稀有類別由建模階段在訓練折內合併)"),
    "processing_method": ("candidate", "-", "yes", "處理法(原標籤小寫)"),
    "processing_group": ("candidate", "-", "yes", "處理法合併為 washed/natural/honey/semi_washed/other"),
    "color": ("candidate", "-", "yes", "生豆顏色(統一寫法)"),
    "harvest_year_raw": ("reference", "-", "yes", "收成年原始字串(格式混亂)"),
    "harvest_year_start": ("candidate", "year", "yes", "收成年起始西元年;無法解析或矛盾者為缺失;不可直接當數值特徵"),
    "moisture_pct": ("candidate", "%", "yes", "含水率;0 視為缺失;<5% 或 >20% 保留數值並標記"),
    "moisture_suspect": ("flag", "0/1", "yes", "含水率 <5% 或 >20% 的標記(數值未更動)"),
    "category_one_defects": ("candidate", "count", "yes", "第一類瑕疵數"),
    "category_two_defects": ("candidate", "count", "yes", "第二類瑕疵數"),
    "quakers": ("candidate", "count", "maybe", "Quaker 豆數;烘焙後才能數出,是否算送測前已知須團隊決定"),
    "n_bags": ("candidate", "bags", "yes", "批次袋數"),
    "bag_weight_kg": ("candidate", "kg", "yes", "每袋重量(公斤);1~2 kg 多為樣品袋"),
    "cert_body": ("candidate", "-", "maybe", "評鑑機構;評鑑當下才確定且與年代高度相關,須團隊決定"),
    "total_cup_points": ("target", "points", "no", "目標:杯測總分(分)"),
    "farm_name": ("candidate", "-", "yes", "農場名稱(僅在 KEEP_ORG_NAMES=True 時存在)"),
    "company": ("candidate", "-", "yes", "公司名稱(僅在 KEEP_ORG_NAMES=True 時存在)"),
    "mill": ("candidate", "-", "yes", "磨坊名稱(僅在 KEEP_ORG_NAMES=True 時存在)"),
}
rows = [{"column": c, "role": meta[c][0], "unit": meta[c][1], "available_at_prediction": meta[c][2],
         "dtype": str(master[c].dtype), "missing_pct": round(master[c].isna().mean() * 100, 1),
         "n_unique": int(master[c].nunique()), "note": meta[c][3]} for c in master.columns]
dictionary = pd.DataFrame(rows)

OUT_DIR.mkdir(parents=True, exist_ok=True)
master.to_csv(OUT_DIR / "master_public.csv", index=False)
dictionary.to_csv(OUT_DIR / "data_dictionary.csv", index=False, encoding="utf-8-sig")

# ---- 摘要 ----
y = master["total_cup_points"]
g = master["group_id"].value_counts()
print(f"主檔:{len(master)} 列、{master.shape[1]} 欄 → {OUT_DIR / 'master_public.csv'}")
print("欄位:", list(master.columns))
print(f"\n目標:平均 {y.mean():.2f}、標準差 {y.std():.2f}、範圍 {y.min():.2f}~{y.max():.2f};"
      f"低於 80 分 {(y < 80).sum()} 筆;85 分以上 {(y >= 85).sum()} 筆")
print("各年筆數:", master["grading_year"].value_counts().sort_index().to_dict())
print(f"\n分組鍵:{master['group_id'].nunique()} 組;層級分布 {master['group_level'].value_counts().to_dict()}")
print(f"  只有 1 筆的組 {int((g == 1).sum())} 個;最大 5 組的筆數 {g.head(5).tolist()}")
big = master[master["group_id"].isin(g.head(5).index)].groupby("group_id")["group_level"].first()
print("  最大 5 組所屬層級:", [big[i] for i in g.head(5).index])
print("\n缺失率前 8 名(%):", master.isna().mean().mul(100).round(1).sort_values(ascending=False).head(8).to_dict())
print("\n資料字典 available_at_prediction = maybe 的欄位:", dictionary.loc[dictionary["available_at_prediction"] == "maybe", "column"].tolist())
