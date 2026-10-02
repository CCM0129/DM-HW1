import sys
import hashlib
from pathlib import Path
import pandas as pd

p = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/tang/arabica_coffee_full_table.csv")
df = pd.read_csv(p)
print("檔案:", p, "| SHA-256 前16碼:", hashlib.sha256(p.read_bytes()).hexdigest()[:16])
print("列數 / 欄數:", df.shape)

y = df["Total_Cup_Points"]
d = pd.to_datetime(df["parsed_grading_date"], errors="coerce")
print("\n評鑑日範圍:", d.min().date(), "到", d.max().date(), "| 無法解析:", int(d.isna().sum()))

print("\n各年筆數、平均總分、低於80分筆數:")
t = pd.DataFrame({"year": d.dt.year, "y": y})
print(t.groupby("year")["y"].agg(n="count", mean="mean", lt80=lambda s: int((s < 80).sum())).round(2).to_string())

ok = y > 0
print("\n總分 <= 0 的列數:", int((~ok).sum()))
print("總分(排除 <= 0): 平均 %.2f、標準差 %.2f、最小 %.2f、最大 %.2f"
      % (y[ok].mean(), y[ok].std(), y[ok].min(), y[ok].max()))
print("低於80分: %d (%.1f%%) | 85分以上: %d"
      % ((y[ok] < 80).sum(), (y[ok] < 80).mean() * 100, (y[ok] >= 85).sum()))

cols = ["Altitude", "Variety", "Processing_Method", "Color", "Moisture", "Farm_Name", "Region"]
print("\n缺失率(%):", df[cols].isna().mean().mul(100).round(1).to_dict())
print("含水率 = 0 的列數:", int((df["Moisture"] == 0).sum()))
