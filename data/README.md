# 資料說明

## 授權與出處
- 資料來源:Kaggle `erwinhmtang/coffee-quality-institute-reviews-may2023`。
  Kaggle 頁面標示:資料庫採 Open Database License (ODbL) 1.0,內容採 Database Contents License (DbCL) 1.0。
  - ODbL 1.0:https://opendatacommons.org/licenses/odbl/1-0/
  - DbCL 1.0:https://opendatacommons.org/licenses/dbcl/1-0/
- 原始資料出處:Coffee Quality Institute(CQI)。
- 本 repo 的 `data/processed/` 是上述資料庫的改作版本(移除欄位、清理與重新編碼),
  依 ODbL 1.0 的相同方式分享條款,同樣以 ODbL 1.0 提供。
- 授權依 Kaggle 頁面所標示;我們沒有獨立查證上傳者是否擁有再授權的權利。
- 原始檔含個人資料(姓名、電話),不放進 repo。

## 本 repo 內的檔案(`data/processed/`)
| 檔案 | 內容 |
|---|---|
| `master_public.csv` | 處理後的資料,1,508 列、22 欄,含目標 `total_cup_points` |
| `data_dictionary.csv` | 每個欄位的角色、單位、缺失率與備註 |

原始檔含個人資料(姓名、電話),**不放進 repo**。請自行從上述來源下載,放到 `data/raw/tang/`。

## 重現
```
python src/clean_step1.py
python src/clean_step2.py
python src/clean_step3.py
```

處理規則與每個欄位的去向見 [`docs/data_cleaning.md`](../docs/data_cleaning.md)。

## 注意
- 評鑑日涵蓋 2010–2018 與 2022–2023,中間 2019–2021 沒有資料。
- `grading_date`、`grading_year` 只用於切分與切片,不當特徵。