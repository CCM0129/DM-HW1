# 資料說明

## 來源
- 原始資料:Coffee Quality Institute(CQI)阿拉比卡杯測紀錄。
- 使用版本:Kaggle `erwinhmtang/coffee-quality-institute-reviews-may2023` 的 `arabica_coffee_full_table.csv`
  (1,509 列、42 欄),下載日期 2026-10-02,SHA-256 前 16 碼 `40abd2ae65af1456`。
- 授權:【待確認】。CQI 原始資料的權利屬 CQI。

## 本 repo 內的檔案(`data/processed/`)
| 檔案 | 內容 |
|---|---|
| `master_public.csv` | 處理後的資料,1,508 列、22 欄,含目標 `total_cup_points` |
| `data_dictionary.csv` | 每個欄位的角色、單位、缺失率與備註 |

原始檔含個人資料(姓名、電話)且授權待確認,**不放進 repo**。請自行從上述來源下載,放到 `data/raw/tang/`。

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