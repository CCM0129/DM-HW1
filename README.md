# DM-HW1:用送測前資訊預測精品咖啡杯測分數

以產地、海拔、品種、處理法等送測前已知的資訊,
預測 CQI 阿拉比卡批次的杯測總分(分),協助進口商決定哪些批次值得送測。

## 資料夾
- `data/`:資料說明、處理後的資料與訓練 / 測試切分(見 `data/README.md`)
- `src/`:資料清理、切分、探索分析與特徵工程程式
- `docs/`:資料清理細節(`data_cleaning.md`)、探索分析與特徵工程(`feature_engineering.md`)、圖表(`eda_figures/`)與管線圖(`pipeline.png`)
- `results/`:探索分析輸出的表格(`results/eda/`)
- `requirements.txt`:套件版本

## 重現
所有指令都在 repo 根目錄執行。資料取得與清理步驟見 `data/README.md`。
```
pip install -r requirements.txt
python src/make_split.py      # 訓練 / 測試切分 → data/processed/split.csv
python src/eda.py             # 探索分析(只用訓練集)→ docs/eda_figures/、results/eda/
python src/features.py        # 特徵工程(Set A / Set C)煙霧測試
python src/plot_pipeline.py   # 管線圖 → docs/pipeline.png
```

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
