# src 程式說明

所有指令都在 **repo 根目錄**執行,依下表順序跑。

| 程式 | 作用 | 讀 → 寫 |
|---|---|---|
| `check_data.py` | 核對原始檔版本與基本統計(唯讀) | 原始檔 → 只印在終端 |
| `clean_step1.py` | 移除錯誤列、統一單位、建立 `row_id` | `data/raw/tang/` → `data/interim/step1.csv` |
| `clean_step2.py` | 統一文字寫法(產區、品種、處理法…) | `step1.csv` → `data/interim/step2.csv` |
| `clean_step3.py` | 建分組鍵、去個資、輸出主檔 | `step2.csv` → `data/processed/master_public.csv`、`data_dictionary.csv` |
| `make_split.py` | 依日期切訓練 / 測試集(最後 20% 為測試) | 主檔 → `data/processed/split.csv` |
| `eda.py` | §4 探索分析(只用訓練集) | 主檔 + 切分 → `docs/eda_figures/*.png` |
| `features.py` | §5 特徵工程模組(見下方) | 不寫檔 |
| `plot_pipeline.py` | 畫管線圖 | → `docs/pipeline.png`、`.svg` |

`data/raw/` 與 `data/interim/` 不進 git;清理規則見 `docs/data_cleaning.md`。

## 範例
```
python src/check_data.py                      # 可加參數指定其他原始檔路徑
python src/clean_step1.py && python src/clean_step2.py && python src/clean_step3.py
python src/make_split.py
python src/eda.py
python src/plot_pipeline.py
```

## features.py
給其他程式 import 用;直接執行相當於test.py(非正式結果)。

| 提供 | 作用 |
|---|---|
| `load_data()` | 回傳 `(train, test)` 兩個 DataFrame |
| `build_features("A")` | Set A:只用原始欄位(97 欄) |
| `build_features("B", k=k)` | Set B:從 Set A 的 97 欄挑出與分數 \|Spearman ρ\| 最高的 k 欄;k 必填,k > 97 時 sklearn 會警告並回傳全部 |
| `build_features("C")` | Set C:原始欄位 + 新特徵(33 欄) |
| `build_features("C", drop_group=g)` | 拿掉特徵做 ablation;`g` 為 `polynomial` / `interactions` / `transforms` / `origin` 之一,或多組的清單,例如 `["transforms", "interactions"]` |
| `TARGET` | 目標欄位名稱 `total_cup_points` |

```python
import sys; sys.path.insert(0, "src")
from features import load_data, build_features, TARGET
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

train, test = load_data()
model = make_pipeline(build_features("C"), Ridge(alpha=1.0))
model.fit(train, train[TARGET])   # 標準化、平均分數編碼等只在這裡用訓練資料學
pred = model.predict(test)        # 每批一個預測分數(分)
```

特徵要包在 Pipeline 裡跟模型一起 fit,不要先算好存檔,否則交叉驗證時會偷看到驗證折的分數。各特徵的理由見 `docs/feature_engineering.md`。
