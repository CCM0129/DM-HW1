# Screening Specialty Coffee Before Cupping

**杯測前篩選精品咖啡:以產地與生豆分級資訊,用線性模型預測杯測分數**

這是論文中**模型、baseline 與評估**部分(§5.2、§6、§7.1)的程式碼。
資料清理與特徵工程在另一個 repo:[`DM-HW1-feature-feature-engineering`](../DM-HW1-feature-feature-engineering/)。

> 📌 **狀態**:所有交叉驗證(CV)結果已完成。**測試集尚未評估**,依驗證協定,全組定案後只評估一次。

---

## 摘要

精品咖啡進口商要請持證杯測師正式杯測,才知道一批生豆是否達到精品級(≥ 80 分),而杯測昂貴又耗時。
我們**只用杯測前就能取得的資訊**(產國、產區、海拔、品種、處理法、含水率、瑕疵數等)預測 CQI 杯測總分,
並排除所有杯測子項分數(香氣、風味等,加總即為總分,屬資料洩漏)。

在 1,203 筆阿拉比卡批次上,我們以依農場分組的 5 折交叉驗證評估模型:

- **Ridge 迴歸 + 領域特徵**的 CV MAE 為 **1.61 分**,與隨機森林相當(1.62 分;p = 0.58)。
- 比「只看產國平均」好(1.67 分),5 折全贏,但幅度小(p = 0.064)。
- 用作篩選工具時(預測 < 80.5 分就不送杯測),在只錯殺 **3.5%** 好豆的前提下,能擋下 **21.8%** 的低分批次;
  只看產國平均在同一門檻下幾乎擋不到(0.6%)。

![screening trade-off](figures/screening_tradeoff.png)
*圖 1:不同篩選門檻下,「錯殺好豆的比例」與「正確擋下低分批次的比例」的取捨(CV out-of-fold 預測)。
曲線越靠左上越好。*

---

## 主要結果

單位:分。CV 為 5 個 GroupKFold 折的平均 ± 標準差;每欄最佳者加粗。

| 方法 | 特徵 | CV MAE ↓ | CV RMSE ↓ | CV R² ↑ | 低分 AUC ↑ | Test MAE | Test R² |
|---|---|---|---|---|---|---|---|
| 訓練集平均 | — | 1.832 ± 0.157 | 2.662 ± 0.299 | −0.008 | — | TBD | TBD |
| 產國平均 | 產國 | 1.674 ± 0.144 | 2.450 ± 0.296 | 0.143 | 0.693 | TBD | TBD |
| 單特徵 OLS | 海拔 | 1.784 ± 0.180 | 2.591 ± 0.292 | 0.045 | 0.621 | TBD | TBD |
| 梯度提升 | Set A(原始) | 1.795 ± 0.198 | 2.488 ± 0.281 | 0.116 | 0.720 | TBD | TBD |
| 隨機森林 | Set A(原始) | 1.621 ± 0.141 | 2.358 ± 0.315 | 0.209 | 0.756 | TBD | TBD |
| Elastic Net | Set C(本研究) | **1.611** ± 0.167 | 2.352 ± 0.315 | **0.214** | 0.793 | TBD | TBD |
| **Ridge(本方法)** | Set C(本研究) | 1.612 ± 0.165 | **2.352** ± 0.309 | **0.214** | **0.794** | TBD | TBD |

**顯著性(跨折 paired t-test,df = 4)**:Ridge 對訓練集平均、單特徵 OLS、梯度提升都是 p < 0.01;
對產國平均 p = 0.064(5 折全贏);對隨機森林 p = 0.58(打平)。
測試集將以**群組 bootstrap**(以農場為單位重抽 2,000 次)給出 95% 信賴區間。

**篩選門檻 80.5 分下的表現**(CV out-of-fold):

| 方法 | 跳過的批次 | 擋下低分批次 ↑ | 錯殺好豆 ↓ |
|---|---|---|---|
| **Ridge,Set C** | 6.0% | **21.8%** | 3.5% |
| 隨機森林,Set A | 6.8% | 20.0% | 4.7% |
| 產國平均 | 1.4% | 0.6% | **1.5%** |

---

## 方法概要

### 預測設定
| 項目 | 定義 |
|---|---|
| 目標 | 杯測總分 Total Cup Points(滿分 100;精品級門檻 80) |
| 一列 | 一個送 CQI 評鑑的阿拉比卡生豆批次 |
| 預測時點 | 生豆到貨、完成物理分級,**尚未杯測** |
| 排除的欄位 | 10 項感官子分數(加總即為目標)、評鑑日期 |

### 模型選擇:為什麼是 Ridge
- **Set C 有中度共線性**:VIF 最高 11.6;例如袋重與「是否樣品袋」相關 −0.93,第一類瑕疵數與其 log 相關 0.84。
- **OLS 係數因此方向錯誤**:4 個特徵在 OLS 與 Ridge 間正負號相反(例如 OLS 中第一類瑕疵數係數為正)。
  加入 L2 懲罰後,方向回到符合假設。
- **不需要特徵篩選**:每個特徵都來自 EDA 的假設;Elastic Net 的最佳 L1 比例為最小的 0.1,Lasso 也沒有比較好。

| 變體 | 懲罰 | 搜尋範圍 | 選定值 | CV MAE |
|---|---|---|---|---|
| OLS | 無 | — | — | 1.666 |
| Lasso | L1 | α ∈ [10⁻⁴, 10⁰],17 點 | α = 0.178 | 1.618 |
| Elastic Net | L1 + L2 | α ∈ [10⁻⁴, 10¹],21 點 × L1 ∈ {0.1, 0.3, …, 1.0} | α = 0.56,L1 = 0.1 | 1.611 |
| **Ridge** | **L2** | **α ∈ [10⁻², 10⁴],25 點** | **α = 1000** | **1.612** |

<p>
<img src="figures/tuning_ridge.png" width="49%" alt="Ridge tuning curve">
<img src="figures/tuning_enet.png" width="49%" alt="Elastic Net tuning curve">
</p>

*圖 2:調參曲線。Ridge 在 α ≤ 10 時與 OLS 相同,α = 1000 時最低,α 更大則過度收縮。*

### 驗證
1. **測試集**:依評鑑日取最後 20%(2017-05-11 起,305 筆),模擬「用過去預測未來」;所有設計定案後只評估一次。
2. **交叉驗證**:訓練集 1,203 筆,5 折 **GroupKFold**,以農場(`group_id`)分組,同一農場的批次不會同時出現在訓練與驗證折。
   所有方法使用同一份固定的 [`folds.csv`](folds.csv)。
3. **前處理不洩漏**:標準化、補缺值、目標編碼都包在 scikit-learn Pipeline 內,只用各折的訓練資料學習。
4. **重訓**:選定模型以全部訓練資料重新訓練,再於測試集評估。

選 5 折的理由:每折約 240 筆、含 26–44 筆低於 80 分的批次,篩選指標才夠穩定;10 折時單一大農場(40 筆)會佔一折的 1/3。

### 評估指標
| 指標 | 用途 |
|---|---|
| **MAE**(主要) | 平均差幾分,進口商直接看得懂 |
| RMSE | 加重懲罰大誤差(把 78 分誤判成 84 分) |
| R² | 與其他研究比較 |
| 擋下低分 / 錯殺好豆 | 以預測分數篩選時的實際取捨 |
| 低分 AUC | 把低分批次排在後面的能力,與門檻無關 |

---

## Repo 結構

```
DM-HW1-model/
├── src/
│   ├── evaluate.py       # 固定折、指標、共用 cross_validate()
│   ├── baselines.py      # 5 個 baseline
│   ├── models.py         # Ridge / Elastic Net 調參、共用 tune()
│   ├── collinearity.py   # VIF、OLS vs Ridge 係數、各線性變體比較
│   ├── compare.py        # 主結果表、paired t-test、篩選門檻
│   └── final_test.py     # 測試集評估(只執行一次)+ 群組 bootstrap
├── results/              # 所有結果表(CSV / JSON),由程式自動產生
├── figures/              # 所有論文用圖,由程式自動產生
├── folds.csv             # 固定的 CV 折(row_id, fold)
└── run_all.py            # 一行重現全部 CV 結果
```

---

## 安裝

需要 Python ≥ 3.11(pandas 3 的要求)、scikit-learn ≥ 1.8(特徵模組使用 `TargetEncoder(cv=KFold(...))`)。
兩個 repo 放在同一層:

```
DM-HW1/
├── DM-HW1-feature-feature-engineering/   # 資料與特徵工程
└── DM-HW1-model/                         # 本 repo
```

```bash
cd DM-HW1-feature-feature-engineering
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt statsmodels   # macOS / Linux:.venv/bin/python
```

套件版本:`scikit-learn==1.9.1`、`pandas==3.0.6`、`numpy==2.4.6`、`scipy==1.17.1`、`matplotlib==3.11.2`。
若特徵 repo 放在別處,設定環境變數 `DM_FEATURE_REPO` 指向它。

---

## 重現結果

在本 repo 根目錄執行(Windows PowerShell):

```powershell
$PY = "..\DM-HW1-feature-feature-engineering\.venv\Scripts\python.exe"

& $PY run_all.py                     # 全部 CV 結果與圖,約 1 分鐘
& $PY src\final_test.py --dry-run    # 檢查測試流程:以訓練集最後 20% 當假測試集
& $PY src\final_test.py --confirm    # 正式評估測試集(只能執行一次)
```

| 指令 | 產生 | 對應論文 |
|---|---|---|
| `src/evaluate.py` | `folds.csv` | §6 |
| `src/baselines.py` | `results/cv_baselines_*.csv` | §6、§7.1 |
| `src/models.py` | `results/tuning_*.csv`、`best_params.json`、`cv_models_*.csv`、`figures/tuning_*.png` | §5.2 |
| `src/collinearity.py` | `results/vif_setC.csv`、`corr_pairs_setC.csv`、`coef_ols_vs_ridge.csv`、`cv_linear_variants.csv` | §5.2 |
| `src/compare.py` | `results/main_cv_table.csv`、`paired_ttest_cv.csv`、`screening_*`、`figures/screening_tradeoff.png` | §7.1 |
| `src/final_test.py --confirm` | `results/test_metrics.csv`、`test_predictions.csv` | §7.1 |

所有隨機性固定為 `RANDOM_SEED = 42`;數字一律由程式寫入 `results/`,論文中的數字不手動複製。
`final_test.py --confirm` 執行過一次後會拒絕再次執行,避免反覆使用測試集。

### 在自己的實驗中重用(例如特徵消融)
```python
import sys; sys.path.insert(0, "../DM-HW1-model/src")
from evaluate import load_train_folds
from models import tune

train, y, folds = load_train_folds()                       # 與主模型相同的資料與折
curve, best, per_fold = tune("ridge", train, y, folds, feature_set="C", drop_group="origin")
```

---

## 資料

- **來源**:Coffee Quality Institute(CQI)評鑑資料,經 Kaggle
  [`erwinhmtang/coffee-quality-institute-reviews-may2023`](https://www.kaggle.com/datasets/erwinhmtang/coffee-quality-institute-reviews-may2023) 取得。
- **規模**:清理後 1,508 筆阿拉比卡批次(訓練 1,203、測試 305),評鑑日涵蓋 2010–2018 與 2022–2023。
- 清理規則與特徵設計見特徵 repo 的 `docs/data_cleaning.md`、`docs/feature_engineering.md`。
- 原始檔含個人資料(姓名、電話),不放進任何 repo。

---

## 限制

- **選擇偏誤**:送評批次本來就品質偏高,分數集中在 80–85 分(標準差 2.7),R² 因此有限;低分批次少(訓練集 13.7%、測試集 5.2%)。
- **時間漂移**:測試集平均 83.1 分,訓練集 82.2 分;測試集的 R² 可能偏低,應以 MAE 為主。
- **CV 略為樂觀**:調參與比較使用同一組折;測試集分數才是不偏的估計。
- **篩選門檻依 CV 結果選定**(規則:錯殺好豆 ≤ 5%),應在測試集上確認。

---

## 引用

```bibtex
@misc{coffee_cup_score_2026,
  title  = {Screening Specialty Coffee Before Cupping: Predicting Cup Scores
            from Origin and Green-Bean Grading with Linear Models},
  author = {TBD},
  year   = {2026},
  note   = {Data Mining course project}
}
```

## 授權

- 程式碼:TBD(建議 MIT)。
- 資料:衍生自 CQI 資料,依 Kaggle 頁面標示以 [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) 提供,
  詳見特徵 repo 的 `README.md`。

## 致謝

資料原始出處為 [Coffee Quality Institute](https://database.coffeeinstitute.org/),感謝 Kaggle 上傳者 Erwin Tang 整理資料。
