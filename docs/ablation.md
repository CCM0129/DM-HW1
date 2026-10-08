# Ablation Study — Coffee Quality Prediction

> **Paper 組交接文件（2026-10-08）**  
> Owner branch: [`ablation-study`](https://github.com/CCM0129/DM-HW1/tree/ablation-study)  
> Status: Seven Ridge ablations reproduced in WSL; five-fold CV, slice analyses, scripts and result CSVs uploaded. **All reported metrics are CV / out-of-fold (OOF), not held-out test scores.**

## 1. 最重要的研究結論

1. **在 A/B/C 三種主要特徵設定中，Set C 的 CV MAE 最低**：Set C **1.6117 ± 0.1653**；A **1.6652 ± 0.1590**；B（k=33）**1.6624 ± 0.1585**。五折中，C 對 A、B 的 MAE 均較低。
2. **Origin-related feature representation 對 Set C 最有貢獻**：`C - origin` 的 MAE 升至 **1.6693**（ΔMAE **+0.0576**），五折都比 Full C 差。但這個比較**不是移除所有產地資訊**；它用 one-hot 取代產地 target encoding，同時移除 `alt_rel_country`。
3. **Polynomial / interaction / transformation groups 的獨立增益有限**：移除後平均 MAE 的差異約在 **−0.0022～+0.0012**，相對五折 SD 很小。尤其 `C - polynomial` 的 MAE **1.6095**，略低於 Full C；因此不要宣稱完整 C 是七組中絕對最佳，或每一種 engineered feature 都有幫助。
4. **統計檢定僅供探索性參考**：這五折訓練樣本彼此重疊，且 α、k 使用 CV 選擇；paired t-test 的 p-value 不是獨立外部驗證，未校正多重比較。

## 2. Experimental setup

| 項目 | 本次實驗設定 |
|---|---|
| Task | 用咖啡生豆／批次相關資訊預測連續杯測分數 `total_cup_points` |
| Data | 1,508 筆；**1,203 training** / **305 chronological holdout test** |
| Validation | 固定 **5-fold GroupKFold**，依 `group_id` 分組；各 setting 共用相同切分 |
| Model | **Ridge Regression**；各 setting 使用相同 α 搜尋流程，各自選擇最佳 α |
| Primary metric | **MAE ↓**（每折計算後回報 mean ± sample SD, `ddof=1`） |
| Secondary metrics | RMSE ↓；R² ↑ |
| Test usage | **本次 Ablation 與 OOF Slice Analysis 未使用 305 筆 holdout test** |

### 三個 feature sets

- **Set A — Raw predictors**：8 個數值欄、2 個低基數類別欄、4 個高基數類別欄（共 14 個原始 predictor 欄），類別以 one-hot 表示。實際 one-hot 展開後維度隨 fold 變動，不能說每折固定 97 欄。
- **Set B — Correlation Top-k**：由 Set A 轉換後欄位，在每個 training fold 內以 `SelectKBest` + absolute Spearman correlation 選擇 Top-k，避免用 validation targets 選特徵。比較 `k={5,10,20,33,50,75,97}`，依 CV MAE 選 **k=33**；33 和 50 的 MAE 幾乎相同（差約 0.00004），因此這個排名不表示 33 在其他資料集一定較佳。
- **Set C — Engineered features**：結合原始數值、非線性／交互作用／轉換特徵，以及以 target encoding 和國家相對海拔表達的產地相關特徵。

> **注意**：Set B 的 k-grid 是前期已進行的探索；目前 `src/ablation.py` 不加參數時預設 `k=33`，不會輸出 k-search 檔。若要在 GitHub 中留下完整挑選證據，執行 `python src/ablation.py --retune-k` 後，另提交新產生的 `results/ablation/set_b_k_search.csv`。

## 3. 七組實驗總表（5-fold CV）

`ΔMAE = 該設定平均 MAE − Full Set C 平均 MAE`；正值表示該設定誤差較高。

| Feature setting | Best α | MAE ± SD ↓ | RMSE ± SD ↓ | R² mean ↑ | ΔMAE vs C |
|---|---:|---:|---:|---:|---:|
| Set A | 56.23 | 1.6652 ± 0.1590 | 2.3996 ± 0.2968 | 0.1808 | +0.0535 |
| Set B (k=33) | 31.62 | 1.6624 ± 0.1585 | 2.4062 ± 0.3072 | 0.1768 | +0.0507 |
| **Set C (Full)** | **1000** | **1.6117 ± 0.1653** | **2.3519 ± 0.3094** | **0.2143** | **0.0000** |
| C - polynomial | 1000 | 1.6095 ± 0.1661 | 2.3512 ± 0.3106 | 0.2149 | −0.0021 |
| C - interactions | 1000 | 1.6129 ± 0.1700 | 2.3605 ± 0.3130 | 0.2090 | +0.0012 |
| C - transforms | 1000 | 1.6098 ± 0.1696 | 2.3542 ± 0.3162 | 0.2132 | −0.0019 |
| C - origin | 316.23 | 1.6693 ± 0.1491 | 2.4082 ± 0.2786 | 0.1740 | +0.0576 |

**消融項目與具體改動：**

| Setting | 具體移除／改動 |
|---|---|
| `C - polynomial` | 移除 `altitude_m^2` |
| `C - interactions` | 移除 `alt_x_washed`、`cat2_x_sample` |
| `C - transforms` | 移除 `log_cat1`、`small_bag`、`specialty_grade` |
| `C - origin` | 移除產地相關 target encoding 和 `alt_rel_country`，產地類別改回 one-hot；**不是刪除 country/region/variety/cert_body 的資訊** |

### Exploratory paired-fold comparisons

使用相同五折的 MAE 配對比較 `other − Set C`：

| 比較 | Mean ΔMAE | Set C MAE 較低的折數 | Two-sided p (uncorrected) |
|---|---:|---:|---:|
| A vs C | +0.0535 | 5/5 | 0.0026 |
| B vs C | +0.0507 | 5/5 | 0.0036 |
| C - polynomial vs C | −0.0021 | 2/5 | 0.2974 |
| C - interactions vs C | +0.0012 | 2/5 | 0.7802 |
| C - transforms vs C | −0.0019 | 2/5 | 0.6117 |
| C - origin vs C | +0.0576 | 5/5 | 0.0259 |

**解讀限制：** 五折不是獨立實驗單位，且同一份 CV 也用於 model / k selection，這些 p-value 僅為描述性／探索性統計；不能當成確認性的顯著性證據，也沒有進行 multiple-testing correction。

## 4. Slice-level analysis（train-only OOF）

每筆 train 樣本的預測值來自**未以該筆樣本訓練的 fold model**，以此計算各 slice 的 MAE；並非 test set 或因果分析。

### Processing method（Full C）

| Processing group | n | OOF MAE ↓ |
|---|---:|---:|
| washed | 739 | 1.612 |
| natural | 228 | 1.683 |
| Missing | 147 | 1.669 |
| semi_washed | 56 | 1.412 |
| other | 22 | 1.182 |
| honey | 11 | 1.246 |

`honey` / `other` 的樣本少，不宜據此斷言模型對這兩類效果較好；各組樣本分布可能不同。

### Altitude（Full C 與 C − origin）

| Altitude band | n | Full C OOF MAE | C − origin OOF MAE | ΔMAE (no origin − C) |
|---|---:|---:|---:|---:|
| ≤1200 m | 324 | 1.651 | 1.703 | +0.052 |
| 1200–1500 m | 330 | 1.744 | 1.758 | +0.015 |
| 1500–1900 m | 290 | 1.198 | 1.283 | +0.085 |
| >1900 m | 23 | 2.523 | 2.787 | +0.265 |
| Missing altitude | 236 | 1.793 | 1.864 | +0.071 |

`>1900 m` 僅 23 筆，且海拔缺失 236 筆；可討論誤差差異，但避免把小樣本切片的差距說成穩定規律。注意 `C − origin` 是**產地特徵表示方法**的改動。

## 5. Paper 可使用的英文段落（draft）

### Experimental setup

We evaluated feature ablations using Ridge regression and fixed five-fold GroupKFold splits on 1,203 training samples. The target was the continuous coffee cupping score (`total_cup_points`). All settings used the same fold assignments and hyperparameter search protocol; supervised feature selection was fitted within training folds. Set A used raw numeric predictors and one-hot-encoded categorical predictors. Set B selected the top 33 encoded features by absolute Spearman correlation, with the value of k chosen from a candidate grid according to cross-validated MAE. Set C added domain-motivated nonlinear, interaction, transformation, and origin-related representations. We report mean and sample standard deviation across folds for MAE, RMSE, and R². The chronological holdout set was not used in the ablation study.

### Ablation results and interpretation

Among the three primary feature settings, Set C achieved the lowest cross-validated MAE (1.612 ± 0.165), compared with Set A (1.665 ± 0.159) and Set B (1.662 ± 0.158). Full Set C outperformed both baselines in all five matched folds. Removing polynomial, interaction, or transformation features changed average MAE by no more than approximately 0.002, suggesting limited incremental benefit under the current Ridge specification. In contrast, replacing origin-related target-encoded representations with one-hot encodings and omitting country-relative altitude increased MAE to 1.669. Thus, the observed improvement was primarily associated with **how origin information was represented**, not with all feature groups equally. The paired-fold tests are exploratory rather than confirmatory because training subsets overlap and model choices were made using the same cross-validation procedure.

### Slice-level analysis and limitations

Out-of-fold errors varied across processing methods and altitude bands. For example, the model had a higher MAE among lots above 1,900 m (2.523; n=23) than among those at 1,500–1,900 m (1.198; n=290), although the high-altitude group was small. The origin-representation ablation increased MAE within each recorded altitude band. These subgroup differences are descriptive and may reflect differences in sample composition or missingness rather than causal effects. Generalization should be checked using the untouched chronological holdout only after the final method is fixed.

## 6. 重現方式與 GitHub 檔案對照

Run from the **model worktree / repository**, with the feature branch/worktree made available via `DM_FEATURE_REPO`:

```bash
cd /mnt/c/Users/iweiw/DM-HW1-model
source ~/dm-hw1-venv/bin/activate
export DM_FEATURE_REPO=/mnt/c/Users/iweiw/DM-HW1-feature
python src/ablation.py
# Optional — rerun candidate k values and save evidence:
# python src/ablation.py --retune-k
```

> 以上是實驗執行者的 WSL 路徑；其他組員需改成自己的環境路徑。`ablation-study` 分支是基於 `model` 分支建立，整合進 `main` 時請先由組員協調合併策略，避免意外帶入其他分支變更。

| GitHub file | 用途 |
|---|---|
| `src/ablation.py` | 重跑七組 Ridge Ablation、輸出總表和 paired comparisons |
| `results/ablation/ablation_summary.csv` | 七組 mean/SD、best α、相對 Full C 的 ΔMAE |
| `results/ablation/ablation_per_fold.csv` | 七組 × 五折的逐折 MAE、RMSE、R² |
| `results/ablation/ablation_paired_tests.csv` | 五折 paired differences + exploratory t-test |
| `results/ridge_setC_processing_oof.csv` | Full C 逐筆 train OOF 預測（供切片重算） |
| `results/ridge_setC_processing_slice.csv` | Processing method 子群 MAE |
| `results/ridge_setC_altitude_slice.csv` | Altitude 子群 MAE / RMSE |
| `results/ablation_altitude_slice.csv` | Full C vs `C - origin` 的海拔子群 MAE |
| `results/ablation/set_b_k_search.csv` | **目前未提交**；只有用 `--retune-k` 重跑後才會生成 |

## 7. 交給 Paper 組前請核對

- **Prediction-time availability / leakage**：小組目前決定保留 `quakers` 和 `cert_body`，但先前資料清理文件曾寫排除。請確認它們在「杯測前預測」的實際情境可取得（尤其 `quakers` 可能涉及烘焙／檢測流程），並同步修正文件；不能只因資料表有欄位便宣稱沒有 leakage。
- **評估描述**：本文件全部成績為 train-only CV/OOF，**不是 305 筆 held-out test 分數**；test 應在完整模型選擇與論文分析固定後才評估一次。
- **統計敘述**：paired t-tests 為探索性，勿以 p < 0.05 宣稱因果或獨立確認的統計顯著性。
- **產地消融命名**：`C - origin` 應稱 *origin representation ablation*，而不是 *remove all origin information*。
- **k 的選擇**：如需完整重現依據，補跑 `--retune-k` 並 commit 產生的 `set_b_k_search.csv`。
- **與模型組整合**：ACL-style 六頁上限，表格可裁切為 A/B/C + 4 組 ablations；其餘配對檢定及細切片數據放分析文字或補充檔案。
