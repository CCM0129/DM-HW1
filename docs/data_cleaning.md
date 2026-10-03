# 資料處理詳細說明

## Step 1(`src/clean_step1.py`):移除明顯錯誤、統一單位、建立列編號
| 處理 | 影響 | 理由 |
|---|---|---|
| 移除總分 ≤ 0 的列 | **移除 1 列**(1,509 → 1,508) | 該列所有感官分數也是 0,是輸入錯誤 |
| 海拔 < 10 或 > 3000 公尺 → 設為缺失 | 17 列 | 不是合理的咖啡種植海拔;其中 16 列數值在 1–2 之間,疑似以公里填寫 |
| 含水率 = 0 → 設為缺失;小數(0.117)轉成百分比 | 253 列 | 0 不可能存在,是隱藏缺失值;原始檔以小數記錄 |
| 含水率 < 5% 或 > 20% → **保留數值,只標記** `moisture_suspect = 1` | 33 列 | 偏離常見範圍,但沒有文獻依據判定是錯誤,留給後續分析判斷 |
| 袋數 = 0 → 設為缺失 | 1 列 | 無意義 |
| 袋重統一為公斤(磅換算);無單位或空白 → 設為缺失 | 29 列 | 無法判斷是 kg 還是 lbs |
| 袋重 ≤ 0 或 > 200 公斤 → 設為缺失 | 25 列 | 多為把整批總重填成袋重 |
| 依評鑑日排序並建立穩定列編號 `row_id` | 全部 | 切分與結果一律用它對應 |

## Step 2(`src/clean_step2.py`):統一文字寫法(**不移除任何列或欄**)
| 欄位 | 處理 | 結果 |
|---|---|---|
| 國名 | 修正編碼錯誤 `Cote d?Ivoire` | 37 種,不變 |
| 產區 | 轉小寫、去重音與標點 | 456 → 419 種 |
| 品種 | 同上,**不合併** | 60 種,不變 |
| 處理法 | 合併成 washed / natural / honey / semi_washed / other | washed 931、natural 293、semi_washed 57、honey 41、other 30、缺失 156;特殊發酵法(如 anaerobic、wet hulling)歸入 other |
| 生豆顏色 | 統一大小寫與拼字 | 18 種 → 6 種;缺失 267 |
| 收成年 | 取起始西元年(如 `4T/10` → 2010)。只有月份範圍、`TEST`、`mmm` 等無法解析者 → 設為缺失;晚於評鑑年或早於評鑑年 5 年以上者 → 設為缺失 | 原始缺失 47 列 + 無法解析 10 列 + 矛盾 2 列 = 缺失 59 列(3.9%) |

## Step 3(`src/clean_step3.py`):建立分組鍵、去個資、輸出可公開的檔案(**不移除任何列**)
原始 42 欄的去向:

| 去向 | 欄位 | 理由 |
|---|---|---|
| **保留於主檔**(改名或轉換後) | `coffee_id`→`source_id`、`Country_of_Origin`、`Region`、`Altitude`、`Variety`、`Processing_Method`、`Color`、`Harvest_Year`、`Moisture`、`Category_One_Defects`、`Category_Two_Defects`、`Number_of_Bags`、`Bag_Weight`、`Total_Cup_Points`(目標) | 送測前已知的資訊或目標 |
| **保留於主檔,但預設不當特徵** | `Grading_Date`(→ `grading_date`、`grading_year`) | 評鑑本身就是要預測的事件,不是送測前已知。只用於時間切分與切片分析 |
| **只用來建分組鍵,不輸出** | `Farm_Name`、`Company`、`Mill` | 同一農場或公司有多筆批次,切分時要分組以免洩漏;名稱雜湊成整數 `group_id`,不公開名稱 |
| **移除:個人資料** | `Producer`、`Owner`、`Certification_Contact`(姓名與電話)、`Certification_Address` | 個資,對預測沒有用途 |
| **移除:識別碼與組織名稱** | `Lot_Number`、`ICO_Number`、`In_Country_Partner` | 可重新識別個別批次或單位,對預測沒有用途 |
| **移除:感官分項(洩漏)** | `Aroma`、`Flavor`、`Aftertaste`、`Acidity`、`Body`、`Balance`、`Uniformity`、`Clean_Cup`、`Sweetness`、`Overall`、`Defects` | 總分 = 十項感官分項相加 − `Defects`,用了等於把答案給模型 |
| **移除:年代代理** | `Status` | 只有新資料有值(198 列,全為 Completed),舊資料全空,等於「年代標籤」 |
| **移除:評鑑後才產生的行政資訊** | `Expiration`、`parsed_expiration`、`parsed_grading_date`(已轉成 `grading_date`) | 評鑑後才產生,或已被轉換 |
| **保留於主檔,當特徵(團隊決定)** | `Quakers`(→ `quakers`)、`Certification_Body`(→ `cert_body`,評鑑機構) | 團隊討論後決定當特徵:測試資料同樣有這兩欄,預測時可以使用。原本的疑慮:`Quakers` 可能需烘焙後才能數出;評鑑機構可能與產國、年代綁在一起,§8 要檢查 |

另外在 Step 3 建立 `group_id`:依序用農場 > 公司 > 磨坊 > 產區 > 產國,占位文字(various、varios、varias fincas 等)視為缺失;共 842 組。

# 預測時間點與特徵的決定
- **預測時間點**:生豆到貨、完成物理分級,**尚未送去杯測**。
- **只使用此時已知的資訊**。`quakers`、`cert_body` 經團隊討論後決定當特徵。
- `grading_date`、`grading_year` 保留在主檔供切分與切片使用,**預設不作為特徵**;
  因此「豆齡」(評鑑日 − 收成年)這類特徵也不使用。若團隊之後決定改變,須在此更新。

# 注意事項
- 評鑑日涵蓋 2010–2018 與 2022–2023,**中間 2019–2021 沒有資料**。
- 總分分布窄(標準差約 2.6),低於 80 分的批次在新資料很少(2022 年 1 筆、2023 年 0 筆),評估力有限。
- 這份資料是業者主動送評的批次,品質偏高,不能代表所有進口生豆。