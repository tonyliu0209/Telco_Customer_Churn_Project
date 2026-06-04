# 📞 Telco Customer Churn 預測與可解釋分析專案
> 透過 StratifiedKFold、Class Weight、Hyperparameter Tuning 與 Threshold Tuning 等策略，逐步提升模型對流失客戶的辨識能力，最終將 Random Forest 的 Churn Recall 提升至 **0.807**，並整合 SHAP 與 LIME 提供可解釋分析。

🔗 Live Demo: https://telcocustomerchurnproject-tpn5nfeyb5asju7k4uixs4.streamlit.app/

---

## 📌 專案概述（Project Overview）

### 🔹 背景問題
客戶流失將提高企業的**獲客成本（Customer Acquisition Cost）**，並降低長期**客戶終身價值（Customer Lifetime Value）**。若無法有效識別高風險客戶，企業需持續投入資源開發新客戶，進而影響營收穩定性。

### 🔹 專案目標
針對電信產業客戶流失問題，以「提升流失客戶辨識能力（Recall）」為核心，透過四階段系統性優化策略建立預測模型，並提供可解釋的決策洞察，作為精準留客行動的依據。

### 🔹 方法
- 比較 **Logistic Regression、Random Forest、XGBoost** 三種模型
- 運用 **StratifiedKFold、class_weight、Optuna、Threshold Tuning** 系統性優化
- 使用 **F1-score 與 Precision-Recall Curve** 作為主要評估指標
- 整合 **SHAP（全域解釋）與 LIME（個人化解釋）** 提升模型可信度

### 🔹 專案價值
提前識別高流失風險客戶，協助業務團隊在對的時間介入，支援資料驅動的精準行銷與客戶關懷策略。

---

## 📊 資料說明（Dataset）

| 項目 | 內容 |
|------|------|
| 資料來源 | IBM Telco Customer Churn Dataset（Kaggle） |
| 資料筆數 | 7,043 筆客戶資料，共 21 個欄位 |
| 目標變數 | `Churn`（Yes / No） |
| 類別分布 | 留存 73.5% / 流失 26.5%（存在類別不平衡） |

---

## ⚙️ 建模流程（Modeling Approach）

### 1️⃣ 資料前處理

- 將 `TotalCharges` 欄位強制轉型並填補空值
- 將 Yes/No 二元特徵轉換為 0/1 編碼；多分類變數套用 One-Hot Encoding
- 以 **`Pipeline` 封裝** StandardScaler + OneHotEncoder，確保 Scaler 在每個 CV Fold 僅對訓練集 fit，**杜絕 Data Leakage**

### 2️⃣ 特徵工程

| 特徵名稱 | 說明 |
|---|---|
| `high_monthly_charge` | MonthlyCharges > 70 → 1 |
| `num_core_services` | PhoneService + InternetService 加總（範圍 0–2） |
| `high_charge_short_tenure` | MonthlyCharges > 70 且 tenure < 12 個月 |
| `high_charge_no_contract` | MonthlyCharges > 70 且 Contract = Month-to-month |

### 3️⃣ 四階段系統性優化

| 階段 | 方法 | 關鍵成果 |
|------|------|---------|
| **① Baseline** | 三模型預設參數 | RF 因過擬合 F1 最低（0.537），Recall 僅 0.492 |
| **② 權重調整** | StratifiedKFold + class_weight / scale_pos_weight | LR Recall 大幅提升（0.47→0.80）；RF 受過擬合影響，改善有限 |
| **③ 超參數搜尋** | GridSearchCV / RandomizedSearchCV | RF 關鍵突破：F1 從 0.529 躍升至 0.618，Recall 從 0.476 提升至 0.751 |
| **④ 精細調優** | Optuna 貝葉斯優化 + Threshold Tuning | Threshold=0.50：F1=0.632、Recall=0.765；下調至 0.44 後：F1=0.629、Recall=**0.807** |

### 4️⃣ 評估指標

| 指標 | 說明 |
|------|------|
| **Recall** | 衡量模型辨識實際流失客戶的能力（主要目標） |
| **Precision** | 預測為流失中實際流失的比例 |
| **F1-score** | Precision 與 Recall 的調和平均（超參數搜尋優化指標） |
| **PR Curve** | 評估模型在類別不平衡資料下的整體辨識能力 |

> 業務場景以「捕捉潛在流失客戶」為優先，同時需兼顧資源成本，因此以 **F1-score** 作為超參數搜尋的優化指標，最終再以 **Threshold Tuning** 進一步提升 Recall。

---

## 📈 最終模型效能（Model Performance）

### 🔹 最終模型：Random Forest（Optuna 調參版）+ Threshold = 0.44

| 指標 | 數值 |
|------|------|
| Churn Recall | **0.807** |
| Churn Precision | 0.515 |
| Churn F1-Score | 0.629 |
| Accuracy | 0.750 |

### 🔹 三模型四版本演進比較（X_test，Threshold = 0.50）

| 模型 | Baseline F1 | StratifiedKFold F1 | SearchCV F1 | Optuna F1 |
|------|------------|-------------------|------------|----------|
| LR   | 0.538 | 0.614 | 0.613 | 0.614 |
| RF   | 0.537 | 0.529 | 0.618 | **0.632** |
| XGB  | 0.566 | 0.578 | 0.616 | 0.616 |

### 🔹 Precision-Recall Curve（RF 最終模型）
![PR Curve](RF_pr_curve.png)

### 🔹 SHAP 特徵重要性 — Beeswarm（RF 最終模型）
![SHAP Beeswarm](RF_shap_beeswarm.png)

### 🔹 關鍵洞察

- **客戶年資（tenure）** 是最關鍵的留存指標，年資越短流失風險越高；建議在入網前 12 個月內加強客戶關懷
- **合約類型** 影響顯著，Month-to-month 按月合約客戶流失傾向最強；長期合約能有效降低流失風險
- **高月費（MonthlyCharges > 70 美元）** 且無長期合約的客戶，流失率顯著較高
- 將決策門檻下調至 **0.44**，Recall 從 0.765 提升至 **0.807**，F1 僅損失 0.003；實際應用中可根據資源與風險承受度進行微調

---

## 🔍 關鍵發現與實驗心得（Key Findings & Lessons Learned）

### 1. 系統性優化比單純更換模型更重要

本專案並非直接追求更複雜的模型，而是透過「StratifiedKFold → 類別權重調整 → Hyperparameter Tuning → Threshold Tuning」的系統性流程逐步改善模型表現。

從三模型四版本比較表可以觀察到，評估流程設計與模型調校策略對最終成果的影響，往往大於單純更換演算法。

---

### 2. 特徵工程需要實驗驗證，直覺設計不一定最優

`num_core_services` 特徵歷經三個版本的實驗比較：

| 版本 | 定義 | RF Optuna F1 |
|------|------|-------------|
| A（最終採用） | `num_core_services`：PhoneService + InternetService（範圍 0–2） | **0.632** |
| B | 核心服務 + 加值服務（共 9 項） | 0.632 |
| C | 加值服務（OnlineSecurity、TechSupport 等 7 項） | 0.629 |

版本 B 與 A 幾乎相同，版本 C 反而略低。原因在於加值服務（OnlineSecurity、TechSupport 等）與 `MonthlyCharges` 存在高度共線性——使用越多加值服務，月費自然越高——因此無法提供額外的預測訊號。最終採用最精簡的版本 A，以 `num_core_services` 作為正式特徵名稱。

---

### 3. Hyperparameter Tuning 存在報酬遞減現象

在完成 GridSearchCV / RandomizedSearchCV 後，Optuna 帶來的額外提升相當有限：

| 模型 | SearchCV CV F1 | Optuna CV F1 | 提升幅度 |
|------|--------------|-------------|---------|
| LR   | 0.620 | 0.621 | +0.001 |
| RF   | 0.628 | 0.631 | +0.003 |
| XGB  | 0.631 | 0.631 | ≈ 0    |

這代表當模型已接近資料集可學習的上限時，更進階的搜尋方法未必能帶來顯著收益。

---

### 4. Random Forest 的改善幅度最大

Random Forest 在預設參數下出現明顯過擬合，導致 Recall 與 F1 表現不佳。透過限制樹深度（`max_depth`）、調整特徵抽樣比例（`max_features`）以及 `balanced_subsample` 權重策略後，X_test F1 從 Baseline 的 **0.537** 提升至 Optuna 的 **0.632**，成為三個模型中改善幅度最大的。

這說明模型診斷與參數調校的重要性，有時遠高於更換模型本身。

---

### 5. Threshold Tuning 的效益高於預期

相較於 Optuna 僅帶來約 0.003 的 CV F1 提升，Threshold Tuning 對 Recall 的改善更為顯著：

| 門檻 | Recall | F1 |
|------|--------|-----|
| 0.50（預設） | 0.765 | 0.632 |
| 0.44（業務最優） | **0.807** | 0.629 |

僅犧牲極少量 F1（−0.003），即可大幅提升流失客戶辨識能力。此結果顯示，在類別不平衡問題中，模型後處理（Post-processing）同樣是重要的優化手段。

---

### 6. 模型選擇應基於多維度評估，而非單一指標

CV F1 幾乎相同（RF 0.631 vs XGB 0.631），然而在獨立測試集上，**RF 的 Churn F1 與 Accuracy 仍略優於 XGB**：

| 考量維度 | RF | XGB |
|---------|----|----|
| X_test Churn F1 | **0.63** | 0.62 |
| X_test Accuracy | **0.76** | 0.74 |
| 調參耗時（Optuna 50 trials） | 582 秒 | **230 秒** |
| 部署大小 | 較大 | **較小** |

> ⚠️ 調參耗時為本機實測結果，實際數值受硬體環境與搜尋空間設定影響，僅供相對比較參考。

值得注意的是，若以純粹 Recall 最大化為目標，XGB（Threshold=0.44 下 Recall 0.837）略高於 RF（0.807）。但流失預警同時需控制誤判率，過高的 False Positive 會浪費有限的挽留資源。因此本專案以 **F1-score** 作為模型選擇依據，並透過 **Threshold Tuning** 在部署時動態調整 Recall 與 Precision 的平衡點。

> 模型選擇反映的是業務優先順序的權衡，而非單純追求某一指標的最大化。

---

## 🔧 工程挑戰與除錯記錄（Engineering Challenges & Debugging）

專案開發過程中遭遇並解決了以下幾項工程問題，這些經歷說明，建立可靠的機器學習系統不僅需要建模能力，同樣需要工程嚴謹度：

- **防止 Data Leakage**：透過 scikit-learn `Pipeline` 封裝前處理步驟，確保 `StandardScaler` 僅在每個 CV Fold 的訓練集上 fit，避免驗證集資訊污染 scaler 的參數。
- **修正 ColumnTransformer 設定錯誤**：Binary 特徵未設定 `passthrough`，導致欄位被意外排除；修正後模型才能正確接收完整特徵。
- **修正 Streamlit 快取失效問題**：`get_test_probs` 函式以底線前綴參數 `_pipe` 導致 Streamlit 跳過 hash，切換模型時快取不失效，PR 曲線顯示舊模型數據；改以模型名稱字串為 cache key 後解決。
- **修正 Session State 行為**：調整 Threshold Slider 時，已抽取的客戶資料會因頁面重新執行而消失；以 `st.session_state` 儲存抽樣資料後，Slider 調整可即時更新預測，同時保留當前客戶資料。

---

## 🚀 Streamlit 互動展示

> 本專案建置互動式 Web 應用程式，整合動態預測、即時 PR 曲線追蹤與 LIME 個人化解釋，完整展示模型的預測與可解釋能力。

### ⚙️ 全域控制側邊欄（Sidebar）
- **模型選擇器**：支援 LR / RF / XGBoost 即時切換，全 Tab 同步連動
- **決策門檻滑桿**：範圍 0.10～0.90，步長 0.01，預設業務最優值 **0.44**

### 🔹 Tab 1：專案概覽
- 業務背景說明與核心成果指標卡片（Recall、Precision、F1、Accuracy）
- 四階段實驗方法論表格
- 資料集前 5 筆樣本預覽
- 三模型四版本歷史指標比較表

### 🔹 Tab 2：預測分析（動態診斷核心）
- 隨機抽取一筆客戶資料（Session State 確保資料不因 Slider 調整而消失）
- **核心預測報告**：實際標籤 ｜ 動態預測結果與風險等級 ｜ 關鍵特徵原始值
- **動態 PR 曲線**：以紅點即時追蹤當前門檻在曲線上的 Precision / Recall 座標
- **LIME 個人化特徵診斷**：計算影響該客戶預測分數的前 6 大特徵正負向權重

### 🔹 Tab 3：模型解釋
- Feature Importance、SHAP Beeswarm、PR Curve（均為 RF 最終模型）
- 業務洞察摘要：年資、合約類型、高月費三大流失驅動因子與動態門檻策略說明

---

## 🛠 技術架構（Tech Stack）

| 類別 | 工具 |
|------|------|
| 資料處理 | pandas、numpy |
| 建模 | scikit-learn（Pipeline、ColumnTransformer、StratifiedKFold、GridSearchCV、RandomizedSearchCV、Logistic Regression、Random Forest）、XGBoost |
| 超參數優化 | Optuna |
| 模型解釋 | SHAP（全域特徵重要性）、LIME（個人化局部解釋） |
| 視覺化 | matplotlib、seaborn |
| 模型儲存 | joblib |
| 部署 | Streamlit |

---

🔗 English version available: [README_EN.md](README_EN.md)
