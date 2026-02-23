# 📞 Telco Customer Churn 預測與解釋分析專案
> **透過機器學習識別高風險流失客戶，並利用 SHAP 提供可解釋之決策建議。**

🔗 Live Demo: https://telcocustomerchurnproject-tpn5nfeyb5asju7k4uixs4.streamlit.app/

---

## 📌 專案概述（Project Overview）
### 🔹 背景問題：
- 客戶流失將提高企業的**獲客成本（Customer Acquisition Cost）**，並降低長期**客戶終身價值（Customer Lifetime Value）**。若無法有效識別高風險客戶，企業需持續投入資源開發新客戶，進而影響營收穩定性。

### 🔹 專案目標：
- 透過探索式資料分析（EDA）找出影響電信客戶流失之關鍵因素，並建立預測模型，以識別潛在流失風險客戶。

### 🔹 方法：
- 比較 **Logistic Regression、Random Forest、XGBoost** 三種模型於類別不平衡資料下之預測效能  
- 使用 **F1-score 與 Precision-Recall Curve** 作為評估指標  
- 透過 **SHAP** 進行模型可解釋性分析  

### 🔹 專案價值：
- 協助企業提前辨識高風險流失客戶，作為精準行銷與客戶關懷策略之決策依據，提升客戶留存率。

---

## 📊 資料說明（Dataset）
### 🔹資料來源：
- Kaggle Telco Customer Churn Dataset
### 🔹資料筆數：
- 7,043 筆客戶資料，共 21 個欄位
### 🔹目標變數：
- Churn（Yes / No）
### 🔹類別分布：
| 類別 | 筆數 | 比例 |
| :--- | :--- | :--- |
| 留存客戶 | 5,174 | 73.5% |
| 流失客戶 | 1,869 | 26.5% |

---

## ⚙️ 建模流程（Modeling Approach）
### 1️⃣ 資料前處理：
- 將 Yes/No 類型特徵轉換為 0/1（二元編碼）
- 對多分類變數進行 One-Hot Encoding（避免引入順序偏誤）

### 2️⃣ 特徵工程（Feature Engineering）：
- 建立高月費指標特徵
- 建立交互特徵：
  - MonthlyCharges × Tenure
  - MonthlyCharges × Contract 類型
- 將多個子服務項目（如 OnlineSecurity、TechSupport、StreamingTV 等）整合為 num_services 特徵，以：
  - 降低維度
  - 提升模型泛化能力
  - 減少可能的共線性問題

### 🔹 類別不平衡處理：
- 未進行重採樣（如 SMOTE）
- 選擇適合不平衡資料之評估指標（F1-score 與 PR Curve）進行模型比較

### 🔹 評估指標選擇理由：
- Recall：衡量模型辨識實際流失客戶的能力
- Precision：衡量被預測為流失客戶中實際流失的比例
- F1-score：在 Precision 與 Recall 之間取得平衡
- 由於專案目標為找出潛在流失客戶，同時避免過度誤判穩定客戶，因此選擇 F1-score 作為主要評估指標。

---

## 📈 模型效能與解釋（Model Performance & Interpretation）
### 🔹 模型評估比較
![Model Comparison](images/model_comparison.png)

### 🔹 Precision-Recall Curve
- 用於評估模型於類別不平衡資料下之辨識能力。
![PR Curve](images/XGB_pr_curve_churn.png)

### 🔹 SHAP 特徵重要性（Beeswarm）
- 呈現全域特徵對模型預測結果之影響程度與方向。
![SHAP Beeswarm](images/XGB_beeswarm_churn.png)

### 🔹 關鍵洞察
- Tenure 與 MonthlyCharges 為影響流失的重要因素
- 長期合約可有效降低流失風險
- 使用電子支票付款之客戶流失比例較高

---

## 🚀 Streamlit 互動展示
> 本專案建置互動式 Web 應用程式，展示模型預測與解釋能力。

### 🔹 Tab 1：專案概覽
- 顯示資料集前五筆資料（Dataset Preview）
- 比較模型評估指標
- 標示最佳模型

### 🔹 Tab 2：預測分析（互動核心）
- 使用者可選擇模型
- 系統隨機抽取一筆客戶資料
- 顯示實際標籤 vs 模型預測結果
- 顯示預測機率與風險等級
- 呈現前三大 SHAP 關鍵影響特徵

### 🔹 Tab 3：模型解釋
- Precision-Recall Curve
- SHAP 全域特徵重要性圖
- 關鍵洞察摘要

---

## 🛠 技術架構（Tech Stack）
### 資料處理：
- pandas
- numpy

### 模型建構：
- scikit-learn (Logistic Regression、Random Forest)
- XGBoost

### 模型解釋：
- SHAP

### 視覺化：
- matplotlib
- seaborn

### 模型儲存：
- joblib

### 部署：
- Streamlit

---

🔗 English version available: [README_EN.md](README_EN.md)
