import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import platform

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve


# 偵測作業系統並設定字體
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 微軟正黑體
else:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac/Linux 常用
plt.rcParams['axes.unicode_minus'] = False # 解決負號變框框

# 設定頁面標題
st.set_page_config(page_title="Telco Customer Churn 預測儀表板", layout="wide")

model_map = {
    "Logistic Regression": "lr_model.pkl",
    "Random Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl"
}

# --- A. 特徵工程函式 (確保強健性) ---
def add_features(df):
    df = df.copy()

    # 0. 先處理目標變數
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # 1. 處理 TotalCharges：最容易噴 isnan 的地方
    if "TotalCharges" in df.columns:
        # errors='coerce' 會把空格轉為 NaN，然後我們填補為 0
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # 2. 建立新特徵 (加入型別檢查，避免重複執行出錯)
    # 只在欄位還是字串時才進行比較
    if "MonthlyCharges" in df.columns:
        df["high_monthly_charge"] = (df["MonthlyCharges"] > 70).astype(int)

    if "PhoneService" in df.columns and "InternetService" in df.columns:
        # 簡易版的服務計數
        df["num_services"] = (df["PhoneService"] == "Yes").astype(int) + \
                             (df["InternetService"] != "No").astype(int)

    if "tenure" in df.columns and "MonthlyCharges" in df.columns:
        df["high_charge_short_tenure"] = (
            (df["MonthlyCharges"] > 70) & (df["tenure"] < 12)
        ).astype(int)

    if "MonthlyCharges" in df.columns and "Contract" in df.columns:
        df["high_charge_no_contract"] = (
            (df["MonthlyCharges"] > 70) & (df["Contract"] == "Month-to-month")
        ).astype(int)

    return df

# --- B. 載入與資料處理 ---
@st.cache_data
def get_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    # 載入後立刻做特徵工程
    df = add_features(df)  
    return df

df = get_data()

RANDOM_STATE = 42

# 固定驗證集（避免每次刷新結果不同）
@st.cache_data
def get_validation_split(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

@st.cache_data
def evaluate_models(df):
    X_train, X_valid, y_train, y_valid = get_validation_split(df)

    results = []

    for name, path in model_map.items():
        model = joblib.load(path)

        preds = model.predict(X_valid)
        probs = model.predict_proba(X_valid)[:, 1]

        acc = accuracy_score(y_valid, preds)
        f1 = f1_score(y_valid, preds)
        roc = roc_auc_score(y_valid, probs)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(roc, 4)
        })

    return pd.DataFrame(results).sort_values("F1 Score", ascending=False)

# 載入模型
@st.cache_resource # 快取模型避免重複載入
def load_model(name):
    return joblib.load(model_map[name])

# --- D. 頁面設計 ---
st.title(f"📊 Telco Customer Churn 預測")

# 使用 Tabs 分隔功能
tab1, tab2, tab3 = st.tabs([
    "📘 專案概覽",
    "🔮 預測分析",
    "🧠 模型解釋"
])

with tab1:
    st.subheader("📋 Dataset Preview")
    st.write("資料集前 5 筆樣本：")
    st.dataframe(df.head())

    st.subheader("📊 模型比較表")

    comparison_df = evaluate_models(df)
    st.dataframe(comparison_df, use_container_width=True)

    best_model_name = comparison_df.iloc[0]["Model"]
    st.success(f"🏆 目前最佳模型：{best_model_name}（依 F1 Score）")    

with tab2:
    st.subheader("🛠️ 模型選擇")

    model_choice = st.selectbox(
        "請選擇分類模型",
        ("Logistic Regression", "Random Forest", "XGBoost")
    )

    pipeline = load_model(model_choice)

    st.divider()

    if st.button("🎲 抽取樣本並預測"):
        # 1. 取得隨機樣本
        raw_sample = df.sample(1)
        
        # --- 新增：先顯示抽到的原始資料 ---
        st.markdown("📋 抽取的客戶原始資料")
        st.dataframe(raw_sample) # 顯示整列資料

        # --- 型別處理 ---
        raw_actual = raw_sample['Churn'].values[0]
        
        # 如果是字串 ("Yes"/"No")，轉成數字；如果是數字 (1/0)，直接用
        if isinstance(raw_actual, str):
            actual_val = 1 if raw_actual.lower() == 'yes' else 0
            actual_display = raw_actual
        else:
            actual_val = int(raw_actual)
            actual_display = "Yes (流失)" if actual_val == 1 else "No (留存)"
        
        # 3. 執行特徵工程 (確保 add_features 有處理 TotalCharges)
        sample_processed = add_features(raw_sample)
        feature_cols = pipeline.feature_names_in_
        X_sample = sample_processed[feature_cols]
        
        # 4. 進行預測
        prediction = int(pipeline.predict(X_sample)[0]) # 強制轉 int
        prob = pipeline.predict_proba(X_sample)[:, 1][0]
        
        # 5. 比對邏輯
        is_correct = (prediction == actual_val)
        
        st.divider()
        st.markdown("### 🔍 預測結果比對")
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="實際狀態", value=actual_display)
        with c2:
            pred_text = "Yes (流失)" if prediction == 1 else "No (留存)"
            # 用 delta 顯示是否正確
            # normal 是綠色，inverse 是紅色
            st.metric(
                label="模型預測", 
                value=pred_text, 
                delta="預測正確" if is_correct else "預測有誤",
                delta_color="normal" if is_correct else "inverse"
            )

            # 6. 顯示流失風險 & 等級
            risk_level = "⚠️ High" if prob > 0.7 else "⚠️ Medium" if prob > 0.3 else "💚 Low"
            st.write(f"模型預測流失機率：{prob:.2%}")
            st.write(f"風險等級：{risk_level}")

        st.divider()

        # 7. 關鍵特徵是依據 SHAP 重要度選擇
        key_features = {
            "tenure": "客戶年資(月)",
            "MonthlyCharges": "月費",
            "Contract": "合約類型"
        }

        st.markdown("### 📌 關鍵特徵概覽：")

        for col, label in key_features.items():
            value = raw_sample[col].values[0]
            st.write(f"**{label}**：{value}")

with tab3:
    st.info("目前圖表以 XGBoost 為示範模型。")
    col1, col2, col3 = st.columns([1, 3, 1])
    
    st.subheader("📊 模型效能")
    st.image(
        "images/XGB_pr_curve_churn.png",
        caption="XGBoost - Precision-Recall Curve",
        # use_container_width=True
    )

    st.divider()

    st.subheader("🔎 特徵影響分析")
    st.image(
        "images/XGB_beeswarm_churn.png",
        caption="XGBoost - SHAP Beeswarm",
        # use_container_width=True
    )

    st.divider()

    st.subheader("📌 Insights")
    st.markdown("""
        - SHAP 平均絕對值排名前三特徵為 tenure、MonthlyCharges、Contract。
        - 長期合約對預測流失機率具有負向影響。
        - Electronic Check 客戶群預測風險較高。
        - PR Curve 展示 Precision 與 Recall 之間的權衡關係。
    """)




