import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import platform

# 偵測作業系統並設定字體
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 微軟正黑體
else:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac/Linux 常用
plt.rcParams['axes.unicode_minus'] = False # 解決負號變框框

# 設定頁面標題
st.set_page_config(page_title="Telco Customer Churn 預測儀表板", layout="wide")

# --- 0. 全域設定與更新檔名 ---
model_map = {
    "Logistic Regression": "lr_best.pkl",
    "Random Forest": "rf_best.pkl",
    "XGBoost": "xgb_best.pkl"
}

# 修正特徵清單 Bug：直接指定 feature_cols，不依賴 pipeline.feature_names_in_
FEATURE_COLS = [
    "SeniorCitizen", "Partner", "Dependents", "tenure",
    "num_core_services", "MonthlyCharges", "high_monthly_charge",
    "high_charge_short_tenure", "high_charge_no_contract",
    "PaperlessBilling", "PaymentMethod", "Contract"
]

# --- A. 特徵工程函式 (修復欄位未轉換 Bug) ---
def add_features(df):
    df = df.copy()

    # 0. 先處理目標變數
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # 修正 Bug：將訓練時設定為 passthrough 的 binary 欄位也從 Yes/No 轉為 0/1
    # 確保進入 Pipeline 時為數值型態，避免 ValueError
    binary_cols = ["Partner", "Dependents", "PaperlessBilling"]
    for col in binary_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].map({"No": 0, "Yes": 1})

    # 1. 處理 TotalCharges：原始資料含空字串，強制轉型並填補空值
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # 2. 建立新特徵 (加入型別檢查，避免重複執行出錯)
    if "MonthlyCharges" in df.columns:
        df["high_monthly_charge"] = (df["MonthlyCharges"] > 70).astype(int)

    # num_core_services：核心服務數量（PhoneService + InternetService，範圍 0–2）
    if "PhoneService" in df.columns and "InternetService" in df.columns:
        df["num_core_services"] = (
            (df["PhoneService"] == "Yes").astype(int)
            + (df["InternetService"] != "No").astype(int)
        )

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
    df = add_features(df)  
    return df

df = get_data()

# 載入模型
@st.cache_resource 
def load_model(name):
    return joblib.load(model_map[name])

# --- B2. 測試集機率快取（以模型名稱字串為 key，切換模型時正確失效）---
@st.cache_data(show_spinner=False)
def get_test_probs(model_name, X_te):
    pipe = load_model(model_name)
    return pipe.predict_proba(X_te)[:, 1]

# --- B3. LIME Explainer 快取（建立成本高，模型不換就不重建）---
@st.cache_resource
def get_lime_explainer(model_name, _X_train_trans, _feature_names):
    from lime.lime_tabular import LimeTabularExplainer
    return LimeTabularExplainer(
        training_data=_X_train_trans,
        feature_names=list(_feature_names),
        class_names=['Stay (留存)', 'Churn (流失)'],
        mode='classification',
        random_state=42
    )

# --- C. 初始化 Session State（確保 Slider 調整時抽樣資料不被重置）---
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None

# --- D. 頁面設計 ---
st.title("📊 Telco Customer Churn 預測儀表板")

# =================================================================
# 📥 左側控制面板 (Sidebar) - 讓模型與門檻變成全域控制
# =================================================================
with st.sidebar:
    st.header("⚙️ 模型與預測門檻設定")
    st.markdown("在這裡調整的參數會即時套用到右側的預測分析中。")
    st.divider()

    # 1. 模型選擇器移到左側
    model_choice = st.selectbox(
        "請選擇分類模型",
        ("Logistic Regression", "Random Forest", "XGBoost"),
        index=1  # 預設選中隨機森林 RF
    )

    # 2. 決策門檻 Slider 移到左側
    threshold = st.slider(
        "調整決策邊界門檻 (Threshold)", 
        min_value=0.10, 
        max_value=0.90, 
        value=0.44, 
        step=0.01,
        help="調低門檻會提高對流失客戶的敏感度（提升 Recall），但可能會降低 Precision。"
    )
    
    st.divider()
    st.caption("💡 提示：根據實驗結果，業務導向最優門檻為 **0.44**")

# 載入選定模型 (寫在 sidebar 外面，讓所有頁面都能共用這個 pipeline 變數)
pipeline = load_model(model_choice)

# --- 右側主畫面的 Tabs 分隔 ---
tab1, tab2, tab3 = st.tabs([
    "📘 專案概覽",
    "🔮 預測分析",
    "🧠 模型解釋"
])

# =================================================================
# Tab 1: 專案概覽
# =================================================================
with tab1:
    # --- 專案簡介 ---
    st.subheader("🗂️ 專案簡介")
    st.markdown("""
    電信產業的客戶獲取成本遠高於留客成本，能提前識別「即將流失」的客戶，
    就能讓業務團隊在對的時間介入，大幅提升留客效率。

    本專案以 **IBM Telco Customer Churn Dataset**（7,043 筆客戶資料）為基礎，
    針對資料集存在的 **類別不平衡問題**（流失客戶僅佔 26.5%），
    以「提升流失客戶的 Recall」為核心目標，歷經四個實驗階段系統性地優化模型，
    最終選定 **Random Forest（Optuna 調參版）** 搭配 **決策門檻 0.44**，
    在測試集上達到 **Churn Recall 0.807**、**F1-Score 0.629**。
    """)

    # 核心成果指標卡片
    st.markdown("#### 🏅 最終模型核心成果（RF + Threshold 0.44）")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Churn Recall",    "80.7%", help="成功識別出的流失客戶比例")
    m2.metric("Churn Precision", "51.5%", help="預測為流失中實際流失的比例")
    m3.metric("Churn F1-Score",  "0.629", help="Precision 與 Recall 的調和平均")
    m4.metric("Accuracy",        "75.0%", help="整體預測正確率")

    st.divider()

    # 四階段實驗方法論
    st.markdown("#### 🔬 四階段實驗方法論")
    st.markdown("""
    | 階段 | 方法 | 關鍵成果 |
    |------|------|---------|
    | **① Baseline** | 三模型預設參數 | RF 因過擬合 F1 最低（0.537），Recall 僅 0.492 |
    | **② 權重調整** | StratifiedKFold + class_weight / scale_pos_weight | LR Recall 大幅提升（0.47→0.80）；RF 受過擬合影響，改善有限 |
    | **③ 超參數搜尋** | GridSearchCV / RandomizedSearchCV | RF 關鍵突破：F1 從 0.529 躍升至 0.618，Recall 從 0.476 提升至 0.751 |
    | **④ 精細調優** | Optuna 貝葉斯優化 + Threshold Tuning | Threshold=0.50：F1=0.632、Recall=0.765；下調至 0.44 後：F1=0.629、Recall=**0.807** |
    """)

    st.divider()

    # 資料集概覽
    st.subheader("📋 數據分佈")
    st.caption("以下為資料集前 5 筆樣本預覽：")
    st.dataframe(df.head(5))

    st.subheader("📊 四個版本模型歷史比較表 (X_test, 預設 Threshold = 0.5)")
    
    # 建立四版本歷史對比 DataFrame
    history_data = {
        "模型": ["LR", "LR", "LR", "LR", "RF", "RF", "RF", "RF", "XGB", "XGB", "XGB", "XGB"],
        "版本": ["Baseline", "StratifiedKFold", "SearchCV", "Optuna"] * 3,
        "Precision": [0.6232, 0.4983, 0.5052, 0.5043, 0.5916, 0.5953, 0.5252, 0.5386, 0.6295, 0.5071, 0.5153, 0.5096],
        "Recall": [0.4733, 0.7995, 0.7807, 0.7834, 0.4920, 0.4759, 0.7513, 0.7647, 0.5134, 0.6711, 0.7647, 0.7781],
        "F1 Score": [0.5380, 0.6140, 0.6134, 0.6136, 0.5372, 0.5290, 0.6183, 0.6320, 0.5655, 0.5777, 0.6157, 0.6159]
    }
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True)

    st.success("🏆 最終結論：經過四階段實驗，選擇 **Random Forest (Optuna 版本)** 作為最終模型，其泛化能力最穩定。預設門檻（Threshold=0.50）：F1-Score = **0.632**、Recall = **76.5%**；業務導向門檻（Threshold=0.44）：F1-Score = **0.629**、Recall = **80.7%**，以極小的 F1 代價換取顯著的 Recall 提升。")

# =================================================================
# Tab 2: 預測分析 (視覺完美排版版：核心報告並排 + 深度診斷)
# =================================================================
with tab2:
    st.subheader("🔮 即時客戶流失狀況預測與深度診斷")
    st.markdown(f"當前主畫面正使用左側設定之模型：**{model_choice}** ｜ 決策門檻：**{threshold:.2f}**")

    # --- 1. 背景資料準備 (使用快取) ---
    @st.cache_data
    def prepare_backend_data():
        from sklearn.model_selection import train_test_split
        raw_df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        processed_df = add_features(raw_df)
        X_data = processed_df[FEATURE_COLS]
        y_data = processed_df['Churn']
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
        )
        return X_tr, X_te, y_tr, y_te

    X_train_bg, X_test_bg, y_train_bg, y_test_bg = prepare_backend_data()

    # --- 2. 隨機抽樣按鈕 ---
    if st.button("🎲 抽取新樣本"):
        st.session_state.selected_sample = df.sample(1)
        
    if st.session_state.selected_sample is not None:
        raw_sample = st.session_state.selected_sample
        
        st.markdown("### 📋 抽取的客戶原始資料")
        st.dataframe(raw_sample)

        # 實際標籤與標籤名稱處理
        raw_actual = raw_sample['Churn'].values[0]
        actual_val = int(raw_actual) if not isinstance(raw_actual, str) else (1 if raw_actual.lower() == 'yes' else 0)
        actual_display = "Yes (流失)" if actual_val == 1 else "No (留存)"
        
        # 特徵工程與對齊
        sample_processed = add_features(raw_sample)
        X_sample = sample_processed[FEATURE_COLS]
        
        # 預測機率與動態門檻判定
        prob = pipeline.predict_proba(X_sample)[:, 1][0]
        prediction = 1 if prob >= threshold else 0
        is_correct = (prediction == actual_val)
        
        st.divider()
        
        # --- ✨ 升級重點：用帶有外框的 Container 把結果與特徵打包平排 ---
        st.markdown("### 🔍 核心預測報告")
        with st.container(border=True):
            # 切分三欄：實際狀態 ｜ 模型預測 ｜ 關鍵特徵
            c1, c2, c3 = st.columns([1, 1.2, 1.2])
            
            with c1:
                st.metric(label="👤 實際客戶狀態", value=actual_display)
                
            with c2:
                pred_text = "Yes (流失)" if prediction == 1 else "No (留存)"
                st.metric(
                    label="🤖 模型動態預測", 
                    value=pred_text, 
                    delta="預測正確" if is_correct else "預測有誤",
                    delta_color="normal" if is_correct else "inverse"
                )
                risk_level = "🔴 High Risk" if prob > 0.7 else "🟡 Medium Risk" if prob > 0.44 else "🟢 Low Risk"
                st.markdown(f"流失機率：**{prob:.2%}** ｜ 風險：**{risk_level}**")
                
            with c3:
                st.markdown("📌 **關鍵特徵概覽（原始值）**")
                # 依據 SHAP 重要度選擇的核心業務特徵
                key_features = {
                    "tenure": "客戶年資 (月)", 
                    "MonthlyCharges": "當月費用", 
                    "Contract": "合約類型"
                }
                for col, label in key_features.items():
                    value = raw_sample[col].values[0]
                    st.write(f"• **{label}**：{value}")

        st.divider()
        
        # --- 4. 深度診斷圖表排版 (左邊放動態 PR 曲線，右邊放 LIME) ---
        col_graph1, col_graph2 = st.columns(2)
        
        with col_graph1:
            st.markdown("### 📈 動態測試集 PR 曲線")
            st.caption("紅點會隨著左側調整的 Threshold 即時移動")
            
            y_probs_test = get_test_probs(model_choice, X_test_bg)
            
            from sklearn.metrics import precision_recall_curve
            precisions, recalls, thresholds_curve = precision_recall_curve(y_test_bg, y_probs_test)
            
            closest_idx = np.argmin(np.abs(thresholds_curve - threshold))
            current_p = precisions[closest_idx]
            current_r = recalls[closest_idx]
            
            fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
            ax_pr.plot(recalls, precisions, color='dodgerblue', lw=2, label='PR 曲線')
            ax_pr.scatter(current_r, current_p, color='red', s=120, zorder=5, 
                       label=f'當前門檻 ({threshold:.2f})\nRecall: {current_r:.2f}\nPrecision: {current_p:.2f}')
            ax_pr.set_xlabel('Recall (召回率)')
            ax_pr.set_ylabel('Precision (精準率)')
            ax_pr.set_xlim([-0.05, 1.05])
            ax_pr.set_ylim([-0.05, 1.05])
            ax_pr.grid(True, linestyle='--', alpha=0.5)
            ax_pr.legend(loc='lower left')
            
            st.pyplot(fig_pr)

        with col_graph2:
            st.markdown("### 🧠 LIME 客戶個人化特徵診斷")
            st.caption("呈現影響「這位客戶」預測分數的關鍵特徵權重")
            
            with st.spinner("LIME 診斷計算中..."):
                import streamlit.components.v1 as components
                import scipy.sparse
                
                preprocessor = pipeline.named_steps['preprocess']
                model = pipeline.named_steps['model']
                
                X_train_trans = preprocessor.transform(X_train_bg)
                X_sample_trans = preprocessor.transform(X_sample)
                
                if scipy.sparse.issparse(X_train_trans):
                    X_train_trans = X_train_trans.toarray()
                if scipy.sparse.issparse(X_sample_trans):
                    X_sample_trans = X_sample_trans.toarray()
                    
                feature_names_trans = preprocessor.get_feature_names_out()

                # 使用快取的 explainer（模型不換就直接取用，不重建）
                explainer = get_lime_explainer(
                    model_choice, X_train_trans, feature_names_trans
                )
                
                exp = explainer.explain_instance(
                    data_row=X_sample_trans[0],
                    predict_fn=model.predict_proba,
                    num_features=6
                )
                
                # 注入白色背景 CSS，確保 LIME 圖表與 PR 曲線視覺一致
                lime_html = exp.as_html()
                white_bg_css = "<style>body,html{background:#ffffff!important;padding:6px}</style>"
                components.html(white_bg_css + lime_html, height=500, scrolling=True)
    else:
        st.info("請點擊上方按鈕抽取一位客戶進行分析預測！")

# =================================================================
# Tab 3: 模型解釋
# =================================================================
with tab3:
    st.info("💡 目前圖表以最終選定的隨機森林 (Random Forest) 模型作為展示。")
    
    # 三欄並排呈現，版面較為整齊
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 特徵重要度")
        st.image(
            "images/RF_feature_importance.png",
            caption="Random Forest - Feature Importance",
            use_container_width=True
        )
        
    with col2:
        st.subheader("🔎 SHAP 蜂群圖")
        st.image(
            "images/RF_shap_beeswarm.png",
            caption="Random Forest - SHAP Beeswarm",
            use_container_width=True
        )
        
    with col3:
        st.subheader("📈 PR 曲線")
        st.image(
            "images/RF_pr_curve.png",
            caption="Random Forest - Precision-Recall Curve",
            use_container_width=True
        )

    st.divider()

    st.subheader("📌 核心業務 Insights (依據 RF SHAP 評估)")
    st.markdown("""
    * **客戶年資 (tenure) 是留存關鍵：** 年資越短的客戶流失風險越高，應在入網前 12 個月內加強關懷。
    * **合約類型影響巨大：** 「Month-to-month（按月）」合約的客戶流失傾向最強；長期合約能有效降低流失風險。
    * **高月費的推波助瀾：** 當月費 (MonthlyCharges) 超過 70 美元且缺乏長期合約束縛時，客戶流失率會顯著飆高。
    * **動態門檻的價值：** 將決策門檻由 0.50 下調至 **0.44**，Recall 從 0.765 提升至 **0.807**，F1 僅損失 0.003；實際應用中可根據資源與風險承受度進行微調。
    """)