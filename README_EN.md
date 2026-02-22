# 📞 Telco Customer Churn Prediction & Explainable Analysis
> **Identifying high-risk churn customers using machine learning and providing interpretable insights through SHAP.**

---

## 📌 Project Overview

### 🔹 Background:
- Customer churn increases **Customer Acquisition Cost (CAC)** and reduces long-term **Customer Lifetime Value (CLV)**. Without an effective mechanism to identify high-risk customers, companies must continuously invest in acquiring new customers, which negatively impacts revenue stability.

### 🔹 Objective:
- Analyzing key factors influencing customer churn through Exploratory Data Analysis (EDA) and building predictive models to identify potential churn risks.

### 🔹 Methods:
- Compared **Logistic Regression, Random Forest, and XGBoost**
- Used **F1-score and Precision-Recall Curve** as primary evaluation metrics
- Applied **SHAP** for model interpretability analysis

### 🔹 Business Value:
- Enable proactive retention strategies by identifying high-risk customers in advance, supporting data-driven marketing and customer engagement decisions.

---

## 📊 Dataset
### 🔹 Source:
- Kaggle Telco Customer Churn Dataset
  
### 🔹 Size:
- 7,043 customer records with 21 features
  
### 🔹 Target Variable:
- `Churn` (Yes / No)
  
### 🔹 Class Distribution:
| Class | Count | Ratio |
|-------|-------|-------|
| Retained | 5,174 | 73.5% |
| Churned | 1,869 | 26.5% |

---

## ⚙️ Modeling Approach

### 1️⃣ Data Preprocessing:
- Converted binary categorical features (Yes/No) into 0/1 encoding
- Applied One-Hot Encoding for multi-class categorical variables

### 2️⃣ Feature Engineering:
- Created a high monthly charge indicator
- Constructed interaction features:
  - MonthlyCharges × Tenure
  - MonthlyCharges × Contract type
- Aggregated multiple service-related features (e.g., OnlineSecurity, TechSupport, StreamingTV) into `num_services` to:
  - Reduce dimensionality
  - Improve generalization
  - Mitigate potential multicollinearity

### 3️⃣ Handling Class Imbalance:
- No resampling (e.g., SMOTE) was applied 
- Selected evaluation metrics suitable for imbalanced data (F1-score and PR Curve)

### 4️⃣ Evaluation Metrics
- **Recall**: Ability to correctly identify actual churn customers
- **Precision**: Proportion of correctly predicted churn customers
- **F1-score**: Harmonic mean of Precision and Recall
- Since the primary goal is to detect potential churn customers while avoiding excessive false positives, **F1-score** was chosen as the main evaluation metric.

---

## 📈 Model Performance & Interpretation

### 🔹 Model Comparison
![Model Comparison](images/model_comparison.png)

### 🔹 Precision-Recall Curve
- Used to evaluate model discrimination ability under class imbalance.
![PR Curve](images/XGB_pr_curve_churn.png)

### 🔹 SHAP Feature Importance (Global Explanation)
- Displays the overall impact and direction of features on model predictions.
![SHAP Beeswarm](images/XGB_beeswarm_churn.png)

### 🔹 Key Insights
- **Tenure** and **MonthlyCharges** are the most influential factors
- Long-term contracts significantly reduce churn risk
- Customers using electronic check payment show higher churn probability

---

## 🚀 Streamlit Interactive Demo

> This project included an interactive web application built with Streamlit to demonstrate model performance and prediction capabilities

### 🔹 Tab 1: Project Overview 
- Dataset preview (first 5 records)
- Model performance comparison
- Best model identification

### 🔹 Tab 2: Prediction Analysis
- Model selection interface
- Random customer sampling
- Actual label vs predicted result
- Predicted churn probability and risk level
- Top 3 SHAP contributing features (local explanation)

### 🔹 Tab 3: Model Interpretation
- Precision-Recall Curve
- SHAP global feature importance
- Summary of key insights

---

## 🛠 Tech Stack

### Data Processing
- pandas
- numpy

### Modeling
- scikit-learn (Logistic Regression, Random Forest)
- XGBoost

### Model Interpretation
- SHAP

### Visualization
- matplotlib
- seaborn

### Model Persistence
- joblib

### Deployment
- Streamlit

---

🔗 **中文版**: [README.md](README.md)
