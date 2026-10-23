# Importing Stack Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve

import mlflow
mlflow.set_experiment("Zurich_Credit_Risk_MLOps_Pipeline")

# 1. Realistic Dataset Simulation (Corporate Firms in Zurich)
np.random.seed(42)
n_samples = 5000

data = {
    'liquidity_ratio': np.random.normal(1.5, 0.4, n_samples),
    'debt_to_equity': np.random.normal(0.8, 0.3, n_samples),
    'profit_margin': np.random.normal(0.12, 0.05, n_samples),
    'operational_cash_flow': np.random.normal(500000, 150000, n_samples),
    'years_in_business': np.random.randint(1, 25, n_samples),
    'credit_history_score': np.random.randint(500, 850, n_samples)
}
df = pd.DataFrame(data)

# Default logic based on financial risk rules
risk_formula = (
    (df['liquidity_ratio'] < 1.0).astype(int) * 2 +
    (df['debt_to_equity'] > 1.2).astype(int) * 2 +
    (df['profit_margin'] < 0.03).astype(int) * 1.5 +
    (df['credit_history_score'] < 600).astype(int) * 3
)
df['default'] = (risk_formula + np.random.normal(0, 0.8, n_samples) > 3.5).astype(int)

# 2. Exploratory Data Analysis (EDA) and Visualizations
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df[df['default'] == 0]['profit_margin'], color='green', label='Adimplente', kde=True, bins=30)
sns.histplot(df[df['default'] == 1]['profit_margin'], color='red', label='Inadimplente', kde=True, bins=30)
plt.title('Distribuição da Margem de Lucro por Status de Crédito')
plt.legend()

plt.subplot(1, 2, 2)
sns.boxplot(x='default', y='debt_to_equity', data=df, palette='Set2')
plt.title('Endividamento (Debt-to-Equity) vs Inadimplência')
plt.tight_layout()
plt.savefig('eda_credit_risk.png')
plt.show()

# 3. Data Preparation and Splitting
X = df.drop('default', axis=1)
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Treinamento de Modelos e MLOps com MLflow
models = {
    "Logistic_Regression": LogisticRegression(random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

best_auc = 0
best_model_name = None
best_model_obj = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        if name == "Logistic_Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, probs)

        # Logging metrics and parameters in MLflow
        mlflow.log_param("model_type", name)
        mlflow.log_metric("AUC_ROC", auc)
        
        print(f"--- Modelo: {name} ---")
        print(f"AUC-ROC: {auc:.4f}")
        print(classification_report(y_test, preds))
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model_obj = model

            print(f"\n[INFO] Best model selected for production: {best_model_name} with an AUC of {best_auc:.4f}")