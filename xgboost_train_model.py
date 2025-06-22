import pandas as pd
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_excel("default_of_credit_card_clients.xlsx", engine="openpyxl", header=1)
df = df.rename(columns={"default payment next month": "Default"})

# Feature binarization
df["LIMIT_BAL"] = pd.qcut(df["LIMIT_BAL"], 2, labels=[0, 1])
df["Age"] = pd.qcut(df["AGE"], 2, labels=[0, 1])
df["PAY_AMT1"] = pd.qcut(df["PAY_AMT1"], 2, labels=[0, 1])
df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 1 if x >= 2 else 0)
df["MARRIAGE"] = df["MARRIAGE"].apply(lambda x: 1 if x == 1 else 0)
df["Default"] = df["Default"].astype(int)

# Select features and target
features = ["LIMIT_BAL", "Age", "PAY_AMT1", "EDUCATION", "MARRIAGE"]
X = df[features].copy()
y = df["Default"]

# Ensure all features are int
X = X.astype(int)

# Train XGBoost model
model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
model.fit(X, y)

# Save model to disk
joblib.dump(model, "xgb_credit_model.joblib")
print("Model saved as xgb_credit_model.joblib")
