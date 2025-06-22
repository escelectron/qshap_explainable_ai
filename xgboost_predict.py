import joblib
import pandas as pd

# Load model
model = joblib.load("models/xgb_credit_model.joblib")

# Predict on dummy input:
sample = pd.DataFrame([[1, 0, 1, 1, 0]], columns=["LIMIT_BAL", "Age", "PAY_AMT1", "EDUCATION", "MARRIAGE"])
pred = model.predict_proba(sample)
print(f"P(Default=1) = {pred[0][1]:.4f}")