import pandas as pd
import joblib

# Load trained model
model = joblib.load("xgb_credit_model.joblib")

# Define feature columns in the same order as training
features = ["LIMIT_BAL", "Age", "PAY_AMT1", "EDUCATION", "MARRIAGE"]

# Sample binary input (can be any [0, 1] combination)
original_profile = [1, 0, 1, 1, 0]  # Example: high limit, young, high repay, high edu, single

# Optional: define intervention (dict of feature: value)
intervention = {
    "PAY_AMT1": 0,
    "EDUCATION": 0
}

# Create DataFrame from original profile
X_obs = pd.DataFrame([original_profile], columns=features)
p_obs = model.predict_proba(X_obs)[0][1]

# Apply intervention
intervened_profile = original_profile.copy()
for key, val in intervention.items():
    index = features.index(key)
    intervened_profile[index] = val

X_do = pd.DataFrame([intervened_profile], columns=features)
p_do = model.predict_proba(X_do)[0][1]

# Output results
print(f"Original Profile:       {dict(zip(features, original_profile))}")
print(f"Intervened Profile:     {dict(zip(features, intervened_profile))}")
print(f"P(Default=1) observed:  {round(p_obs, 4)}")
print(f"P(Default=1) do():      {round(p_do, 4)}")
print(f"Delta:                  {round(p_do - p_obs, 4)}")
