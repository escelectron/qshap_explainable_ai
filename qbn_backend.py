"""
Quantum Explainable AI : QBN Backend
Author: Pranav Sanghadia
License: MIT
"""

import joblib
import numpy as np
import os

# Load model
model_path = os.path.join("models", "xgb_credit_model.joblib")
model = joblib.load(model_path)

# Define feature columns
feature_columns = ["LIMIT_BAL", "Age", "PAY_AMT1", "EDUCATION", "MARRIAGE"]

def run_inference(profile, do_intervene=False, intervention_target=None):

    if isinstance(profile, list):
        input_data = {feat: val for feat, val in zip(feature_columns, profile)}
    else:
        input_data = profile.copy()

    if do_intervene and intervention_target:
        input_data.update(intervention_target)

    row = np.array([input_data[feat] for feat in feature_columns]).reshape(1, -1)
    prob = model.predict_proba(row)[0][1]
    return prob


def batch_inference(profiles, interventions=None):
    results = []
    for profile in profiles:
        prob = run_inference(profile, do_intervene=bool(interventions), intervention_target=interventions)
        # Build a dict from the profile and append probability
        result = {feat: val for feat, val in zip(feature_columns, profile)}
        result["Delta"] = prob
        results.append(result)
    return results

