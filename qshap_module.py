
"""
Q-SHAP+ Module
Author: Pranav Sanghadia
License: MIT
"""

import numpy as np
from itertools import product

# Assumed feature columns
feature_columns = ["LIMIT_BAL", "Age", "PAY_AMT1", "EDUCATION", "MARRIAGE"]

def generate_profiles():
    return [list(p) for p in product([0, 1], repeat=len(feature_columns))]

def compute_qshap_single(run_inference_fn, profiles, feature_name):
    idx = feature_columns.index(feature_name)
    deltas = []

    for profile in profiles:
        profile_do1 = profile.copy()
        profile_do0 = profile.copy()
        profile_do1[idx] = 1
        profile_do0[idx] = 0

        p_do1 = run_inference_fn(profile_do1, do_intervene=True, intervention_target={feature_name: 1})
        p_do0 = run_inference_fn(profile_do0, do_intervene=True, intervention_target={feature_name: 0})

        delta = p_do1 - p_do0
        deltas.append(delta)

    return round(np.mean(deltas), 4)

def compute_qshap_all(run_inference_fn, profiles):
    shap_results = {}
    for feature in feature_columns:
        shap_results[feature] = compute_qshap_single(run_inference_fn, profiles, feature)
    return shap_results
