"""
Q-SHAP+ Module
Author: Pranav Sanghadia
License: MIT
"""

import numpy as np
from itertools import product

# We assume the same feature_columns as QBN Backend
feature_columns = ["LIMIT_BAL", "Age", "PAY_AMT1", "EDUCATION", "MARRIAGE"]

def generate_profiles():
    """
    Generates all binary combinations of feature profiles.
    """
    return [list(p) for p in product([0, 1], repeat=len(feature_columns))]


def compute_qshap_single(model, profiles, feature_name):
    """
    Compute Q-SHAP+ for a single feature intervention.
    """
    idx = feature_columns.index(feature_name)
    deltas = []

    for profile in profiles:
        profile_do1 = profile.copy()
        profile_do0 = profile.copy()
        profile_do1[idx] = 1
        profile_do0[idx] = 0

        p_do1 = model.run_inference(profile_do1, do_intervene=False)
        p_do0 = model.run_inference(profile_do0, do_intervene=False)

        delta = p_do1 - p_do0
        deltas.append(delta)

    phi_i = round(np.mean(deltas), 4)
    return phi_i


def compute_qshap_all(model, profiles):
    """
    Compute Q-SHAP+ values for all features.
    """
    shap_results = {}
    for feature in feature_columns:
        shap_results[feature] = compute_qshap_single(model, profiles, feature)
    return shap_results


def compute_multi_intervention(model, profiles, feature_pair):
    """
    Compute multi-intervention Q-SHAP+ for a pair of features.
    """
    i, j = feature_columns.index(feature_pair[0]), feature_columns.index(feature_pair[1])
    deltas = []

    for profile in profiles:
        p11 = profile.copy()
        p00 = profile.copy()
        p11[i], p11[j] = 1, 1
        p00[i], p00[j] = 0, 0

        prob11 = model.run_inference(p11, do_intervene=False)
        prob00 = model.run_inference(p00, do_intervene=False)

        delta = prob11 - prob00
        deltas.append(delta)

    phi_ij = round(np.mean(deltas), 4)
    return phi_ij
