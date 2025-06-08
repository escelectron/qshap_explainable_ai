"""
Quantum Explainable AI Dashboard with Q-SHAP+
Author: Pranav Sanghadia
License: MIT
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from itertools import product

from interventional_V2 import load_data
from qbn_backend import batch_inference, feature_columns, run_inference
from qshap_module import compute_qshap_all, compute_multi_intervention, generate_profiles

# Streamlit page setup
st.set_page_config(page_title="Quantum Explainable AI Dashboard", layout="wide")
st.title("Quantum Bayesian Network Explainability Dashboard")

# Load dataset
df = load_data()

# Generate all possible binary combinations
profiles = generate_profiles()

# Sidebar for control panel
st.sidebar.header("Intervention Controls")

interventions = {}
for col in feature_columns:
    value = st.sidebar.radio(f"{col}", [None, 0, 1], index=0,
                              format_func=lambda x: "No Intervention" if x is None else f"{x}")
    if value is not None:
        interventions[col] = value

# Compute batch inference for current intervention state
results = batch_inference(profiles, interventions if interventions else None)
results_df = pd.DataFrame(results)

# Prepare labels for visualization
results_df['Profile'] = results_df[feature_columns].astype(str).agg('-'.join, axis=1)

# Plot results
st.subheader("Interventional Inference Results")
st.plotly_chart(
    px.bar(results_df, x="Profile", y="Delta", title="Δ P(Default=1) After Intervention"),
    use_container_width=True
)
st.dataframe(results_df)

# Divider for Q-SHAP section
st.markdown("---")
st.header("Q-SHAP+ Explainability")

# Compute full Q-SHAP+ attribution values
qshap_values = compute_qshap_all(__import__('qbn_backend'), profiles)
qshap_df = pd.DataFrame(list(qshap_values.items()), columns=['Feature', 'Q-SHAP+ Value'])
qshap_df = qshap_df.sort_values(by='Q-SHAP+ Value', ascending=False)

# Plot Q-SHAP values
st.plotly_chart(
    px.bar(qshap_df, x='Feature', y='Q-SHAP+ Value', title="Feature Attribution (Q-SHAP+)"),
    use_container_width=True
)
st.dataframe(qshap_df)

# Optional multi-intervention (future extension — experimental)
# st.header("Q-SHAP+ Multi-Intervention (Experimental)")
# phi_multi = compute_multi_intervention(__import__('qbn_backend'), profiles, ("LIMIT_BAL", "Age"))
# st.write(f"Multi-Intervention for LIMIT_BAL and Age: {phi_multi}")

