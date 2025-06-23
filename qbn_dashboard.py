
"""
Quantum Explainable AI Dashboard with Q-SHAP+
Author: Pranav Sanghadia
License: MIT
"""
import shap
import streamlit as st
import pandas as pd
import plotly.express as px
from itertools import product

from interventional_V2 import load_data
from qbn_backend import batch_inference, feature_columns, run_inference
from qshap_module import compute_qshap_all, generate_profiles, compute_classical_shap_values

# Streamlit page setup
st.set_page_config(page_title="Quantum Explainable AI Dashboard", layout="wide")
st.title("Quantum Bayesian Network Explainability Dashboard")

# Load dataset
df = load_data()

# Sidebar for backend selection
st.sidebar.header("Select Mode")
mode = st.sidebar.radio("Select Task", ["Quantum Inference", "Q-SHAP+ Explainability"], index=0, key="task_selector")

# Sidebar for intervention controls (only for inference)
interventions = {}
if mode == "Quantum Inference":
    st.sidebar.header("Intervention Controls")
    for col in feature_columns:
        value = st.sidebar.radio(f"{col}", [None, 0, 1], index=0,
                                  format_func=lambda x: "No Intervention" if x is None else f"{x}")
        if value is not None:
            interventions[col] = value

# Generate all binary profiles
profiles = generate_profiles()

# Main logic based on mode
if mode == "Quantum Inference":
    results = batch_inference(profiles, interventions if interventions else None)
    results_df = pd.DataFrame(results)
    results_df['Profile'] = results_df[feature_columns].astype(str).agg('-'.join, axis=1)

    st.subheader("Interventional Inference Results")
    st.plotly_chart(
        px.bar(results_df, x="Profile", y="Delta", title="Δ P(Default=1) After Intervention"),
        use_container_width=True
    )
    st.dataframe(results_df)

elif mode == "Q-SHAP+ Explainability":
    st.sidebar.header("Explainability Options")
    explain_method = st.sidebar.radio("Choose Explainability Method", ["Q-SHAP+", "Classical SHAP"], key="explain_method_selector")

    if explain_method == "Q-SHAP+":
        qshap_values = compute_qshap_all(run_inference, profiles)
        qshap_df = pd.DataFrame(list(qshap_values.items()), columns=['Feature', 'Q-SHAP+ Value'])
        qshap_df = qshap_df.sort_values(by='Q-SHAP+ Value', ascending=False)

        st.subheader("Q-SHAP+ Attribution Results")
        st.plotly_chart(
            px.bar(qshap_df, x='Feature', y='Q-SHAP+ Value', title="Feature Attribution (Q-SHAP+)"),
            use_container_width=True
        )
        st.dataframe(qshap_df)

    elif explain_method == "Classical SHAP":
        import joblib
        import shap
        from xgboost import Booster

        # Load XGBoost model and profiles
        model = joblib.load("models/xgb_credit_model.joblib")
        profiles_df = pd.DataFrame(profiles, columns=feature_columns)

        # Use TreeExplainer instead of general SHAP Explainer
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(profiles_df)

        # Aggregate SHAP values
        mean_shap = shap_vals.mean(axis=0)
        shap_df = pd.DataFrame({
            'Feature': feature_columns,
            'SHAP Value': mean_shap
        }).sort_values(by='SHAP Value', ascending=False)

        st.subheader("Classical SHAP Attribution Results")
        st.plotly_chart(
            px.bar(shap_df, x='Feature', y='SHAP Value', title="Feature Attribution (Classical SHAP)"),
            use_container_width=True
        )
        st.dataframe(shap_df)



