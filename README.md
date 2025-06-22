# QSHAP Explainable AI

This project implements an interventional explainability framework using Quantum Bayesian Networks (QBNs) and Quantum SHAP (Q-SHAP) techniques for credit risk prediction.

The project includes:

- ✅ Quantum Bayesian Network backend (using Pennylane)
- ✅ Classical interventional inference backend
- ✅ Streamlit-based dashboard for interactive simulation
- ✅ Initial work toward Q-SHAP explainability module

## Dataset

The model uses the open-source "Default of Credit Card Clients" dataset.

Dataset file:  
`data/default_of_credit_card_clients.xlsx`

## Project Structure

| File                   | Purpose                                                      |
|------------------------|--------------------------------------------------------------|
| `qbn_dashboard.py`     | Main Streamlit app                                           |
| `interventional_V2.py` | Data preprocessing and classical interventional calculations |
| `qbn_backend.py`       | Quantum circuit backend                                      |
| `qshap_module.py`      | Quantum SHAP calculations (WIP)                              |
| `data/`                | Contains dataset file                                        |
| `requirements.txt`     | All required Python packages                                 |
| `LICENSE`              | License information                                          |


## Deployment

This project is designed for hosting on [Streamlit Community Cloud](https://streamlit.io/cloud).

After cloning:

```bash
pip install -r requirements.txt
streamlit run qbn_dashboard.py
