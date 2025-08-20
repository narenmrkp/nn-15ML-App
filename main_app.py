# main_app.py
# ml_projects.py
import os
HF_API_KEY = os.getenv("HF_API_KEY")

# Example: using HF dataset loading
from datasets import load_dataset
# dataset = load_dataset("your_dataset_name", use_auth_token=HF_API_KEY)

import streamlit as st
import ml_projects as mp
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="15 ML Projects", layout="wide")
st.title("📁 ML Portfolio — 15 Projects (HF-friendly)")

# show HF token status
hf_token = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
if hf_token:
    st.sidebar.success("Hugging Face token found (will be used for HF dataset access).")
else:
    st.sidebar.warning("No Hugging Face token found in .env (public HF datasets may still work).")

PROJECTS = [
    ("01 • RF - Fraud", mp.proj_01_rf_fraud),
    ("02 • DT - Diabetes", mp.proj_02_dt_diabetes),
    ("03 • SVM - Cancer", mp.proj_03_svm_cancer),
    ("04 • LogReg - Heart", mp.proj_04_logreg_heart),
    ("05 • KNN - Iris", mp.proj_05_knn_iris),
    ("06 • NB - Spam", mp.proj_06_nb_spam),
    ("07 • LR - Student", mp.proj_07_lr_student),
    ("08 • RFR - House", mp.proj_08_rfr_house),
    ("09 • DTR - Salary", mp.proj_09_dtr_salary),
    ("10 • SVR - Car", mp.proj_10_svr_car),
    ("11 • KMeans - Seg", mp.proj_11_kmeans_seg),
    ("12 • Hier - GDP", mp.proj_12_hier_gdp),
    ("13 • DBSCAN - IoT", mp.proj_13_dbscan_iot),
    ("14 • PCA+LogReg", mp.proj_14_pca_logreg),
    ("15 • Ensemble Voting", mp.proj_15_ensemble),
]

# display 3 rows x 5 columns
cols_per_row = 5
st.markdown("Choose a project (small buttons):")
for i in range(0, len(PROJECTS), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, col in enumerate(cols):
        idx = i + j
        if idx < len(PROJECTS):
            label, func = PROJECTS[idx]
            # small button style via container width and key
            if col.button(label, key=f"btn_{idx}", use_container_width=True):
                st.session_state["selected"] = idx

st.markdown("---")
selected = st.session_state.get("selected", None)
if selected is None:
    st.info("Click any project button above to run the demo.")
else:
    label, func = PROJECTS[selected]
    st.subheader(f"▶️ {label}")
    try:
        func()
    except Exception as e:
        st.error(f"Error running project `{label}`:\n{e}")
