import streamlit as st
import ui_engine as ui
from analysis import dataset_analysis
import pandas as pd


# Function to initiate all session variables
def init_session_variables():
    if "regression_data" not in st.session_state:
        st.session_state.regression_data=None
    
    if "classifiers_data" not in st.session_state:
        st.session_state.classifiers_data=None
    
    if "experience_level" not in st.session_state:
        st.session_state.experience_level=""
    
    if "employment_type" not in st.session_state:
        st.session_state.employment_type=""
    
    if "employee_residence" not in st.session_state:
        st.session_state.employee_residence=""
    
    if "job_title" not in st.session_state:
        st.session_state.job_title=""
    
    if "employee_residence" not in st.session_state:
        st.session_state.employee_residence=""
    
    if "company_location" not in st.session_state:
        st.session_state.company_location=""
    
    if "company_size" not in st.session_state:
        st.session_state.company_size=""
    
    if "remote_ratio" not in st.session_state:
        st.session_state.remote_ratio=0
    
    if "hypertuning_enabled" not in st.session_state:
        st.session_state.hypertuning_enabled=False
    
    if "predict_button_clicked" not in st.session_state:
        st.session_state.predict_button_clicked=False



# Main Entry Point
def main():
    with st.spinner("...initializing session variables..."):
        init_session_variables()
    
    ui.build_ui()



main()