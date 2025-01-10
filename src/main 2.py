import streamlit as st
import ui_engine as ui
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


# Function to load and cache the datasets for improved performance
def load_and_cache_datasets():
    st.session_state.regression_data = load_data("./src/data/DS_salaries_regression.csv")    # Load dataset for Regression Models
    st.session_state.classifiers_data = load_data("./src/data/DS_salaries_regression_classification_ONLY_numerical.csv")    # Load dataset for Classifier Models


# Function to load the dataset from a file
def load_data(file_path):
    try:
        return pd.read_csv(file_path)    # Load the CSV file into a pandas DataFrame
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")    # Display error message if file is missing
        st.stop()    # Halt application execution


# Main Entry Point
def main():
    with st.spinner("...initializing session variables..."):
        init_session_variables()
    
    with st.spinner("...loading datasets..."):
        load_and_cache_datasets()
    
    ui.build_ui()



main()