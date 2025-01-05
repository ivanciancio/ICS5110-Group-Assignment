import streamlit as st
import pandas as pd



# Define prediction features and target variable
FEATURES = ['experience_level', 'employment_type', 'job_title', 'company_location', 'company_size', 'remote_ratio']    # List of features for prediction

# Define mappings for user interface values to dataset codes
EXPERIENCE_LEVEL_MAPPING = [
    "Entry Level",    # For newcomers to the field
    "Mid Level",      # For those with some experience
    "Senior Level",   # For experienced professionals
    "Executive"       # For leadership positions
]

EMPLOYMENT_TYPE_MAPPING = [
    "Full-Time",      # Standard full-time employment
    "Part-Time",      # Part-time positions
    "Contract",       # Fixed-term contract work
    "Freelance"       # Independent contractor work
]

COMPANY_SIZE_MAPPING = [
    "Small",            # Small enterprises
    "Medium",           # Mid-sized organisations
    "Large"             # Large corporations
]




def init_session_variables():
    if "regression_data" not in st.session_state:
        st.session_state.regression_data=None
    
    if "classifiers_data" not in st.session_state:
        st.session_state.classifiers_data=None
    
    if "experience_level" not in st.session_state:
        st.session_state.experience_level="Entry Level"
    
    if "employment_type" not in st.session_state:
        st.session_state.employment_type="Full-Time"
    
    if "job_title" not in st.session_state:
        st.session_state.job_title=""
    
    if "company_location" not in st.session_state:
        st.session_state.company_location=""
    
    if "company_size" not in st.session_state:
        st.session_state.company_size="Small"
    
    if "remote_ratio" not in st.session_state:
        st.session_state.remote_ratio=50


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

