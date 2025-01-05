import streamlit as st
import pandas as pd
import utilities as utils

from analysis import dataset_analysis as data_analysis

from models import gaussian_naive_bayes
from models import knn
from models import gradient_booster_regressor
from models import random_forest



def build_ui():
    tab1,tab2=st.tabs(["Salary Prediction","Dataset Analysis"])
    with tab1:
        build_form()
    with tab2:
        data_analysis.perform()



def build_form():
    # Create the streamlit application interface
    st.title("Find Out Your Predicted Salary")
    st.write("Answer the following questions to see your predicted salary range:")
    
    # Get unique values for dropdowns
    unique_jobs = sorted(st.session_state.regression_data['job_title'].unique())
    unique_locations = sorted(st.session_state.regression_data['company_location'].unique())
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)        
        with col1:
            st.session_state.experience_level = st.selectbox(
                label = "Experience Level:",
                options = [""] + utils.EXPERIENCE_LEVEL_MAPPING,
                index = 0
            )
            
            st.session_state.employment_type = st.selectbox(
                label = "Employment Type:",
                options = [""] + utils.EMPLOYMENT_TYPE_MAPPING,
                index = 0
            )
            
            st.session_state.job_title = st.selectbox(
                "Job Title:",
                [""] + unique_jobs,
                index = 0
            )
        
        with col2:
            st.session_state.company_location = st.selectbox(
                "Company Location:",
                [""] + unique_locations,
                index = 0
            )
            
            st.session_state.company_size = st.selectbox(
                label = "Company Size:",
                options = [""] + utils.COMPANY_SIZE_MAPPING,
                index = 0
            )
            
            st.session_state.remote_ratio = st.slider(
                "Remote Work Percentage:",
                min_value = 0,
                max_value = 100,
                value = st.session_state.remote_ratio,
                step = 10
            )
        
        # Process prediction when requested
        predict_salary_click = st.form_submit_button("Predict Salary")
    
    if predict_salary_click:
        # Validate that all fields are filled
        if not all([st.session_state.experience_level, st.session_state.employment_type, st.session_state.job_title, st.session_state.company_location, st.session_state.company_size]):
            st.error("Please fill in all fields before prediction.")
        else:
            predict_salary()



def predict_salary():
    classifier_tab,regression_tab=st.tabs(["Classifier Models", "Regression Models"])
    with classifier_tab.container(border=True):
        clas_tab1,clas_tab2=st.tabs(["Gaussian Naive Bayes", "KNN"])
        with clas_tab1.container(border=True):
            gaussian_naive_bayes.perform()
        with clas_tab2.container(border=True):
            model_pipeline=knn.perform()
    
    with regression_tab.container(border=True):
        # Create input dataframe for prediction
        input_data = pd.DataFrame({
            "experience_level": [st.session_state.experience_level],
            "employment_type": [st.session_state.employment_type],
            "job_title": [st.session_state.job_title],
            "company_location": [st.session_state.company_location],
            "company_size": [st.session_state.company_size],
            "remote_ratio": [st.session_state.remote_ratio]
        })
        
        reg_tab1,reg_tab2=st.tabs(["Gradient Booster Regressor", "Random Forest"])
        with reg_tab1.container(border=True):
            cv_mae_mean,cv_mae_std,cv_mae_scores,preprocessor,model_pipeline,X,train_mae,test_mae,X_train,X_test,features,categorical_features,numerical_features=gradient_booster_regressor.init()
            gradient_booster_regressor.predict(model_pipeline,input_data)
            st.divider()
            gradient_booster_regressor.display_stats(cv_mae_mean,cv_mae_std,cv_mae_scores,preprocessor,model_pipeline,X,train_mae,test_mae,X_train,X_test,features,categorical_features,numerical_features)
        with reg_tab2.container(border=True):
            preprocessor,model_pipeline,mae,total_samples,train_samples,test_samples,unique_job_titles,unique_locations,features,categorical_features,numerical_features=random_forest.init()
            random_forest.predict(model_pipeline,input_data)
            st.divider()
            random_forest.display_results(preprocessor,model_pipeline,mae,total_samples,train_samples,test_samples,unique_job_titles,unique_locations,features,categorical_features,numerical_features)


