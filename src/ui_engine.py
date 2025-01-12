import streamlit as st
import pandas as pd

from data import data_mappings as dm

from analysis import dataset_analysis as data_analysis

from models import gaussian_naive_bayes
from models import knn
from models import gradient_booster_regressor
from models import random_forest



def build_ui():
    with st.spinner("loading in progress, please wait..."):
        tab1,tab2=st.tabs(["Salary Prediction","Dataset Analysis"])
        with tab2:
            with st.spinner("performing data analysis, please wait..."):
                data_analysis.perform()
        with tab1:
            build_form()



def build_form():
    # Create the streamlit application interface
    st.title("Find Out Your Predicted Salary")
    st.write("Answer the following questions to see your predicted salary range:")
    
    # Get unique values for dropdowns
    unique_jobs = sorted(st.session_state.regression_data['job_title'].unique())
    unique_locations = sorted(st.session_state.regression_data['company_location'].unique())
    unique_employee_residences = sorted(st.session_state.regression_data['employee_residence'].unique())
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)        
        with col1:
            st.session_state.experience_level = st.selectbox(
                label = "Experience Level:",
                options = [""] + dm.EXPERIENCE_LEVEL_MAPPING,
                index = 0
            )
            
            st.session_state.employee_residence = st.selectbox(
                "Employee Residence:",
                [""] + unique_employee_residences,
                index = 0
            )
            
            st.session_state.company_size = st.selectbox(
                label = "Company Size:",
                options = [""] + dm.COMPANY_SIZE_MAPPING,
                index = 0
            )
        
        with col2:
            st.session_state.employment_type = st.selectbox(
                label = "Employment Type:",
                options = [""] + dm.EMPLOYMENT_TYPE_MAPPING,
                index = 0
            )
            
            st.session_state.job_title = st.selectbox(
                "Job Title:",
                [""] + unique_jobs,
                index = 0
            )
            
            st.session_state.company_location = st.selectbox(
                "Company Location:",
                [""] + unique_locations,
                index = 0
            )
        
        st.session_state.remote_ratio = st.slider(
            "Remote Work Percentage:",
            min_value = 0,
            max_value = 100,
            value = st.session_state.remote_ratio,
            step = 50
        )
        col1,col2 = st.columns(2)
        with col2:
            hypertuning_toggled = st.toggle(
                "Enable Regressor Model Hypertuning",
                value=st.session_state.hypertuning_enabled,
                key="hypertuning_toggled"
            )
        with col1:
            # Process prediction when requested
            predict_salary_click = st.form_submit_button("Predict Salary", use_container_width=True)
    
    if predict_salary_click or st.session_state.predict_button_clicked:
        #st.session_state.predict_button_clicked=True
        st.session_state.hypertuning_enabled=hypertuning_toggled
        # Validate that all fields are filled
        if not all([st.session_state.experience_level, st.session_state.employment_type, st.session_state.job_title, st.session_state.employee_residence, st.session_state.company_location, st.session_state.company_size]):
            st.error("Please fill in all fields before prediction.")
        else:
            predict_salary()



def predict_salary():
    # Create input dataframe for prediction
    form_data = {
        "experience_level": [dm.EXPERIENCE_LEVEL_MAPPER[st.session_state.experience_level]],
        "employment_type": [dm.EMPLOYMENT_TYPE_MAPPER[st.session_state.employment_type]],
        "job_title": [st.session_state.job_title],
        "employee_residence": [st.session_state.employee_residence],
        "company_location": [st.session_state.company_location],
        "company_size": [dm.COMPANY_SIZE_MAPPER[st.session_state.company_size]],
        "remote_ratio": [st.session_state.remote_ratio]
    }
    
    if st.session_state.hypertuning_enabled:
        classifier_tab,regression_tab,regression_with_hypertuning_tab=st.tabs(["Classifier Models", "Regression Models", "Regression Models with Hypertuning"])
    else:
        classifier_tab,regression_tab=st.tabs(["Classifier Models", "Regression Models"])
    
    with st.spinner("...still making predictions, please wait..."):
        with classifier_tab.container(border=True):
            x_input = [
                form_data["experience_level"][0],
                form_data["employment_type"][0],
                form_data["job_title"][0],
                form_data["employee_residence"][0],
                str(form_data["remote_ratio"][0]),
                form_data["company_location"][0],
                form_data["company_size"][0]
            ]
            
            clas_tab1,clas_tab2=st.tabs(["Gaussian Naive Bayes", "KNN"])
            with clas_tab1.container(border=True):
                with st.spinner("training Gaussian Naive Bayes AI model, please wait..."):
                    model_pipeline,stats=gaussian_naive_bayes.init()
                with st.spinner("making predictions, please wait..."):
                    gaussian_naive_bayes.predict(model_pipeline,x_input)
                st.divider()
                gaussian_naive_bayes.display_stats(stats)
            with clas_tab2.container(border=True):
                with st.spinner("training KNN AI model, please wait..."):
                    model_pipeline,stats=knn.init()
                with st.spinner("making predictions, please wait..."):
                    knn.predict(model_pipeline,x_input)
                st.divider()
                knn.display_stats(stats)
        
        with regression_tab.container(border=True):
            df=pd.DataFrame(form_data)
            reg_tab1,reg_tab2=st.tabs(["Random Forest", "Gradient Booster Regressor"])
            with reg_tab1.container(border=True):
                with st.spinner("training Random Forest AI model, please wait..."):
                    model_pipeline,preprocessor,stats=random_forest.init()
                with st.spinner("making predictions, please wait..."):
                    random_forest.predict(model_pipeline,df)
                st.divider()
                random_forest.display_stats(model_pipeline,preprocessor,stats)
            with reg_tab2.container(border=True):
                with st.spinner("training Gradient Booster Regressor AI model, please wait..."):
                    model_pipeline,preprocessor,stats=gradient_booster_regressor.init()
                with st.spinner("making predictions, please wait..."):
                    gradient_booster_regressor.predict(model_pipeline,df)
                st.divider()
                gradient_booster_regressor.display_stats(model_pipeline,preprocessor,stats)

        if st.session_state.hypertuning_enabled:
            with regression_with_hypertuning_tab.container(border=True):
                df=pd.DataFrame(form_data)
                reg_tab1,reg_tab2=st.tabs(["Random Forest", "Gradient Booster Regressor"])
                with reg_tab1.container(border=True):
                    with st.spinner("training Random Forest AI model, please wait..."):
                        model_pipeline,preprocessor,stats=random_forest.init(True)
                    st.divider()
                    with st.spinner("making predictions, please wait..."):
                        random_forest.predict(model_pipeline,df)
                    st.divider()
                    random_forest.display_stats(model_pipeline,preprocessor,stats)
                with reg_tab2.container(border=True):
                    with st.spinner("training Gradient Booster Regressor AI model, please wait..."):
                        model_pipeline,preprocessor,stats=gradient_booster_regressor.init(True)
                    st.divider()
                    with st.spinner("making predictions, please wait..."):
                        gradient_booster_regressor.predict(model_pipeline,df)
                    st.divider()
                    gradient_booster_regressor.display_stats(model_pipeline,preprocessor,stats)



