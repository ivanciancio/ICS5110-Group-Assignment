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
        
        # Process prediction when requested
        predict_salary_click = st.form_submit_button("Predict Salary")
    
    if predict_salary_click:
        # Validate that all fields are filled
        if not all([st.session_state.experience_level, st.session_state.employment_type, st.session_state.job_title, st.session_state.company_location, st.session_state.company_size]):
            st.error("Please fill in all fields before prediction.")
        else:
            with st.spinner("making predictions, please wait..."):
                predict_salary()



def predict_salary():
    # Create input dataframe for prediction
    form_data = {
        "experience_level": [st.session_state.experience_level],
        "employment_type": [st.session_state.employment_type],
        "job_title": [st.session_state.job_title],
        "employee_residence": [st.session_state.employee_residence],
        "company_location": [st.session_state.company_location],
        "company_size": [st.session_state.company_size],
        "remote_ratio": [st.session_state.remote_ratio]
    }
    
    classifier_tab,regression_tab=st.tabs(["Classifier Models", "Regression Models"])
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
                X,X_train,X_val,y_val,X_test,acc,y_pred,model_pipeline=gaussian_naive_bayes.init()
            with st.spinner("making predictions, please wait..."):
                gaussian_naive_bayes.predict(model_pipeline,x_input)
            st.divider()
            gaussian_naive_bayes.display_results(X,X_train,X_val,y_val,X_test,acc,y_pred)
        with clas_tab2.container(border=True):
            with st.spinner("training KNN AI model, please wait..."):
                X,X_train,X_val,y_val,X_test,acc,num_neighbours,performance,y_pred,model_pipeline=knn.init()
            with st.spinner("making predictions, please wait..."):
                knn.predict(model_pipeline,x_input)
            st.divider()
            knn.display_results(X,X_train,X_val,y_val,X_test,acc,num_neighbours,performance,y_pred)
    
    with regression_tab.container(border=True):
        df=pd.DataFrame(form_data)
        reg_tab1,reg_tab2,reg_tab3,reg_tab4=st.tabs(["Gradient Booster Regressor", "Gradient Booster Regressor with Hypertuning", "Random Forest", "Random Forest with Hypertuning"])
        with reg_tab1.container(border=True):
            with st.spinner("training Gradient Booster Regressor AI model, please wait..."):
                cv_mae_mean,cv_mae_std,cv_mae_scores,preprocessor,model_pipeline,X,train_mae,test_mae,X_train,X_test,features,categorical_features,numerical_features=gradient_booster_regressor.init()
            with st.spinner("making predictions, please wait..."):
                gradient_booster_regressor.predict(model_pipeline,df)
            st.divider()
            gradient_booster_regressor.display_results(cv_mae_mean,cv_mae_std,cv_mae_scores,preprocessor,model_pipeline,X,train_mae,test_mae,X_train,X_test,features,categorical_features,numerical_features)
        with reg_tab3.container(border=True):
            with st.spinner("training Random Forest AI model, please wait..."):
                preprocessor,model_pipeline,model_perf_metrics,cv_metrics,total_samples,train_samples,test_samples,unique_employee_residences,unique_job_titles,unique_locations,features,categorical_features,numerical_features=random_forest.init()
            with st.spinner("making predictions, please wait..."):
                random_forest.predict(model_pipeline,df)
            st.divider()
            random_forest.display_results(preprocessor,model_pipeline,model_perf_metrics,cv_metrics,total_samples,train_samples,test_samples,unique_employee_residences,unique_job_titles,unique_locations,features,categorical_features,numerical_features)
        with reg_tab2.container(border=True):
            with st.spinner("training Gradient Booster Regressor AI model, please wait..."):
                cv_mae_mean,cv_mae_std,cv_mae_scores,preprocessor,model_pipeline,X,train_mae,test_mae,X_train,X_test,features,categorical_features,numerical_features=gradient_booster_regressor.init(True)
            with st.spinner("making predictions, please wait..."):
                gradient_booster_regressor.predict(model_pipeline,df)
            st.divider()
            gradient_booster_regressor.display_results(cv_mae_mean,cv_mae_std,cv_mae_scores,preprocessor,model_pipeline,X,train_mae,test_mae,X_train,X_test,features,categorical_features,numerical_features)
        with reg_tab4.container(border=True):
            with st.spinner("training Random Forest AI model, please wait..."):
                preprocessor,model_pipeline,model_perf_metrics,cv_metrics,total_samples,train_samples,test_samples,unique_employee_residences,unique_job_titles,unique_locations,features,categorical_features,numerical_features=random_forest.init(True)
            with st.spinner("making predictions, please wait..."):
                random_forest.predict(model_pipeline,df)
            st.divider()
            random_forest.display_results(preprocessor,model_pipeline,model_perf_metrics,cv_metrics,total_samples,train_samples,test_samples,unique_employee_residences,unique_job_titles,unique_locations,features,categorical_features,numerical_features)







