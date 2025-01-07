# Import required libraries for data manipulation, machine learning and web interface
import streamlit as st      # Web application framework for user interface
import pandas as pd        # Data manipulation and analysis library
from sklearn.compose import ColumnTransformer    # For combining preprocessing steps
from sklearn.ensemble import RandomForestRegressor    # Machine learning model for prediction
from sklearn.preprocessing import OneHotEncoder    # For encoding categorical variables
from sklearn.model_selection import train_test_split    # For splitting dataset into training and testing sets
from sklearn.pipeline import Pipeline    # For creating sequential data processing steps
from sklearn.metrics import mean_absolute_error    # For calculating model accuracy

from data import data_mappings as dm

# Model Parameter Tuning
N_ESTIMATORS = 550
RANDOM_STATE = 42
MAX_DEPTH = 35
MAX_FEATURES = 15
MIN_SAMPLE_SPLIT = 2
MIN_SAMPLE_LEAF = 1


def init():
    # Define prediction features and target variable
    features = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size', 'remote_ratio']    # List of features for prediction
    # Separate features for preprocessing
    categorical_features = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']    # Features needing encoding
    numerical_features = ['remote_ratio']    # Features to remain numeric

    # Create data preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),    # Transform categorical variables
            ('num', 'passthrough', numerical_features)    # Keep numerical variables unchanged
        ])
    
    # Create complete machine learning pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),    # Apply data preprocessing
        ('random_forest_model', RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            max_depth=MAX_DEPTH,
            max_features=MAX_FEATURES,
            min_samples_split=MIN_SAMPLE_SPLIT,
            min_samples_leaf=MIN_SAMPLE_LEAF,
            ))    # Apply data preprocessing
    ])

    # Split dataset and train the model
    X_train, X_test, y_train, y_test = train_test_split(st.session_state.regression_data[dm.REGRESSION_FEATURES], st.session_state.regression_data['salary_in_usd'], test_size=0.2, random_state=RANDOM_STATE)    # Create training and testing sets
    model_pipeline.fit(X_train, y_train)    # Train the model

    # Calculate model performance metrics
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)    # Calculate prediction accuracy

    # Calculate dataset statistics
    total_samples = len(st.session_state.regression_data[dm.REGRESSION_FEATURES])    # Total number of records
    train_samples = len(X_train)    # Number of training samples
    test_samples = len(X_test)    # Number of testing samples
    unique_employee_residences = len(st.session_state.regression_data['employee_residence'].unique())    # Count of unique employee residences
    unique_job_titles = len(st.session_state.regression_data['job_title'].unique())    # Count of unique job titles
    unique_locations = len(st.session_state.regression_data['company_location'].unique())    # Count of unique locations
    
    return preprocessor,model_pipeline,mae,total_samples,train_samples,test_samples,unique_employee_residences,unique_job_titles,unique_locations,features,categorical_features,numerical_features


def predict(model_pipeline,input_data):
    st.subheader("Prediction Outcome")
    try:
        # Generate salary prediction
        prediction = model_pipeline.predict(input_data)[0]

        # Calculate salary range with ±10% variation
        salary_range = (prediction * 0.9, prediction * 1.1)

        # Display prediction results
        st.success(f"Predicted Salary Range (±10%): \${float(salary_range[0]):0,.2f} - \${float(salary_range[1]):0,.2f}")
        st.info(f"Base Prediction: ${float(prediction):0,.2f}")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")    # Handle and display any errors


def display_results(preprocessor,model_pipeline,mae,total_samples,train_samples,test_samples,unique_employee_residences,unique_job_titles,unique_locations,features,categorical_features,numerical_features):
    # Display model information in tabular format
    st.subheader("Model Information")
    model_info = {
        'Parameter': [
            'Model Type',
            'Number of Trees',
            'Random State',
            'Maximum Depth',
            'Maximum Features',
            'Minimum Samples Split',
            'Minimum Samples Leaf',
            'Model Performance (MAE)'
        ],
        'Value': [
            'Random Forest Regressor',
            N_ESTIMATORS,
            RANDOM_STATE,
            MAX_DEPTH,
            MAX_FEATURES,
            MIN_SAMPLE_SPLIT,
            MIN_SAMPLE_LEAF,
            f'${mae:,.2f}'
        ]
    }
    model_info_df = pd.DataFrame(pd.DataFrame(model_info))
    html_table = model_info_df.to_html(index=False)
    st.markdown(html_table,unsafe_allow_html=True)    # Present model parameters
    
    st.write(f"In the context of our task, the calculated MAE was approximately ${mae:,.2f}, meaning the model's salary predictions are off by about that amount on average.")
    
    # Display dataset information in tabular format
    st.subheader("Testing and Training Information")
    training_info = {
        'Metric': [
            'Total Dataset Size',
            'Training Set Size',
            'Testing Set Size',
            'Unique Employee Residences',
            'Unique Job Titles',
            'Unique Company Locations',
            'Training Data Percentage',
            'Testing Data Percentage'
        ],
        'Value': [
            f'{total_samples:,}',
            f'{train_samples:,}',
            f'{test_samples:,}',
            f'{unique_employee_residences}',
            f'{unique_job_titles:,}',
            f'{unique_locations:,}',
            f'{(train_samples/total_samples)*100:.1f}%',
            f'{(test_samples/total_samples)*100:.1f}%'
        ]
    }
    
    stats_df = pd.DataFrame(pd.DataFrame(training_info))
    html_table = stats_df.to_html(index=False)
    st.markdown(html_table,unsafe_allow_html=True)    # Present dataset statistics
    
    # Display features used for prediction
    st.subheader("Features Used for Prediction")
    features_df = pd.DataFrame({'Features': features})
    st.markdown(features_df.to_html(index=False),unsafe_allow_html=True)    # Present feature list
    
    
    # Display feature importance
    st.subheader("Top 10 Most Important Features")
    feature_names = (
        preprocessor.named_transformers_['cat']
        .get_feature_names_out(categorical_features)
        .tolist() + numerical_features
    )    
    
    importances = model_pipeline.named_steps['random_forest_model'].feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)

    st.markdown(feature_importance.style.format({'Importance': '{:.4f}'}).to_html(index=False),unsafe_allow_html=True)