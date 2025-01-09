# Import required libraries for data manipulation, machine learning and web interface
import streamlit as st      # Web application framework for user interface
import pandas as pd        # Data manipulation and analysis library
import numpy as np
import random
import time

from sklearn.compose import ColumnTransformer    # For combining preprocessing steps
from sklearn.ensemble import RandomForestRegressor    # Machine learning model for prediction
from sklearn.preprocessing import OneHotEncoder    # For encoding categorical variables
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, cross_val_score   # For splitting dataset into training and testing sets
from sklearn.pipeline import Pipeline    # For creating sequential data processing steps
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score    # For calculating model accuracy

from data import data_mappings as dm

# Model Parameter Tuning
N_ESTIMATORS = 550 # 50
RANDOM_STATE = 42
MAX_DEPTH = 35 # 30
MAX_FEATURES = 15
MIN_SAMPLE_SPLIT = 2 # 10
MIN_SAMPLE_LEAF = 1


def discover_optimal_params(pipeline, param_grid, X_train, y_train):
    st.write("**Hyperparameter Tuning Progress**")
    chart = st.line_chart()
    progress_bar = st.progress(0)
    param_combinations = list(ParameterGrid(param_grid))
    mae_scores = []
    iterations = []
    best_score = float('inf')
    best_params = None
    st.divider()
    
    st.write("**Details**")
    with st.empty():
        for i, params in enumerate(param_combinations):
            # Update progress bar
            progress_bar.progress((i + 1) / len(param_combinations))

            # Update pipeline with current params
            pipeline.set_params(**params)
            
            # Perform cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            mean_cv_score = -np.mean(cv_scores)
            
            mae_scores.append(mean_cv_score)
            iterations.append(i + 1)
            
            # Update best parameters if current score is better
            if mean_cv_score < best_score:
                best_score = mean_cv_score
                best_params = params
            
            # Display progress metrics
            test_results={"Tested Params": params, "MAE": mean_cv_score, "Best MAE": best_score, "Best Params": best_params}
            st.write(test_results)
            
            # Update real-time chart
            chart.line_chart(pd.DataFrame({'Iteration': iterations, 'MAE': mae_scores}).set_index('Iteration'))

            # Add a delay for better visualization
            time.sleep(0.1)
        st.write("")

    # Set the best parameters on the pipeline
    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)
    
    # st.success("Tuning Complete")
    # st.write("Best Parameters:", best_params)
    # st.write(f"Best MAE: {best_score}")
    
    return pipeline


def get_a_hypertuned_model(pipeline, X_train, y_train):
    # Hyperparameter tuning
    with st.spinner("...hyperparameter tuning..."):
        param_grid = {
            'random_forest_model__n_estimators': [50, 100, 150, 300, 500, 550],
            'random_forest_model__max_depth': [20, 25, 30, 35],
            'random_forest_model__min_samples_split': [2, 10],
            'random_forest_model__max_features': [15, 20, 'sqrt', 'log2', None]
        }
        with st.container(border=True):
            pipeline = discover_optimal_params(pipeline, param_grid, X_train, y_train)
            best_model = pipeline

        with st.container(border=True):
            grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=2)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
    
    return best_model


def init(use_hypertuning=False):
    random.seed(RANDOM_STATE)
    # Define prediction features and target variable
    features = dm.REGRESSION_FEATURES    # List of features for prediction
    # Separate features for preprocessing
    categorical_features = features[0:-1]    # Features needing encoding
    numerical_features = [features[-1]]    # Features to remain numeric

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
            #min_samples_leaf=MIN_SAMPLE_LEAF,
            ))    # Apply data preprocessing
    ])

    # Split dataset and train the model
    X_train, X_test, y_train, y_test = train_test_split(st.session_state.regression_data[dm.REGRESSION_FEATURES], st.session_state.regression_data['salary_in_usd'], test_size=0.2, random_state=RANDOM_STATE)    # Create training and testing sets
    if use_hypertuning:
        model_pipeline = get_a_hypertuned_model(model_pipeline,X_train,y_train)    # Train a hypertuned model
    else:
        model_pipeline.fit(X_train, y_train)    # Train the model

    # Calculate model performance metrics
    model_perf_metrics = {}
    y_pred = model_pipeline.predict(X_test)
    model_perf_metrics['mae'] = mean_absolute_error(y_test, y_pred)    # Calculate prediction accuracy
    model_perf_metrics['mse'] = mean_squared_error(y_test, y_pred)
    model_perf_metrics['r2'] = r2_score(y_test, y_pred)
    
    # Cross-validation 
    cv_metrics={}
    cv_metrics['cv_score'] = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_metrics['cv_mae_scores'] = -cv_metrics['cv_score']
    cv_metrics['cv_mae_mean'] = cv_metrics['cv_mae_scores'].mean()
    cv_metrics['cv_mae_std'] = cv_metrics['cv_mae_scores'].std()
    
    # Calculate dataset statistics
    total_samples = len(st.session_state.regression_data[dm.REGRESSION_FEATURES])    # Total number of records
    train_samples = len(X_train)    # Number of training samples
    test_samples = len(X_test)    # Number of testing samples
    unique_employee_residences = len(st.session_state.regression_data['employee_residence'].unique())    # Count of unique employee residences
    unique_job_titles = len(st.session_state.regression_data['job_title'].unique())    # Count of unique job titles
    unique_locations = len(st.session_state.regression_data['company_location'].unique())    # Count of unique locations
    
    return preprocessor,model_pipeline,model_perf_metrics,cv_metrics,total_samples,train_samples,test_samples,unique_employee_residences,unique_job_titles,unique_locations,features,categorical_features,numerical_features


def predict(model_pipeline,input_data):
    st.subheader("Prediction Outcome")
    try:
        # Generate salary prediction
        prediction = model_pipeline.predict(input_data)[0]

        # Calculate salary range with ±15% variation
        salary_range = (prediction * 0.85, prediction * 1.15)

        # Display prediction results
        st.success(f"Predicted Salary Range (±15%): \${float(salary_range[0]):0,.2f} - \${float(salary_range[1]):0,.2f}")
        st.info(f"Base Prediction: ${float(prediction):0,.2f}")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")    # Handle and display any errors


def display_results(preprocessor,model_pipeline,model_perf_metrics,cv_metrics,total_samples,train_samples,test_samples,unique_employee_residences,unique_job_titles,unique_locations,features,categorical_features,numerical_features):
    # Display model evaluation in tabular format
    st.subheader("Model Evaluation")
    model_info = {
        'Parameter': [
            'Model Type',
            'Number of Trees',
            'Random State',
            'Maximum Depth',
            'Maximum Features',
            'Minimum Samples Split',
            'Minimum Samples Leaf',
            'Mean Absolute Error (MAE)',
            'Mean Squared Error (MSE)',
            'R² Score'
        ],
        'Value': [
            'Random Forest Regressor',
            N_ESTIMATORS,
            RANDOM_STATE,
            MAX_DEPTH,
            MAX_FEATURES,
            MIN_SAMPLE_SPLIT,
            MIN_SAMPLE_LEAF,
            model_perf_metrics['mae'],
            model_perf_metrics['mse'],
            model_perf_metrics['r2']
        ]
    }
    model_info_df = pd.DataFrame(pd.DataFrame(model_info))
    html_table = model_info_df.to_html(index=False)
    st.markdown(html_table,unsafe_allow_html=True)    # Present model parameters
    
    st.write(f"In the context of our task, the calculated MAE was approximately ${model_perf_metrics['mae']:,.2f}, meaning the model's salary predictions are off by about that amount on average.")
    
    
    st.subheader("Cross-Validation Results")
    st.write(f"Cross-Validation MAE: \${cv_metrics['cv_mae_mean']:,.2f} ± \${cv_metrics['cv_mae_std']:,.2f}")
    cv_results_df = pd.DataFrame({
        'Fold': range(1, len(cv_metrics['cv_mae_scores'])+1),
        'MAE': cv_metrics['cv_mae_scores']
    })
    st.markdown(cv_results_df.style.format({'MAE': '${:,.2f}'}).to_html(index=True),unsafe_allow_html=True)
    
    
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
    st.subheader("Top 15 Most Important Features")
    feature_names = (
        preprocessor.named_transformers_['cat']
        .get_feature_names_out(categorical_features)
        .tolist() + numerical_features
    )    
    
    importances = model_pipeline.named_steps['random_forest_model'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(15)
    st.markdown(feature_importance_df.style.format({'Importance': '{:.4f}'}).to_html(index=False),unsafe_allow_html=True)
    st.bar_chart(feature_importance_df.set_index('Feature'))

