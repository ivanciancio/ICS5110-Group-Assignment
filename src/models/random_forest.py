import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from joblib import Parallel, delayed
import multiprocessing

from sklearn.compose import ColumnTransformer    # For combining preprocessing steps
from sklearn.ensemble import RandomForestRegressor    # Machine learning model for prediction
from sklearn.preprocessing import StandardScaler, OneHotEncoder    # For encoding categorical variables
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, cross_val_score, KFold   # For splitting dataset into training and testing sets
from sklearn.pipeline import Pipeline    # For creating sequential data processing steps
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import data_mappings as dm

# Model Parameter Tuning
N_ESTIMATORS = 50
RANDOM_STATE = 42
MAX_DEPTH = 14
MAX_FEATURES = 15
MIN_SAMPLE_SPLIT = 15
MIN_SAMPLE_LEAF = 1

def evaluate_parameters(params, pipeline, X_train, y_train):
    """Evaluate a single parameter combination with log-transformed target"""
    pipeline.set_params(**params)
    cv_scores = cross_val_score(
        pipeline, 
        X_train, 
        y_train, 
        cv=5, 
        scoring='neg_mean_absolute_error',
        n_jobs=1  # Using 1 job here since we're already parallelizing the parameter search
    )
    return -np.mean(cv_scores), params

def discover_optimal_params(pipeline, param_grid, X_train, y_train):
    st.write("**Hyperparameter Tuning Progress**")
    
    param_combinations = list(ParameterGrid(param_grid))
    n_combinations = len(param_combinations)
    
    # Calculate number of CPU cores to use (leave one core free for system)
    n_cores = max(1, multiprocessing.cpu_count() - 1)
    st.info(f"Using {n_cores} CPU cores for parallel processing")
    
    chart = st.line_chart()
    progress_bar = st.progress(0)
    st.divider()
    placeholder = st.empty()
    
    # Initialize tracking variables
    mae_scores = []
    iterations = []
    best_score = float('inf')
    best_params = None
    
    # Create batches of parameter combinations for progress updates
    batch_size = max(1, n_combinations // 350)  # Update progress roughly 350 times
    
    try:
        # Initialize parallel processing
        with Parallel(n_jobs=n_cores, backend='loky') as parallel:
            # Process parameter combinations in batches
            for batch_start in range(0, n_combinations, batch_size):
                batch_end = min(batch_start + batch_size, n_combinations)
                batch_combinations = param_combinations[batch_start:batch_end]
                
                # Evaluate batch of parameter combinations in parallel
                batch_results = parallel(
                    delayed(evaluate_parameters)(params, pipeline, X_train, y_train)
                    for params in batch_combinations
                )
                
                # Process batch results
                for mean_cv_score, params in batch_results:
                    iterations.append(len(mae_scores) + 1)
                    mae_scores.append(mean_cv_score)
                    
                    if mean_cv_score < best_score:
                        best_score = mean_cv_score
                        best_params = params
                
                # Update progress and display
                progress = (batch_end) / n_combinations
                progress_bar.progress(progress)
                
                # Update display with latest results
                with placeholder:
                    test_results = {"Tested Params": params, "Best Params": best_params}
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Current MAE: :blue[${np.expm1(mean_cv_score):,.2f}]")
                        st.write("Latest Tested Params:")
                        st.write(pd.DataFrame(test_results)['Tested Params'])
                    with col2:
                        st.write(f"Best MAE: :green[${np.expm1(best_score):,.2f}]")
                        st.write("Best Params:")
                        st.write(pd.DataFrame(test_results)['Best Params'])
                
                # Update chart
                chart.line_chart(pd.DataFrame({
                    'Iteration': iterations,
                    'MAE': np.expm1(mae_scores)  # Transform back for visualization
                }).set_index('Iteration'))
                
                # Small delay for UI updates
                time.sleep(0.1)

            with placeholder:
                st.write("")
    
    except Exception as e:
        st.error(f"An error occurred during parallel processing: {str(e)}")
        raise e
    
    finally:
        st.success("Tuning Complete")
        st.write(f"Best MAE: :green[${np.expm1(best_score):,.2f}]")
        st.write("Best Parameters:", pd.DataFrame(best_params.items(), columns=['Model Parameter', 'Optimal Value']))
        
        # Set the best parameters and fit on full training data
        pipeline.set_params(**best_params)
        
        global N_ESTIMATORS, MAX_DEPTH, MAX_FEATURES, MIN_SAMPLE_SPLIT, MIN_SAMPLE_LEAF
        N_ESTIMATORS = best_params['rf_m__n_estimators']
        MAX_DEPTH = best_params['rf_m__max_depth']
        MAX_FEATURES = best_params['rf_m__max_features']
        MIN_SAMPLE_SPLIT = best_params['rf_m__min_samples_split']
        MIN_SAMPLE_LEAF = best_params['rf_m__min_samples_leaf']
    
    return pipeline

def get_a_hypertuned_model(pipeline, X_train, y_train):
    # Hyperparameter tuning
    with st.spinner("...hyperparameter tuning..."):
        param_grid = {
            'rf_m__n_estimators': [50, 100, 150, 300],
            'rf_m__max_depth': [14, 20, 25, 30, 35],
            'rf_m__min_samples_split': [2, 10, 15],
            'rf_m__max_features': [7, 15, 20],
            'rf_m__min_samples_leaf': [1, 4, 5, 10]
        }
        with st.container(border=True):
            pipeline = discover_optimal_params(pipeline, param_grid, X_train, y_train)
            best_model = pipeline
    
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
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ])
    
    # Create complete machine learning pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),    # Apply data preprocessing
        ('rf_m', RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            max_depth=MAX_DEPTH,
            max_features=MAX_FEATURES,
            min_samples_split=MIN_SAMPLE_SPLIT,
            min_samples_leaf=MIN_SAMPLE_LEAF,
            n_jobs=1  # Set to 1 since we're handling parallelization manually
            ))
    ])

    # Log transform the target variable
    y = np.log1p(st.session_state.regression_data['salary_in_usd'])

    # Split dataset with log-transformed target
    X_train, X_test, y_train, y_test = train_test_split(
        st.session_state.regression_data[dm.REGRESSION_FEATURES], 
        y,  # Using log-transformed target
        test_size=0.2, 
        random_state=RANDOM_STATE
    )
    
    if use_hypertuning:
        model_pipeline = get_a_hypertuned_model(model_pipeline, X_train, y_train)

    model_pipeline.fit(X_train, y_train)    # Train the model

    # Calculate metrics (transform predictions basck to original scale)
    y_pred = np.expm1(model_pipeline.predict(X_test))
    y_test_original = np.expm1(y_test)
    
    # Calculate model performance metrics
    model_perf_metrics = {}
    model_perf_metrics['mae'] = mean_absolute_error(y_test_original, y_pred)
    model_perf_metrics['mse'] = mean_squared_error(y_test_original, y_pred)
    model_perf_metrics['r2'] = r2_score(y_test_original, y_pred)
    
    # Cross-validation with log-transformed target
    cv_metrics = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_metrics['cv_score'] = cross_val_score(
        model_pipeline, 
        X_train, 
        y_train,
        cv=cv, 
        scoring='neg_mean_absolute_error',
        n_jobs=1
    )
    cv_metrics['cv_mae_scores'] = np.expm1(-cv_metrics['cv_score'])  # Transform back for interpretability
    cv_metrics['cv_mae_mean'] = cv_metrics['cv_mae_scores'].mean()
    cv_metrics['cv_mae_std'] = cv_metrics['cv_mae_scores'].std()
    
    # Calculate dataset statistics
    total_samples = len(st.session_state.regression_data[dm.REGRESSION_FEATURES])    # Total number of records
    train_samples = len(X_train)    # Number of training samples
    test_samples = len(X_test)    # Number of testing samples
    unique_employee_residences = len(st.session_state.regression_data['employee_residence'].unique())    # Count of unique employee residences
    unique_job_titles = len(st.session_state.regression_data['job_title'].unique())    # Count of unique job titles
    unique_locations = len(st.session_state.regression_data['company_location'].unique())    # Count of unique locations
    
    return model_pipeline, preprocessor, {
        'model_perf_metrics': model_perf_metrics,
        'cv_metrics': cv_metrics,
        'total_samples': total_samples,
        'train_samples': train_samples,
        'test_samples': test_samples,
        'unique_employee_residences': unique_employee_residences,
        'unique_job_titles': unique_job_titles,
        'unique_locations': unique_locations,
        'features': features,
        'categorical_features': categorical_features,
        'numerical_features': numerical_features
    }

def predict(model_pipeline, input_data):
    st.subheader("Prediction Outcome")
    # Generate salary prediction
    try:
        # Make prediction on log scale
        log_prediction = model_pipeline.predict(input_data)[0]
        # Transform back to original scale
        prediction = np.expm1(log_prediction)
        
        #prediction = abs(model_pipeline.predict(input_data)[0])
        #prediction = model_pipeline.predict(input_data)[0]
        
        # Calculate salary range with ±15% variation
        lower_bound = prediction * 0.85
        upper_bound = prediction * 1.15

        st.success(f"Predicted Salary Range (±15%): \${lower_bound:,.2f} - \${upper_bound:,.2f}")
        st.info(f"Base Prediction: ${prediction:,.2f}")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")    # Handle and display any errors

def display_stats(model_pipeline, preprocessor, stats):
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
            'R² Score',
            'Target Transformation'
        ],
        'Value': [
            'Random Forest Regressor',
            N_ESTIMATORS,
            RANDOM_STATE,
            MAX_DEPTH,
            MAX_FEATURES,
            MIN_SAMPLE_SPLIT,
            MIN_SAMPLE_LEAF,
            stats['model_perf_metrics']['mae'],
            stats['model_perf_metrics']['mse'],
            stats['model_perf_metrics']['r2'],
            'Natural Log (log1p)'
        ]
    }
    model_info_df = pd.DataFrame(pd.DataFrame(model_info))
    html_table = model_info_df.to_html(index=False)
    st.markdown(html_table, unsafe_allow_html=True)
    
    st.write(f"In the context of our task, the calculated MAE was approximately ${stats['model_perf_metrics']['mae']:,.2f}, meaning the model's salary predictions are off by about that amount on average.")
    
    st.subheader("Cross-Validation Results")
    st.write(f"Cross-Validation MAE: \${stats['cv_metrics']['cv_mae_mean']:,.2f} ± \${stats['cv_metrics']['cv_mae_std']:,.2f}")
    cv_results_df = pd.DataFrame({
        'Fold': range(1, len(stats['cv_metrics']['cv_mae_scores'])+1),
        'MAE': stats['cv_metrics']['cv_mae_scores']
    })
    st.markdown(cv_results_df.style.format({'MAE': '${:,.2f}'}).to_html(index=True), unsafe_allow_html=True)
    
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
            'Testing Data Percentage',
            'Target Transformation'
        ],
        'Value': [
            f'{stats['total_samples']:,}',
            f'{stats['train_samples']:,}',
            f'{stats['test_samples']:,}',
            f'{stats['unique_employee_residences']}',
            f'{stats['unique_job_titles']:,}',
            f'{stats['unique_locations']:,}',
            f'{(stats['train_samples']/stats['total_samples'])*100}%',
            f'{(stats['test_samples']/stats['total_samples'])*100:.0f}%',
            'Natural Log (log1p)'
        ]
    }
    
    stats_df = pd.DataFrame(pd.DataFrame(training_info))
    html_table = stats_df.to_html(index=False)
    st.markdown(html_table, unsafe_allow_html=True)
    
    # Displaying features used for prediction
    st.subheader("Features Used for Prediction")
    features_df = pd.DataFrame({'Features': stats['features']})
    st.markdown(features_df.to_html(index=False), unsafe_allow_html=True)
    
    # Displaying feature importance
    st.subheader("Top 10 Most Important Features")
    feature_names = (
        preprocessor.named_transformers_['cat']
        .get_feature_names_out(stats['categorical_features'])
        .tolist() + stats['numerical_features']
    )    
    
    importances = model_pipeline.named_steps['rf_m'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)
    st.markdown(feature_importance_df.style.format({'Importance': '{:.4f}'}).to_html(index=False), unsafe_allow_html=True)
    st.bar_chart(feature_importance_df.set_index('Feature'))


