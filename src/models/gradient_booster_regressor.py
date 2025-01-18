# Gradient boosting regression model implementation for salary prediction using log transformation
import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from joblib import Parallel, delayed
import multiprocessing

from data import data_mappings as dm

# Initial model hyperparameters
LEARNING_RATE = 0.05
N_ESTIMATORS = 500
RANDOM_STATE = 42
MAX_DEPTH = 6
MIN_SAMPLE_SPLIT = 5
MIN_SAMPLE_LEAF = 4
SUBSAMPLE = 0.8

def evaluate_parameters(params, pipeline, X_train, y_train):
    # Evaluates parameter combinations using cross-validation with log-transformed target
    pipeline.set_params(**params)
    cv_scores = cross_val_score(
        pipeline, 
        X_train, 
        y_train, 
        cv=5, 
        scoring='neg_mean_absolute_error',
        n_jobs=1  # Single job due to parallel processing implementation
    )
    return -np.mean(cv_scores), params

def discover_optimal_params(pipeline, param_grid, X_train, y_train):
    # Main hyperparameter optimisation function using parallel processing
    st.write("**Hyperparameter Tuning Progress**")
    
    param_combinations = list(ParameterGrid(param_grid))
    n_combinations = len(param_combinations)
    
    # Configures CPU usage for parallel processing
    n_cores = max(1, multiprocessing.cpu_count() - 1)
    st.info(f"Using {n_cores} CPU cores for parallel processing")
    
    # Progress tracking components
    chart = st.line_chart()
    progress_bar = st.progress(0)
    st.divider()
    placeholder = st.empty()
    
    # Optimisation tracking variables
    mae_scores = []
    iterations = []
    best_score = float('inf')
    best_params = None
    
    # Defines batch size for progress updates
    batch_size = max(1, n_combinations // 350)
    
    try:
        # Implements parallel parameter evaluation
        with Parallel(n_jobs=n_cores, backend='loky') as parallel:
            for batch_start in range(0, n_combinations, batch_size):
                batch_end = min(batch_start + batch_size, n_combinations)
                batch_combinations = param_combinations[batch_start:batch_end]
                
                # Processes parameter combinations in parallel
                batch_results = parallel(
                    delayed(evaluate_parameters)(params, pipeline, X_train, y_train)
                    for params in batch_combinations
                )
                
                # Updates optimisation tracking
                for mean_cv_score, params in batch_results:
                    iterations.append(len(mae_scores) + 1)
                    mae_scores.append(mean_cv_score)
                    
                    if mean_cv_score < best_score:
                        best_score = mean_cv_score
                        best_params = params
                
                # Updates progress visualisations
                progress = (batch_end) / n_combinations
                progress_bar.progress(progress)
                
                with placeholder:
                    test_results = {"Tested Params": params, "Best Params": best_params}
                    col1, col2 = st.columns(2)
                    with col1:
                        # Displays current results with inverse log transformation
                        st.write(f"Current MAE: :blue[${np.expm1(mean_cv_score):,.2f}]")
                        st.write("Latest Tested Params:")
                        st.write(pd.DataFrame(test_results)['Tested Params'])
                    with col2:
                        st.write(f"Best MAE: :green[${np.expm1(best_score):,.2f}]")
                        st.write("Best Params:")
                        st.write(pd.DataFrame(test_results)['Best Params'])
                
                # Updates progress chart with transformed MAE scores
                chart.line_chart(pd.DataFrame({
                    'Iteration': iterations,
                    'MAE': np.expm1(mae_scores)
                }).set_index('Iteration'))
                
                time.sleep(0.1)  # UI update delay

            with placeholder:
                st.write("")
    
    except Exception as e:
        st.error(f"An error occurred during parallel processing: {str(e)}")
        raise e
    
    finally:
        # Finalises optimisation results
        st.success("Tuning Complete")
        st.write(f"Best MAE: :green[${np.expm1(best_score):,.2f}]")
        st.write("Best Parameters:", pd.DataFrame(best_params.items(), columns=['Model Parameter', 'Optimal Value']))
        
        # Updates model with optimal parameters
        pipeline.set_params(**best_params)
        
        # Updates global hyperparameters
        global LEARNING_RATE, N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLE_SPLIT, MIN_SAMPLE_LEAF, SUBSAMPLE
        LEARNING_RATE = best_params['gb_m__learning_rate']
        N_ESTIMATORS = best_params['gb_m__n_estimators']
        MAX_DEPTH = best_params['gb_m__max_depth']
        MIN_SAMPLE_SPLIT = best_params['gb_m__min_samples_split']
        MIN_SAMPLE_LEAF = best_params['gb_m__min_samples_leaf']
        SUBSAMPLE = best_params['gb_m__subsample']
    
    return pipeline

def get_a_hypertuned_model(pipeline, X_train, y_train):
    # Defines hyperparameter search space
    with st.spinner("...hyperparameter tuning..."):
        param_grid = {
            'gb_m__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'gb_m__n_estimators': [50, 100, 150, 300],
            'gb_m__max_depth': [20, 25, 30, 35],
            'gb_m__min_samples_split': [2, 10],
            'gb_m__max_features': [7, 15, 20],
            'gb_m__min_samples_leaf': [1, 4, 5, 6, 10],
            'gb_m__subsample': [0.2, 0.4, 0.6, 0.8]
        }
        with st.container(border=True):
            pipeline = discover_optimal_params(pipeline, param_grid, X_train, y_train)
            best_model = pipeline
    
    return best_model

def init(use_hypertuning=False):
    # Initialises model pipeline and preprocessing
    features = dm.REGRESSION_FEATURES
    # Separates features by type for preprocessing
    categorical_features = features[0:-1]    # Features for encoding
    numerical_features = [features[-1]]      # Features for scaling

    # Configures preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ])

    # Creates complete model pipeline
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('gb_m', GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLE_SPLIT,
            min_samples_leaf=MIN_SAMPLE_LEAF,
            subsample=SUBSAMPLE,
            random_state=RANDOM_STATE
        ))
    ])

    # Applies log transformation to target variable
    y = np.log1p(st.session_state.regression_data['salary_in_usd'])
    
    # Performs initial cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_pipeline, st.session_state.regression_data[features], 
                                y, cv=cv, scoring='neg_mean_absolute_error')
    cv_mae_scores = np.expm1(-cv_scores)  # Transforms scores back to original scale
    cv_mae_mean = cv_mae_scores.mean()
    cv_mae_std = cv_mae_scores.std()

    # Splits data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        st.session_state.regression_data[features], 
        y,
        test_size=0.2, 
        random_state=42
    )
    
    # Optional hyperparameter tuning
    if use_hypertuning:
        model_pipeline = get_a_hypertuned_model(model_pipeline, X_train, y_train)

    model_pipeline.fit(X_train, y_train)

    # Calculates performance metrics
    train_pred = np.expm1(model_pipeline.predict(X_train))
    test_pred = np.expm1(model_pipeline.predict(X_test))
    train_mae = mean_absolute_error(np.expm1(y_train), train_pred)
    test_mae = mean_absolute_error(np.expm1(y_test), test_pred)
    
    return model_pipeline, preprocessor, {
        'cv_mae_mean': cv_mae_mean,
        'cv_mae_std': cv_mae_std,
        'cv_mae_scores': cv_mae_scores,
        'X': st.session_state.regression_data[features],
        'train_mae': train_mae,
        'test_mae': test_mae,
        'X_train': X_train,
        'X_test': X_test,
        'features': features,
        'categorical_features': categorical_features,
        'numerical_features': numerical_features
    }

def predict(model_pipeline, input_data):
    # Generates salary predictions with confidence intervals
    st.subheader("Prediction Outcome")
    try:
        # Makes prediction using log scale
        log_prediction = model_pipeline.predict(input_data)[0]
        # Transforms prediction back to original scale
        prediction = np.expm1(log_prediction)
        
        # Calculates prediction range
        lower_bound = prediction * 0.85
        upper_bound = prediction * 1.15
        
        # Displays formatted results
        st.success(f"Predicted Salary Range (±15%): \${lower_bound:,.2f} - \${upper_bound:,.2f}")
        st.info(f"Base Prediction: ${prediction:,.2f}")
    
    except Exception as e:
        st.error(f"An error occurred during prediction. Please try different input values.")
        st.write(f"Error details: {str(e)}")

def display_stats(model_pipeline, preprocessor, stats):
    # Displays cross-validation results
    st.subheader("Cross-Validation Results")
    st.write(f"Average MAE across folds: \${stats['cv_mae_mean']:,.2f} ± \${stats['cv_mae_std']:,.2f}")
    cv_results = pd.DataFrame({
        'Fold': range(1, 6),
        'MAE': stats['cv_mae_scores']
    })
    st.table(cv_results.style.format({'MAE': '${:,.2f}'}))

    # Displays performance metrics
    st.subheader("Model Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training MAE", f"${stats['train_mae']:,.2f}")
    with col2:
        st.metric("Testing MAE", f"${stats['test_mae']:,.2f}")

    # Displays model configuration
    st.subheader("Model Information Used")
    model_info = pd.DataFrame({
        'Parameter': ['Algorithm', 'Number of Trees', 'Learning Rate', 'Max Tree Depth', 
                    'Min Samples Split', 'Min Samples Leaf', 'Subsample Ratio', 'Cross-Validation Folds',
                    'Target Transformation'],
        'Value': ['Gradient Boosting', N_ESTIMATORS, LEARNING_RATE, MAX_DEPTH, 
                MIN_SAMPLE_SPLIT, MIN_SAMPLE_LEAF, SUBSAMPLE, '5', 'Log1p']
    })
    st.table(model_info)

    # Displays dataset information
    st.subheader("Testing and Training Information")
    split_info = pd.DataFrame({
        'Metric': ['Total Dataset Size', 'Training Data Size', 'Testing Data Size', 
                'Number of Features', 'Number of Categorical Features', 'Number of Numerical Features',
                'Target Transformation'],
        'Value': [f"{len(stats['X'])} records", f"{len(stats['X_train'])} records", 
                f"{len(stats['X_test'])} records", f"{len(stats['features'])}", 
                f"{len(stats['categorical_features'])}", f"{len(stats['numerical_features'])}",
                "Natural Log (log1p)"]
    })
    st.table(split_info)

    # Analyses feature importance
    st.subheader("Top 10 Most Important Features")
    feature_names = (
        preprocessor.named_transformers_['cat']
        .get_feature_names_out(stats['categorical_features'])
        .tolist() + stats['numerical_features']
    )
    
    importances = model_pipeline.named_steps['gb_m'].feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)

    st.table(feature_importance.style.format({'Importance': '{:.4f}'}))
    st.bar_chart(feature_importance.set_index('Feature'))