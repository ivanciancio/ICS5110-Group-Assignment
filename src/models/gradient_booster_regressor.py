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

from data import data_mappings as dm


# Model Parameter Tuning
N_ESTIMATORS = 500
RANDOM_STATE = 42
MAX_DEPTH = 6
MIN_SAMPLE_SPLIT = 5
MIN_SAMPLE_LEAF = 4


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
    
    global N_ESTIMATORS,MAX_DEPTH,MIN_SAMPLE_SPLIT,MIN_SAMPLE_LEAF
    N_ESTIMATORS = best_params['regressor__n_estimators']
    MAX_DEPTH = best_params['regressor__max_depth']
    MIN_SAMPLE_SPLIT = best_params['regressor__min_samples_split']
    MIN_SAMPLE_LEAF = best_params['regressor__min_samples_leaf']
    
    # st.success("Tuning Complete")
    # st.write("Best Parameters:", best_params)
    # st.write(f"Best MAE: {best_score}")
    
    return pipeline


def get_a_hypertuned_model(pipeline, X_train, y_train):
    # Hyperparameter tuning
    with st.spinner("...hyperparameter tuning..."):
        param_grid = {
            'regressor__n_estimators': [50, 100, 150, 300],
            'regressor__max_depth': [20, 25, 30, 35],
            'regressor__min_samples_split': [2, 10],
            'regressor__max_features': [7, 15, 20, 'sqrt', 'log2', None],
            'regressor__min_samples_leaf': [1, 4, 5]
        }
        with st.container(border=True):
            pipeline = discover_optimal_params(pipeline, param_grid, X_train, y_train)
            best_model = pipeline

        # with st.container(border=True):
        #     grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=2)
        #     grid_search.fit(X_train, y_train)
        #     best_model = grid_search.best_estimator_
    
    return best_model



def init(use_hypertuning=False):
    # Define features and target
    features = dm.REGRESSION_FEATURES
    # Separate numerical and categorical features
    categorical_features = features[0:-1]    # Features needing encoding
    numerical_features = [features[-1]]    # Features to remain numeric

    # Create preprocessing pipeline with both scaling and encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ])

    # Create model pipeline with GradientBoostingRegressor
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.05,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLE_SPLIT,
            min_samples_leaf=MIN_SAMPLE_LEAF,
            subsample=0.8,
            random_state=RANDOM_STATE
        ))
    ])

    # Perform cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_pipeline, st.session_state.regression_data[features], st.session_state.regression_data['salary_in_usd'], cv=cv, scoring='neg_mean_absolute_error')
    cv_mae_scores = -cv_scores
    cv_mae_mean = cv_mae_scores.mean()
    cv_mae_std = cv_mae_scores.std()

    # Split the data and train the final model
    X_train, X_test, y_train, y_test = train_test_split(st.session_state.regression_data[features], st.session_state.regression_data['salary_in_usd'], test_size=0.2, random_state=42)
    if use_hypertuning:
        model_pipeline = get_a_hypertuned_model(model_pipeline,X_train,y_train)    # Train a hypertuned model
    else:
        model_pipeline.fit(X_train, y_train)

    # Calculate metrics
    train_pred = model_pipeline.predict(X_train)
    test_pred = model_pipeline.predict(X_test)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    return cv_mae_mean,cv_mae_std,cv_mae_scores,preprocessor,model_pipeline,st.session_state.regression_data[features],train_mae,test_mae,X_train,X_test,features,categorical_features,numerical_features


def predict(model_pipeline,input_data):
    st.subheader("Prediction Outcome")
    # Make prediction
    try:
        prediction = model_pipeline.predict(input_data)[0]
        
        # Calculate salary range (±15% from prediction)
        lower_bound = prediction * 0.85
        upper_bound = prediction * 1.15
        
        # Display results with enhanced formatting
        st.success(f"Predicted Salary Range (±15%): \${lower_bound:,.2f} - \${upper_bound:,.2f}")
        st.info(f"Base Prediction: ${prediction:,.2f}")
    
    except Exception as e:
        st.error(f"An error occurred during prediction. Please try different input values.")
        st.write(f"Error details: {str(e)}")


def display_results(cv_mae_mean,cv_mae_std,cv_mae_scores,preprocessor,model_pipeline,X,train_mae,test_mae,X_train,X_test,features,categorical_features,numerical_features):
    # Display cross-validation results
    st.subheader("Cross-Validation Results")
    st.write(f"Average MAE across folds: \${cv_mae_mean:,.2f} ± \${cv_mae_std:,.2f}")
    cv_results = pd.DataFrame({
        'Fold': range(1, 6),
        'MAE': cv_mae_scores
    })
    st.table(cv_results.style.format({'MAE': '${:,.2f}'}))

    # Display model performance metrics
    st.subheader("Model Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training MAE", f"${train_mae:,.2f}")
    with col2:
        st.metric("Testing MAE", f"${test_mae:,.2f}")

    # Display model information
    st.subheader("Model Information Used")
    model_info = pd.DataFrame({
        'Parameter': ['Algorithm', 'Number of Trees', 'Learning Rate', 'Max Tree Depth', 'Min Samples Split', 'Min Samples Leaf', 'Subsample Ratio', 'Cross-Validation Folds'],
        'Value': ['Gradient Boosting', N_ESTIMATORS, '0.05', MAX_DEPTH, MIN_SAMPLE_SPLIT, MIN_SAMPLE_LEAF, '0.8', '5']
    })
    st.table(model_info)

    # Display testing and training information
    st.subheader("Testing and Training Information")
    split_info = pd.DataFrame({
        'Metric': ['Total Dataset Size', 'Training Data Size', 'Testing Data Size', 
                'Number of Features', 'Number of Categorical Features', 'Number of Numerical Features'],
        'Value': [f"{len(X)} records", f"{len(X_train)} records", f"{len(X_test)} records", 
                f"{len(features)}", f"{len(categorical_features)}", f"{len(numerical_features)}"]
    })
    st.table(split_info)

    # Display feature importance
    st.subheader("Top 10 Most Important Features")
    feature_names = (
        preprocessor.named_transformers_['cat']
        .get_feature_names_out(categorical_features)
        .tolist() + numerical_features
    )
    
    importances = model_pipeline.named_steps['regressor'].feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)

    st.table(feature_importance.style.format({'Importance': '{:.4f}'}))
    st.bar_chart(feature_importance.set_index('Feature'))