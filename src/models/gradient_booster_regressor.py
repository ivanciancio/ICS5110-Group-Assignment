import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

from data import data_mappings as dm


def init():
    # Define features and target
    features = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size', 'remote_ratio']
    # Separate numerical and categorical features
    categorical_features = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']
    numerical_features = ['remote_ratio']

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
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=42
        ))
    ])

    # Perform cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_pipeline, st.session_state.regression_data[dm.REGRESSION_FEATURES], st.session_state.regression_data['salary_in_usd'], cv=cv, scoring='neg_mean_absolute_error')
    cv_mae_scores = -cv_scores
    cv_mae_mean = cv_mae_scores.mean()
    cv_mae_std = cv_mae_scores.std()

    # Split the data and train the final model
    X_train, X_test, y_train, y_test = train_test_split(st.session_state.regression_data[dm.REGRESSION_FEATURES], st.session_state.regression_data['salary_in_usd'], test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)

    # Calculate metrics
    train_pred = model_pipeline.predict(X_train)
    test_pred = model_pipeline.predict(X_test)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    return cv_mae_mean,cv_mae_std,cv_mae_scores,preprocessor,model_pipeline,st.session_state.regression_data[dm.REGRESSION_FEATURES],train_mae,test_mae,X_train,X_test,features,categorical_features,numerical_features


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
        'Value': ['Gradient Boosting', '500', '0.05', '6', '5', '4', '0.8', '5']
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