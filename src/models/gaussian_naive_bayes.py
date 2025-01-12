import streamlit as st
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from data import data_mappings as dm


def init():
    #Separating independant variable and dependent variable("salary_group") and removing the salary_in_usd from the dataset
    X = st.session_state.classifiers_data.drop(['salary_group','salary_in_usd'], axis=1)
    y = st.session_state.classifiers_data['salary_group']

    # Split the dataset into training/validation and testing sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split the dataset into training and validation subsets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    #fit the standardscaler on the training data
    scaler=StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    X_test = scaler.fit_transform(X_test)

    # calculate performance metrics for the best value of k = 91
    clf = GaussianNB ()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_val)
    acc=accuracy_score(y_val, y_pred)
    
    return clf, { 'X': X,'X_train': X_train,'X_val': X_val,'y_val': y_val,'X_test': X_test,'acc': acc,'y_pred': y_pred }


def predict(model,xinput):
    # example of input and mapping
    xinput=[dm.ymapping] + xinput

    x_to_predict=[dm.ymapping,dm.get_value_by_label(xinput[1], dm.exlemapping),dm.get_value_by_label(xinput[2], dm.emtymapping),dm.get_value_by_label(xinput[3], dm.jtmapping),\
                    dm.get_value_by_label(xinput[4], dm.ermapping),dm.get_value_by_label(xinput[5], dm.rrmapping),dm.get_value_by_label(xinput[6], dm.clmapping),\
                    dm.get_value_by_label(xinput[7], dm.csmapping)]

    new_data_point = np.array([x_to_predict])

    # predict salary class for the xinput
    predict_salary_class = model.predict(new_data_point)
    
    class_text = "Low"
    if "M" in predict_salary_class:
        class_text = "Medium"
    if "H" in predict_salary_class:
        class_text = "High"
        
    st.success(f"Predicted Salary Class: {predict_salary_class} (i.e. a '{class_text}' class)")
    st.info(f"The salary group prediction for the input vector {x_to_predict} was {predict_salary_class}")


def display_stats(stats):
    st.subheader("Model Information")
    
    st.write("Total number of datapoints: " + str(stats['X'].shape[0]))
    st.write("")
    st.write("Number of datapoints in the training set: " + str(stats['X_train'].shape[0]))
    st.write("Number of datapoints in the validation set: " + str(stats['X_val'].shape[0]))
    st.write("Number of datapoints in the test set: " + str(stats['X_test'].shape[0]))
    st.write("")
    st.write("The performance of the Gaussian Naive Bayes is: %.2f"%(stats['acc']))
    st.write("")

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(stats['y_val'], stats['y_pred'], labels=['L', 'M', 'H'])

    #precision = precision_score(y_test, y_pred)
    classification_metrics = classification_report(stats['y_val'], stats['y_pred'], target_names=['L', 'M', 'H'])

    # Display results
    st.write("Confusion Matrix with Gaussian Naive Bayes:")
    st.write(conf_matrix)
    st.write("")
    st.write("Classification Report with Gaussian Naive Bayes:")
    st.write(classification_metrics)
