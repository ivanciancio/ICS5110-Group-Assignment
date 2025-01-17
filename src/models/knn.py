import streamlit as st
import numpy as np # linear algebra
import matplotlib.pyplot as plt # for data visualization purposes

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier
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
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # accuracy vs k value distribution with Euclidean distance metric
    num_neighbours = np.arange(1,361)
    performance = []
    for k in num_neighbours:
        clf = KNeighborsClassifier (n_neighbors=k, metric='euclidean')
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_val)
        acc=accuracy_score(y_val, y_pred)
        performance.append(acc)

    # best_index = np.argmax(performance)

    # calculate performance metrics for the best value of k = 91
    clf = KNeighborsClassifier (n_neighbors=91, metric='euclidean')
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_val)
    acc=accuracy_score(y_val, y_pred)

    y_predt = clf.predict(X_test)
    acct=accuracy_score(y_test, y_predt)
    
    return clf, { 'X': X,'y': y,'scaler': scaler,'X_train': X_train,'X_val': X_val,'y_val': y_val,'X_test': X_test,'y_test': y_test,'acc': acc,'acct': acct,'num_neighbours': num_neighbours,'performance': performance,'y_pred': y_pred,'y_predt': y_predt}


def predict(model,scaler,xinput):
    # example of input and mapping
    xinput=[dm.ymapping] + xinput

    x_to_predict=[dm.ymapping,dm.get_value_by_label(xinput[1], dm.exlemapping),dm.get_value_by_label(xinput[2], dm.emtymapping),dm.get_value_by_label(xinput[3], dm.jtmapping),\
                    dm.get_value_by_label(xinput[4], dm.ermapping),dm.get_value_by_label(xinput[5], dm.rrmapping),dm.get_value_by_label(xinput[6], dm.clmapping),\
                    dm.get_value_by_label(xinput[7], dm.csmapping)]

    #new_data_point = np.array([x_to_predict])

    #apply scaler to the input data point
    new_data_point = scaler.transform([x_to_predict])

    # predict salary class for the xinput
    predict_salary_class = model.predict(new_data_point)
    
    class_text = "Low"
    salary_range="lesser than $74,015.60"
    if "M" in predict_salary_class:
        class_text = "Medium"
        salary_range="greater than or equal to 74,015.6 USD and lesser than $128,875.00"
    if "H" in predict_salary_class:
        class_text = "High"
        salary_range="greater than or equal to $128,875.00"
        
    st.success(f"Predicted Salary Class: {predict_salary_class} (i.e. a '{class_text}' class)")
    st.info(f"The salary group prediction for the input data point is {class_text.upper()}, meaning that the salary is {salary_range}.")
    # st.info(f"The salary group prediction for the input vector {x_to_predict} was {predict_salary_class}")


def display_stats(stats):
    st.subheader("Model Information")
    
    st.write("Total number of datapoints: " + str(stats['X'].shape[0]))
    st.write("")
    st.write("Number of datapoints in the training set: " + str(stats['X_train'].shape[0]))
    st.write("Number of datapoints in the validation set: " + str(stats['X_val'].shape[0]))
    st.write("Number of datapoints in the test set: " + str(stats['X_test'].shape[0]))
    st.write("")
    st.write("The performance of the KNN with K=91 is: %.2f"%(stats['acc']))
    st.write("")
    
    # plot the distribution
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(stats['num_neighbours'], stats['performance'])
    plt.grid(True)
    plt.close(fig)
    st.pyplot(fig)

    
    st.write(f"Target feature ditricution {stats['y'].value_counts()}")

    """
    # accuracy vs k value distribution with Euclidean distance metric
    num_neighbours = np.arange(1,361)
    performance = []
    for k in num_neighbours:
        clf = KNeighborsClassifier (n_neighbors=k, metric='euclidean')
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_val)
        acc=accuracy_score(y_val, y_pred)
        performance.append(acc)

    # plot the distribution
    plt.plot(num_neighbours, performance)
    plt.grid(True)
    plt.show()
    best_index = np.argmax(performance)
    """

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(stats['y_val'], stats['y_pred'], labels=['L', 'M', 'H'])

    #precision = precision_score(y_test, y_pred)
    classification_metrics = classification_report(stats['y_val'], stats['y_pred'], target_names=['L', 'M', 'H'])

    # Display results
    st.write("Confusion Matrix with K=91:")
    st.write(conf_matrix)
    st.write("")
    st.write("Classification Report with k=91:")
    st.write(classification_metrics)

    # Compute the confusion matrix
    conf_matrixt = confusion_matrix(stats['y_test'], stats['y_predt'], labels=['L', 'M', 'H'])

    #precision = precision_score(y_test, y_pred)
    classification_metrics_t = classification_report(stats['y_test'], stats['y_predt'], target_names=['L', 'M', 'H'])

    # Display results
    st.write("Confusion Matrix with K=91 with the test set:")
    st.write(conf_matrix)
    st.write("")
    st.write("Classification Report with k=91 with the test set:")
    st.write(classification_metrics)