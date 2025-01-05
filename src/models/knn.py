import streamlit as st
import numpy as np # linear algebra
import matplotlib.pyplot as plt # for data visualization purposes

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler



def perform():
    #Separating independant variable and dependent variable("salary_group") and removing the salary_in_usd from the dataset
    X = st.session_state.classifiers_data.drop(['salary_group','salary_in_usd'], axis=1)
    y = st.session_state.classifiers_data['salary_group']

    # Split the dataset into training/validation and testing sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split the dataset into training and validation subsets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    st.write("Total number of datapoints: " + str(X.shape[0]))
    st.write("Number of datapoints in the training set: " + str(X_train.shape[0]))
    st.write("Number of datapoints in the validation set: " + str(X_val.shape[0]))
    st.write("Number of datapoints in the test set: " + str(X_test.shape[0]))

    #fit the standardscaler on the training data
    scaler=StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    X_test = scaler.fit_transform(X_test)

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
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(num_neighbours, performance)
    plt.grid(True)
    plt.close(fig)
    st.pyplot(fig)
    best_index = np.argmax(performance)

    # calculate performance metrics for the best value of k = 91
    clf = KNeighborsClassifier (n_neighbors=91, metric='euclidean')
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_val)
    acc=accuracy_score(y_val, y_pred)
    st.write("The performance of the KNN with K=91 is: %.2f"%(acc))

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_val, y_pred, labels=['L', 'M', 'H'])

    #precision = precision_score(y_test, y_pred)
    classification_metrics = classification_report(y_val, y_pred, target_names=['L', 'M', 'H'])

    # Display results
    st.write("Confusion Matrix with K=91:")
    st.write(conf_matrix)
    st.write("")
    st.write("Classification Report with k=91:")
    st.write(classification_metrics)
    
    return clf

