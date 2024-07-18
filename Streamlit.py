import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score, mean_squared_error, r2_score, confusion_matrix

#read data
# for i in range(1, 11):
#     if i == 1:
#         df=pd.read_csv('data_preprocessed-{}.csv'.format(str(i)))
#     else:
#         df = pd.concat([df, pd.read_csv('data_preprocessed-{}.csv'.format(str(i)))], axis=0)

# to avoid heavy file
X_test = pd.read_csv('X_test.csv', index_col=0)
y_test_reg = pd.read_csv('y_test_reg.csv', index_col=0)
y_test_clf = pd.read_csv('y_test_clf.csv', index_col=0)

# Scaling data (only numerical features)
numerical_features = ["m (kg)","ep (KW)","Erwltp (g/km)","Fuel consumption"]
scaler = joblib.load("scaler")
X_test[numerical_features] = scaler.transform(X_test[numerical_features])



# Streamlit

st.title("CO2 emissions by vehicles")
st.sidebar.title("Table of contents")


pages=["Home", "Exploration", "Preprocessing", "Modelling", "Optimization", "Interpretation"]
page=st.sidebar.radio("Go to", pages)

if page == pages[0] : 
    st.write("### Home")
    st.write("Title of Project: CO2 emissions by vehicles")
    st.write("Objective: We aim to identify the vehicles with the most CO2 emissions and find the cars contributing the most to air pollution.")
    st.write("Scope: We will use the dataset “Monitoring of CO2 emissions from passenger cars” of 2021, distributed by the European Environment Agency. The data is discussed in the Data Audit File.")

if page == pages[1] : 
    st.write("### Exploration")
    st.write("### Head of the data:")
    st.dataframe(X_test.head(10))
    
    st.write("### Shape of the data:")
    st.write(X_test.shape)
    
    st.write("### Description of the data:")
    st.dataframe(X_test.describe())

    st.write("### Check existing of NaN data:")
    if st.checkbox("Show NA") :
        st.dataframe(X_test.isna().sum())
    
    st.write("### Check data Types for different columns:")
    st.dataframe(X_test.dtypes)

if page == pages[2] : 
    st.write("### Preprocessing")
    fig = plt.figure()
    sns.countplot(x = 'target_clf', data = y_test_clf)
    plt.xlabel("Ewltp (g/km)")
    plt.title("Distribution of the CO2 emissions of registered cars")
    st.pyplot(fig)

    fig = plt.figure()
    sns.histplot(x = 'Fuel consumption', data = X_test, bins='auto')
    plt.title("Distribution of Fuel consumption of registered cars")
    plt.tight_layout()
    st.pyplot(fig)

    # fig = plt.figure()
    # sns.countplot(x = 'Pclass', data = df)
    # plt.title("Distribution of the passengers class")
    # st.pyplot(fig)

    # fig = sns.displot(x = 'Age', data = df)
    # plt.title("Distribution of the passengers age")
    # st.pyplot(fig)

    # fig = plt.figure()
    # sns.countplot(x = 'Survived', hue='Sex', data = df)
    # st.pyplot(fig)

    # fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    # st.pyplot(fig)

    # fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    # st.pyplot(fig)


if page == pages[3] :
    st.write("### Modelling")
    subpages=["Classification", "Regression", "Neural Networks"]
    subpage=st.sidebar.radio("Choose Modelling Type:", subpages)

    def classifier(model):
        if model == 'Decision Tree':
            clf = joblib.load("Models/Classification/DecisionTreeClassifier.sav")
        elif model == 'KNeighbors':
            clf = joblib.load("Models/Classification/KNeighborsClassifier.sav")
        elif model == 'Logistic Regression':
            clf = joblib.load("Models/Classification/LogisticRegression.sav")
        elif model == 'XGBoost':
            clf = joblib.load("Models/Classification/XGBClassifier.sav")
        return clf

    def regressor(classifier):
        if classifier == 'Decision Tree':
            reg = joblib.load("Models/Regression/DecisionTreeRegressor.sav")
        elif classifier == 'Elastic Net':
            reg = joblib.load("Models/Regression/ElasticNet.sav")
        elif classifier == 'Linear Regression':
            reg = joblib.load("Models/Regression/LinearRegression.sav")
        elif classifier == 'XGBoost':
            reg = joblib.load("Models/Regression/XGBRegressor.sav")
        return reg

    def NN_Model(task):
        if task == 'Classification':
            model = joblib.load("Models/NN/NN Classification.sav")
        elif task == 'Regression':
            model = joblib.load("Models/NN/NN Regression.sav")
        return model
                    
    def classification_accuracy_check(clf, choice):
        if choice == 'Accuracy':
            return clf.score(X_test, y_test_clf)
        elif choice == 'Confusion matrix':
            return confusion_matrix(y_test_clf, clf.predict(X_test))

    def regression_accuracy_check(reg, choice):
        if choice == 'Accuracy':
            return reg.score(X_test, y_test_reg)
        elif choice == 'RMSE':
            y_test_pred = reg.predict(X_test)
            return np.sqrt(mean_squared_error(y_test_reg, y_test_pred))
                     
    if subpage == subpages[0]:
        st.write("### Classification")

        choice = ['Decision Tree', 'KNeighbors', 'Logistic Regression', 'XGBoost']
        option = st.selectbox('Choose the model', choice)
        st.write('The chosen model is :', option)

        clf = classifier(option)
        display = st.radio('What do you want to show ?', ('Accuracy', 'Confusion matrix'))
        if display == 'Accuracy':
            st.write(classification_accuracy_check(clf, display))
        elif display == 'Confusion matrix':
            st.dataframe(classification_accuracy_check(clf, display))

    if subpage == subpages[1]:
        st.write("### Regression")

        choice = ['Decision Tree', 'Elastic Net', 'Linear Regression', 'XGBoost']
        option = st.selectbox('Choose the model', choice)
        st.write('The chosen model is :', option)

        reg = regressor(option)
        display = st.radio('What do you want to show ?', ('Accuracy', 'RMSE'))
        if display == 'Accuracy':
            st.write(regression_accuracy_check(reg, display))
        elif display == 'RMSE':
            st.write(regression_accuracy_check(reg, display))

    if subpage == subpages[2]:
        st.write("### Neural Networks")

        choice = ['Classification', 'Regression']
        option = st.selectbox('Choose the task', choice)
        st.write('The chosen task is :', option)

        model = NN_Model(option)
        if option == 'Classification':
            display = st.radio('What do you want to show ?', ('Accuracy', 'Confusion matrix'))
            if display == 'Accuracy':
                st.write(classification_accuracy_check(model, display))
            elif display == 'Confusion matrix':
                st.dataframe(classification_accuracy_check(model, display))
        elif option == 'Regression':
            display = st.radio('What do you want to show ?', ('Accuracy', 'RMSE'))
            if display == 'Accuracy':
                st.write(regression_accuracy_check(model, display))
            elif display == 'RMSE':
                st.write(regression_accuracy_check(model, display))
            
if page == pages[4] : 
    st.write("### Optimization")


if page == pages[5] : 
    st.write("### Interpretation")