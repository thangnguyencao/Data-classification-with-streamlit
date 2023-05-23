from csv import list_dialects
from re import I
import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder


def check(c, x):
    d = 0
    for i in x:
        if c!=i: 
            d+=1
    return d

st.title("Classifier")
st.header("Upload dataset")
uploaded_file = st.file_uploader("Choose file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    df = "data/" + uploaded_file.name
    with open(df, "wb") as f:
        f.write(bytes_data) 

    st.header("Display dataset")
    dataframe = pd.read_csv(df)
    st.write(dataframe)
    
    st.header("Input features")
    X = dataframe
    X.drop(['Sample code number'], axis = 1)
    for i in X.columns:
        agree = st.checkbox(i)
        if agree == False:
            X = X.drop(i, 1)
    st.write(X)
    flag = 0
    for i in X.columns:
        if X[i].dtypes == object:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            flag = 1
    
    st.header("Output features")
    d = 0
    y = dataframe
    if flag == 0:
        for i in dataframe.columns:
            agree_1 = False
            if check(i, X.columns) == len(X.columns):
                agree_1 = st.checkbox(i, False, str(d))
                d+=1
            if agree_1 == False:
                y = y.drop(i, 1)
    else:
        for i in dataframe.columns:
            agree_1 = st.checkbox(i, 1)
            if agree_1 == False:
                y = y.drop(i, 1)
    st.write(y)

    st.header("Hyperparameters")
    train_per = st.slider(
        'Train Test split',
        0, 100, 80)
    st.write('Training', train_per,'%')
    st.write('Test', 100 - train_per,'%')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_per) * 0.01, random_state=0)

    st.header("Model")
    type_model = st.radio(
        "Choose regression model to train",
        ('Logistic Regression', 'Decision Tree', 'SVM', 'XGBoost'))

    if st.button("Run"):
        if type_model == 'Logistic Regression':
            st.write('Logistic Regression init')
            lr = LogisticRegression().fit(X_train, y_train)

            y_pred = lr.predict(X_test)
            st.write(y_pred)
            st.write(y_test)

            #rmse = math.sqrt(mean_squared_error(y_test, y_pred))
            mse = mean_squared_error(y_test,y_pred)
            mae = mean_absolute_error(y_test,y_pred)
            matricesMAE = st.checkbox('MAE')
            matricesMSE = st.checkbox('MSE')
            # st.write('Root mean squared error:', rmse)
            if matricesMSE: 
                st.write('Mean squared error:', mse)
            if matricesMAE:
                st.write('Mean absolute error:', mae)
            st.write('Score: ', lr.score(X_test, y_test))

        elif type_model == 'Decision Tree':
            st.write('Decision Tree init')
            dt = DecisionTreeRegressor().fit(X_train, y_train)

            y_pred = dt.predict(X_test)
            st.write(y_pred)
            st.write(y_test)

            rmse = math.sqrt(mean_squared_error(y_test, y_pred))
            st.write('Root mean squared error:', rmse)
            st.write('Score:', dt.score(X_test, y_test))
        
        elif type_model == 'SVM':
            st.write('SVM init')
            svm =  SVC().fit(X_train, y_train)

            y_pred = svm.predict(X_test)
            st.write(y_pred)
            st.write(y_test)

            rmse = math.sqrt(mean_squared_error(y_test, y_pred))
            st.write('Root mean squared error:', rmse)
            st.write('Score:', svm.score(X_test, y_test))

            
        elif type_model == 'XGBoost':
            st.write('XGBoost init')
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            xg = xgb.XGBClassifier()
            xg.fit(X_train,y_train)

            y_pred = xg.predict(X_test)
            st.write(y_pred)
            st.write(y_test)

            rmse = math.sqrt(mean_squared_error(y_test, y_pred))
            st.write('Root mean squared error:', rmse)
            st.write('Score:', xg.score(X_test, y_test))
            st.write('Acc:', accuracy_score(y_test,y_pred))

    st.header("Compare different models")
    options = st.multiselect(
    'Models will be used to compared:',
    ['Logistic Regression', 'Decision Tree', 'SVM','XGBoost'],
    ['Logistic Regression', 'Decision Tree', 'SVM','XGBoost'])

    st.write('Models selected:', options)
    df_acc = pd.DataFrame(columns = ['Models', 'Accuracy'])
    for i in options:
        if i == 'Logistic Regression':
            lr = LogisticRegression().fit(X_train, y_train)
            df_acc = df_acc.append({'Models' : 'LR', 'Accuracy' : lr.score(X_test, y_test)}, ignore_index = True)
        elif i == 'Decision Tree':
            dt = DecisionTreeClassifier().fit(X_train, y_train)
            df_acc = df_acc.append({'Models' : 'DT', 'Accuracy' : dt.score(X_test, y_test)}, ignore_index = True)
        elif i == 'SVM':
            svm =  SVC().fit(X_train, y_train)
            df_acc = df_acc.append({'Models' : 'SVM', 'Accuracy' : svm.score(X_test, y_test)}, ignore_index = True)
        elif i == 'XGBoost':
            xg = xgb.XGBClassifier();
            sv = xg.fit(X_train, y_train)
            df_acc = df_acc.append({'Models' : 'SVR Poly', 'Accuracy' : sv.score(X_test, y_test)}, ignore_index = True)
    st.write(df_acc)