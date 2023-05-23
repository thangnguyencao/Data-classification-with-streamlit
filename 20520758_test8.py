import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 

st.title("Grid Search")
st.header("Upload dataset")
uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    df = "data/" + uploaded_file.name
    with open(df, "wb") as f:
        f.write(bytes_data) 

    dataframe = pd.read_csv(df)
    st.write(dataframe)
    
    st.header("Input features")
    X = dataframe.iloc[:, :-1]
    for i in X.columns:
        agree = st.checkbox(i, 1)
        if agree == False:
            X = X.drop(i, 1)
    st.write(X)
    
    st.header("Outputs")
    y = dataframe.iloc[:, -1]
    st.write(y)

    st.header("Parameters:")
    train_per = st.slider(
        'Train Test Split',
        0, 100, 80)
    st.write('Train', train_per,'%')
    st.write('Test', 100 - train_per,'%')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_per) * 0.01, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    st.header("Tuning model")
    tune = st.radio("", ('Yes', 'No'))

    st.header("Hyper-parameters:")
    st.subheader('Kernel:')
    kernel = st.radio("Model:", ('linear', 'poly', 'rbf', 'sigmoid'))
    st.subheader('Choose C:')
    c = st.number_input('C:')
    st.subheader('Choose gamma:')
    gamma = st.number_input('Gamma:')

    st.header("K-Fold Cross-validation")
    k_fold = st.checkbox('K-Fold Cross-validation')
    if k_fold == True:
        num = st.number_input('Insert the number of fold:')
        st.write('The number is ', num)
        num = int(num)

    if st.button('Run'):
        st.write('SVM init')
        df_me = pd.DataFrame(columns = ['F1-score', 'Accuracy', 'Precision', 'Recall'])
        df_me_2 = pd.DataFrame(columns = ['F1-score', 'Accuracy', 'Precision', 'Recall'])
        svm = SVC(C=c, kernel=kernel, gamma=gamma)

        if k_fold == True:
            folds = KFold(n_splits = num, shuffle = True, random_state = 100)
            scores = cross_val_score(svm, X_train, y_train, scoring='f1_macro', cv=folds)
            scores_2 = cross_val_score(svm, X_train, y_train, scoring='accuracy', cv=folds)
            scores_3 = cross_val_score(svm, X_train, y_train, scoring='precision', cv=folds)
            scores_4 = cross_val_score(svm, X_train, y_train, scoring='recall', cv=folds)
            for i in range(len(scores)):
                df_me = df_me.append({'F1-score' : scores[i], 'Accuracy' : scores_2[i], 'Precision' : scores_3[i], 'Recall' : scores_4[i]}, ignore_index = True)
            st.write(df_me)

            parameters = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 
                        'C':[0.1, 1, 10, 100, 1000],
                        'gamma':['scale', 'auto'],
                        'gamma':[1, 0.1, 0.01, 0.001, 0.0001]}
            clf = GridSearchCV(estimator=SVC(), param_grid=parameters, refit = True, verbose = 3, n_jobs=-1, cv=folds)
            clf.fit(X_train, y_train)
            if tune == 'Yes':
                st.write("After hyper-parameter tuning")
                st.write(clf.best_estimator_)
                st.write("Best parameters set:")
                st.write(clf.best_params_)
                st.write("Best score:")
                st.write(clf.best_score_)
                scores = cross_val_score(clf, X_train, y_train, scoring='f1_macro', cv=folds)
                scores_2 = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=folds)
                scores_3 = cross_val_score(clf, X_train, y_train, scoring='precision', cv=folds)
                scores_4 = cross_val_score(clf, X_train, y_train, scoring='recall', cv=folds)
                for i in range(len(scores)):
                    df_me_2 = df_me_2.append({'F1-score' : scores[i], 'Accuracy' : scores_2[i], 'Precision' : scores_3[i], 'Recall' : scores_4[i]}, ignore_index = True)
                st.write(df_me, df_me_2)
                st.subheader('Visualize after using GridSearchCV')
                st.bar_chart(df_me_2['F1-score'])
                st.bar_chart(df_me_2['Accuracy'])
                st.bar_chart(df_me_2['Precision'])
                st.bar_chart(df_me_2['Recall'])
        else:
            svm.fit(X_train, y_train) 
            y_pred = svm.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            pre = precision_score(y_test, y_pred)
            re = recall_score(y_test, y_pred)
            df_me = df_me.append({'F1-score' : f1, 'Accuracy' : acc, 'Precision' : pre, 'Recall' : re}, ignore_index = True)
            st.write(df_me)

            parameters = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 
                        'C':[0.1, 1, 10, 100, 1000],
                        'gamma':['scale', 'auto'],
                        'gamma':[1, 0.1, 0.01, 0.001, 0.0001]}
            clf = GridSearchCV(estimator=SVC(), param_grid=parameters, refit = True, verbose = 3, n_jobs=-1)
            clf.fit(X_train, y_train)
            if tune == 'Yes':
                st.write("After hyper-parameter tuning")
                st.write(clf.best_estimator_)
                st.write("Best parameters set:")
                st.write(clf.best_params_)
                st.write("Best score:")
                st.write(clf.best_score_)
                y_pred_2 = clf.predict(X_test)
                f1_2 = f1_score(y_test, y_pred_2)
                acc_2 = accuracy_score(y_test, y_pred_2)
                pre_2 = precision_score(y_test, y_pred_2)
                re_2 = recall_score(y_test, y_pred_2)
                df_me = df_me.append({'F1-score' : f1_2, 'Accuracy' : acc_2, 'Precision' : pre_2, 'Recall' : re_2}, ignore_index = True)
                st.write(df_me)
                st.bar_chart(df_me['F1-score'])
                st.bar_chart(df_me['Accuracy'])
                st.bar_chart(df_me['Precision'])
                st.bar_chart(df_me['Recall'])

                