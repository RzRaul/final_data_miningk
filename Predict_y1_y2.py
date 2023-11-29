# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 04:36:23 2023

@author: Rulas
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import scipy.stats

def run_model(model, X_train, y_train, title):
    model.fit(X_train, y_train)
    
    # y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # evaluate_model(y_train, y_train_pred,"Training")
    evaluate_model(y_test, y_test_pred, "Test")

    

def evaluate_model(true, predicted, label):
    predicted = pd.DataFrame(predicted,columns=true.columns)
    mae,mse, rmse, r2_square,pear = [],[],[],[],[]
    sse = np.array(np.sum((true - predicted)**2))
    x_index = [i for i in range(true.shape[0])]
    for col in true.columns:
        mae.append(mean_absolute_error(true[col], predicted[col]))
        mse.append(mean_squared_error(true[col], predicted[col]))
        rmse.append(np.sqrt(mean_squared_error(true[col], predicted[col])))
        r2_square.append(r2_score(true[col], predicted[col]))
        pear.append(scipy.stats.pearsonr(true[col], predicted[col]).pvalue)
    print('Puntajes de {}'.format(label))
    print("- RMSE {}".format(['{:.4f}'.format(i) for i in rmse]))
    print("- MAE  {}".format(['{:.4f}'.format(i) for i in mae]))
    print("- R2   {}".format(['{:.4f}'.format(i) for i in r2_square]))
    print("- SSE  {}".format(['{:.4f}'.format(i) for i in sse]))
    print("- Corr {}".format(pear))
    print('----------------------------------')
    plt.figure(figsize=(8, 6))
    plt.plot(x_index, predicted, alpha=0.6)
    plt.plot(x_index, true, alpha=0.6)
    plt.xlabel("Valores reales")
    plt.ylabel("Valores predichos")
    plt.title('Valores reales vs predicción {}'.format(label))
    plt.show()


df = pd.read_csv('PruebaconY.csv')
df.describe()
df_clean = df.dropna()

y = df_clean[['y1','y2']]
X = df_clean.drop(columns=['y1','y2','biogas','metano'])


numeric_transformer = StandardScaler()

X = numeric_transformer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=41)
X_train.shape, X_test.shape

modelLinear = LinearRegression()
run_model(modelLinear, X_train, y_train, 'Linear Regression')

modelTree = DecisionTreeRegressor(random_state=0)
run_model(modelTree, X_train, y_train, 'DecisionTreeRegressor')

modelNN = MLPRegressor(random_state=42, max_iter=1000,batch_size=5).fit(X_train, y_train)
run_model(modelNN, X_train, y_train, 'MLPRegressor')


