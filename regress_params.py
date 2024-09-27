# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 08:17:00 2024

@author: tom
"""

import pandas as pd
import numpy as np
import os
import functools
import itertools

import sklearn.linear_model
import sklearn.preprocessing
import sklearn.metrics

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

ts = 0.1

inputs_list = ["RMAX", "QSD", "QREV", "SA"]
outputs_list = ["param_a", "param_b", "param_alpha"]

# od_path = "C:\\Users\\Thomas Ball\\OneDrive - University of Cambridge"
od_path = "E:\\OneDrive\\OneDrive - University of Cambridge"
results_path = os.path.join(od_path, "Work\\P_curve_shape\\dat\\fits_main", "data_fits_A.csv")

# load data
df = pd.read_csv(results_path, index_col = 0)
df = df[~df.model_name.isnull()]

def get_polys(df, order=4):
    poly = sklearn.preprocessing.PolynomialFeatures(order)
    arr = poly.fit_transform(df)
    names= poly.get_feature_names_out()
    return pd.DataFrame(arr, columns = names)

# remove entries with max_y < 1
df = df[df.MAX_Y == 1]

X = df.loc[:, df.columns.isin(inputs_list)]
X = X.loc[:, ~X.isnull().all(axis = 0)]

XP = get_polys(X, order = 3)    
names = XP.columns

Y = df.loc[:, df.columns.isin(outputs_list)]

for p, param in enumerate(Y):
    
    dat = Y[param]
    
    model = sklearn.linear_model.LinearRegression()
    
    x_train, x_test, y_train, y_test = train_test_split(XP, dat, 
                                                        test_size=ts,
                                                        random_state=999)
    
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    
    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    rmse = sklearn.metrics.root_mean_squared_error(y_test, y_pred)
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    coefs = model.coef_
    
    coefs_names = [f"{names[i]}:{coefs[i]}" for i in range(len(coefs))]
    
    print(f"{param.upper()}")
    print(f"MSE: {mse}, R2: {r2}")
    print(f"COEFFS: {coefs_names} \n")



