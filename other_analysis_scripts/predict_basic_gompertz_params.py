# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 14:20:12 2025

@author: Thomas Ball
"""

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker

import functools
import itertools

import sklearn.linear_model
import sklearn.preprocessing
import sklearn.metrics
import sklearn.ensemble

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

test_split = 0.15

# my onedrive path, computer dependent..
od_path = "C:\\Users\\Thomas Ball\\OneDrive - University of Cambridge"
# od_path = "E:\\OneDrive\\OneDrive - University of Cambridge"

# dir that the simulation outputs are in
results_path = os.path.join(od_path, "Work\\P_curve_shape\\version2\\dat\\simulation_results\\results_model_A_bfit")
# path to output fitted data
data_fits_path = os.path.join(od_path, "Work\\P_curve_shape\\version2\\dat\\data_fits_A_basic_gompertz.csv")
model_fit_path = os.path.join(od_path, "Work\\P_curve_shape\\version2\\dat\\linear_model_A_basic_gompertz.csv")

inputs_list = ["RMAX", "QSD"]

df = pd.read_csv(data_fits_path, index_col=0)

outputs_list = [_ for _ in df.columns if "param" in _]

df = df[df.MAX_Y == 1]

X = df.loc[:, df.columns.isin(inputs_list)]
X = X.loc[:, ~X.isnull().all(axis = 0)]

def get_polys(df, order=4):
    poly = sklearn.preprocessing.PolynomialFeatures(order)
    arr = poly.fit_transform(df)
    names= poly.get_feature_names_out()
    return pd.DataFrame(arr, columns = names)

XP = get_polys(X, order = 4)    
names = XP.columns
Y = df.loc[:, df.columns.isin(outputs_list)]

odf = pd.DataFrame()

for p, param in enumerate(Y):
    
    dat = Y[param]
    
    model = sklearn.linear_model.LinearRegression()
    # model = sklearn.ensemble.RandomForestRegressor()
    
    x_train, x_test, y_train, y_test = train_test_split(XP, dat, 
                                                        test_size=test_split,
                                                        random_state=999)
    
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    
    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    
    print(f"{param.upper()}")
    print(f"MSE: {mse}, R2: {r2}")
    
    coefs = model.coef_
    
    rounder = 3
    coefs_names = [f"{names[i]}:{round(coefs[i], rounder)}" for i in range(len(coefs))]
    
    print(f"COEFFS: {coefs_names} \n")

    xdf = pd.DataFrame((coefs, names, [mse] * len(coefs), 
                        [r2] * len(coefs),
                        [param] * len(coefs),
                        ["Model A"] * len(coefs))).T
    
    xdf.columns = ["TERM", "COEFFICIENT", "MSE", "R2", "PARAM", "MODEL"]
    odf = pd.concat([odf, xdf])
    
odf.to_csv(model_fit_path)
