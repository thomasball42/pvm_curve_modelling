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

for model_name in ["A", "B", "C", "D"]:

    # path to output fitted data
    data_fits_path = f"..\\results\\data_fits\\data_fits_{model_name}_basic_gompertz.csv"
    model_fit_path = f"..\\results\\predictive_models\\linear_model_{model_name}_basic_gompertz_parameters.csv"
    
    inputs_list = ["RMAX", "QSD", "SA", "QREV"]
    
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
    
    XP = get_polys(X, order = 6)    
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
        
        print(f"{model_name}, {param.upper()}", f" MSE: {mse}, R2: {r2}")
        
        coefs = model.coef_
        
        rounder = 3
        coefs_names = [f"{names[i]}:{round(coefs[i], rounder)}" for i in range(len(coefs))]
        
        # print(f"COEFFS: {coefs_names} \n")
    
        xdf = pd.DataFrame((coefs, names, [mse] * len(coefs), 
                            [r2] * len(coefs),
                            [param] * len(coefs),
                            [f"Model {model_name}"] * len(coefs))).T
        
        xdf.columns = ["TERM", "COEFFICIENT", "MSE", "R2", "PARAM", "MODEL"]
        odf = pd.concat([odf, xdf])
        
    odf.to_csv(model_fit_path)
