# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 08:17:00 2024

@author: tom

NOTE THAT 'X' in K'X' is 1-X for P_E curves. (i.e. K10 is  90% chance of extinction)
"""

import pandas as pd
import numpy as np
import os
import functools
import itertools

import sklearn.linear_model
import sklearn.preprocessing
import sklearn.metrics
import sklearn.ensemble

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

ts = 0.20
KX = 90

inputs_list = ["RMAX", "QSD", "QREV", "SA"]
outputs_list = ["param_a", "param_b", "param_alpha"]

output_path = "..\\results\\predictive_models"
outfile_name = os.path.join(output_path, f"linear_model_k{KX}_parameters.csv")


fig, axs = plt.subplots(2,2, sharex=True, sharey=True)

odf = pd.DataFrame()

KXX = f"k{100-KX}"

for zi, z in enumerate(["A", "B", "C", "D"]):
    results_path = f"..\\results\\data_fits\\data_fits_{z}.csv"
    
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
    
    XP = get_polys(X, order = 4)    
    names = XP.columns
    
    Y = df.loc[:, df.columns.isin(outputs_list)]
    Y = df[KXX].to_frame()
    
    
    
    for p, param in enumerate(Y):
        
        dat = Y[param]
        
        model = sklearn.linear_model.LinearRegression()
        # model = sklearn.ensemble.RandomForestRegressor()
        
        x_train, x_test, y_train, y_test = train_test_split(XP, dat, 
                                                            test_size=ts,
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
                            [f"k{KX}"] * len(coefs),
                            [z] * len(coefs))).T
        xdf.columns = ["COEFFICIENT", "TERM", "MSE", "R2", "PARAM", "MODEL"]
        odf = pd.concat([odf, xdf])
        
        # plot
        r = zi //2
        c = zi % 2
        
        ax = axs[r, c]
        # ax.set_xlim(0, 160)
        # ax.set_ylim(-10, 160)
        ax.scatter(dat, model.predict(XP), c = "k", alpha = 0.5, marker = "x",label = f"R2: {round(r2, 5)}")
        ax.set_title(f"Model{z}")
        ax.legend()
    
fig.text(0.5, 0.005, f'Observed k{KX}', ha='center')
fig.text(0.005, 0.5, f'Predicted k{KX}', va='center', rotation='vertical')
fig.set_size_inches(7,7)
fig.tight_layout()

odf.to_csv(outfile_name, index = False)
    
    