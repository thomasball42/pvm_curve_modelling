# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:55:22 2024

@author: Thomas Ball
"""

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# import matplotlib.ticker

# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score

import scipy.optimize

import _curve_fit

od_path = "C:\\Users\\Thomas Ball\\OneDrive - University of Cambridge"

results_path = os.path.join(od_path, "Work\\P_curve_shape\\results")

# =============================================================================
# Load data
# =============================================================================
f = []
for path, subdirs, files in os.walk(results_path):
    for name in files:
        f.append(os.path.join(path, name))
f = [file for file in f if ".csv" in file]
ddf = pd.DataFrame()
for file in f:
    dat = pd.read_csv(file, index_col = 0)
    runName = dat.runName.unique().item()
    model = runName.split("_")[0]
    sa = dat.Sa.unique().item()
    rmax = dat.Rmax.unique().item()
    x = dat.K
    y = dat.P
    func = _curve_fit.gompertz
    try:
        params, y_predicted, R2, RSS = _curve_fit.betterfit_gompertz(func, x, y)
    except RuntimeError: 
        params, y_predicted, R2, RSS = (None, None, None, None)
    ddf.loc[len(ddf), ["runName", "model", "rmax", "sa", "param_a", "param_b", 
                      "param_alpha", "R2"]  ] = [runName, model, rmax, sa, *params,
                                                 R2]

    
#%% ===========================================================================
# do some analysing
# =============================================================================
for model in ddf.model.unique():
    
    df = ddf[ddf.model == model]
    
    input_params = df[['rmax', 'sa']]
    model_params = df[['param_a', 'param_b', 'param_alpha']]
    
    fig, axs = plt.subplots(len(input_params.T), len(model_params.T))
    for i, ip in enumerate(input_params):
        for m, mp in enumerate(model_params):
            ax = axs[i, m]

            x, y, z = df[[ip, mp, input_params.columns.difference([ip]).item()]].values.T
            
            ax.scatter(x, y, c=z, cmap='viridis', alpha=0.7)
            slope, inter, r, p_value, std_err = scipy.stats.linregress(x, y)
            ax.plot(x, _curve_fit.lin(x, slope, inter), label = f"r2:{round(r**2, 2)}")
            ax.set_xlabel(ip)
            ax.set_ylabel(mp)
            ax.legend()
            
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    fig.suptitle(f"Model '{model}'", fontsize=14)
    fig.tight_layout()
    
    
    
        





