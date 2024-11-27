# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:07:29 2024

@author: tom
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import _curve_fit

import scipy.stats
from scipy.stats import gaussian_kde as kde
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

od_path = "C:\\Users\\Thomas Ball\\OneDrive - University of Cambridge"
# od_path = "E:\\OneDrive\\OneDrive - University of Cambridge"

fig = plt.figure()
axs = []
for i, I in enumerate(["A", "B"]):
    
    results_path = os.path.join(od_path, "Work\\P_curve_shape\\dat\\fits_main", f"data_fits_{I}.csv")
    
    ddf = pd.read_csv(results_path, index_col = 0)
    ddf = ddf[ddf.MAX_Y == 1]
    ddf = ddf[np.abs(ddf.param_a) < ddf.param_a.max() * 1]

    density = 30
    
    def interpolateZ(x, y, z, density):
        x = x[~np.isnan(z)]
        y = y[~np.isnan(z)]
        z = z[~np.isnan(z)]
        xg, yg = np.meshgrid(np.linspace(x.min(), x.max(), density), 
                              np.linspace(y.min(), y.max(), density))
        zg = griddata((x, y), z, (xg, yg), method='cubic')
        return xg, yg, zg
    
    def predict_z(x, y, z, x_val, y_val, method='cubic'):
        x = x[~np.isnan(z)]
        y = y[~np.isnan(z)]
        z = z[~np.isnan(z)]
        z_pred = griddata((x, y), z, (x_val, y_val), method=method)
        return z_pred
    
    for model in ddf.model.unique():
        
        df = ddf[ddf.model == model]
        
        input_params = df[['RMAX', 'QSD']]
        ipl = {"RMAX": "r$_{max}$",
                "QSD"  : "$\sigma$",
                "param_a" : "$a$",
                "param_b" : "$b$",
                "param_alpha" : "$\gamma$"}
        model_params = df[['param_a', 'param_b', 'param_alpha']]
        
        for m, mp in enumerate(model_params):
            
            ax = fig.add_subplot(2, len(model_params.columns), (3*i)+m+1, projection='3d')
            axs.append(ax)
            
            x, y = input_params.values.T
            z = np.array(df[mp])
            
            xg, yg, zg = interpolateZ(x, y, z, density)
            
            ax.scatter(x, y, z, alpha=0.9, s=30, c = matplotlib.cm.get_cmap('viridis')(0.3) )
            ax.set_title(f"Param {ipl[mp]}")
            ax.plot_surface(xg, yg, zg, alpha=0.5, cmap = matplotlib.cm.get_cmap('viridis'))
            ax.set_xlabel(ipl[input_params.columns[0]])
            ax.set_ylabel(ipl[input_params.columns[1]])
            ax.set_zlabel(ipl[mp])

fig.text(0.01, 0.90, "a", size = 17, va='center', rotation='horizontal')
fig.text(0.01, 0.40, "b", size = 17, va='center', rotation='horizontal')

fig.set_size_inches(8, 6)
# fig.suptitle(f"Model '{model}'", fontsize=14)
fig.tight_layout(pad = 2.12)
# plt.show()
    