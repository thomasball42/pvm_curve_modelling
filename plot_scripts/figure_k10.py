# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:01:14 2024

@author: Thomas Ball
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

import _curve_fit

results_path = "..\\results\\data_fits"

g_inv = _curve_fit.mod_gompertz

cmap = matplotlib.colormaps['viridis']

s = 75

fig, axs = plt.subplots(1, 2, sharex=False, sharey=True)

for i, I in enumerate(["A", "B"]):
    
    fp = os.path.join(results_path, f"data_fits_{I}.csv")
    df = pd.read_csv(fp, index_col = 0)
    df = df[~(df.MAX_Y < 1)]
    
    # ax = axs[0, i]
    ax = axs[i]
    
    for qsd in df.QSD.unique():
        
        
        dat = df[df.QSD==qsd]
        color = cmap((qsd - df.QSD.min())/(df.QSD.max()-df.QSD.min()))
        
        label = f"$\\sigma$={round(qsd, 3)}"
        x = dat.RMAX

        y = dat.k90
        
        ax.plot(x, y, color = color, alpha = 1,)
        ax.scatter(x, y, color = color, alpha = 0.8, marker = "o", s = s,label = label)
        
        ax.set_xlabel("$r_{max}$")
    
        ax.text(0.05, 0.95, I.lower(), transform=ax.transAxes, ha='left', va='top', fontsize=12)
        
# # axs[0,0].legend()
# # axs[0,1].legend(ncols=2)

# # axs[0].legend(ncols=2)
# axs[1].legend(ncols=2)
    
# ax = axs[1, 0]
# ax = axs[0]

# df = pd.read_csv(os.path.join(results_path, "data_fits_C.csv"),
#                  index_col=0)
# df = df[~(df.MAX_Y < 1)]

# qsd = 0.08
# for sa in df.SA.unique():
    
#     dat = df[(df.SA==sa)&(df.QSD==qsd)]
#     color = cmap((sa - df.SA.min())/(df.SA.max()-df.SA.min()))
#     label = f"$S_a$={round(sa, 3)}"
#     x = dat.RMAX
#     y = dat.k90
#     ax.plot(x, y, color = color, alpha = 1,)
#     ax.scatter(x, y, color = color, alpha = 0.8, s = s, label = label)
#     ax.set_xlabel("$r_{max}$")
# ax.text(0.05, 0.95, "a", transform=ax.transAxes, ha='left', va='top', fontsize=12)    
# ax.legend()

# #%%
# # ax = axs[1, 1]
# ax = axs[1]

# sa = 0.35
# qsd = 0.08
# df = pd.read_csv(os.path.join(results_path, "data_fits_D.csv"),
#                  index_col=0)
# df = df[~(df.MAX_Y < 1)]

# for qrev in df.QREV.unique():
    
#     dat = df[(df.SA==sa)&(df.QSD==qsd)&(df.QREV==qrev)]
#     # dat = df[(df.SA==sa)&(df.QREV==qrev)]
    
#     color = cmap((qrev - df.QREV.min())/(df.QREV.max()-df.QREV.min()))
    
#     label = f"$Z$={round(qrev, 3)}"
#     x = dat.RMAX
#     y = dat.k90
    
#     ax.plot(x, y, color = color, alpha = 1,)
#     ax.scatter(x, y, color = color, alpha = 0.8, s = s, label = label)
    
#     ax.set_xlabel("$r_{max}$")
    
ax.legend()
ax.text(0.05, 0.95, "b", transform=ax.transAxes, ha='left', va='top', fontsize=12)

fig.text(0.005, 0.52, '$K_{10}$', va='center', rotation='vertical')

fig.set_size_inches(8, 4.3)
fig.tight_layout()