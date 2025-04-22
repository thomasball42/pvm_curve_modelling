# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 17:27:04 2025

@author: tom
"""

import pandas as pd
import numpy as np
import os 
import sys
import math

import matplotlib.pyplot as plt
import matplotlib.ticker

# my onedrive path, computer dependent..
# od_path = "C:\\Users\\Thomas Ball\\OneDrive - University of Cambridge"
od_path = "E:\\OneDrive\\OneDrive - University of Cambridge"

# dir that the simulation outputs are in
dat_fits_path = os.path.join(od_path, "Work\\P_curve_shape\\version2\\data")

models = ["A", "B", "C", "D"]

f = []
for path, subdirs, files in os.walk(dat_fits_path):
    for name in files:
        f.append(os.path.join(path, name))

fig, axs = plt.subplots()

for m, model in enumerate(models):
    
    files = [_ for _ in f if os.path.split(_)[-1] == f"data_fits_{model}.csv"]
    
    if len(files) != 1:
        quit("Found the wrong number of matching files (either 0 or >1)...")
        
    else: file = files[0]
    
    dat = pd.read_csv(file, index_col = 0)
    
    g = dat.param_alpha
    
    g = g[(~np.isnan(g)) & (dat.MAX_Y == 1)]
    
    # axs.boxplot(g, positions = [m])
    
    mean = g.mean()
    stde = 3*np.std(g) / np.sqrt(len(g))
    axs.scatter(m, mean, marker = "x", color = "k")
    axs.errorbar(m, mean, yerr = stde, color = "k")
    
axs.set_ylim(-0.5, 1.25)
axs.set_xticks(np.arange(len(models)), labels = [f"Model_{x}" for x in models])
axs.axhline(1, linestyle = "--", color = "k", alpha = 0.5)
axs.set_ylabel("Gamma parameter")
fig.tight_layout()