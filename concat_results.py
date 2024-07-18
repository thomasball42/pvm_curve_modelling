# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:48:50 2024

@author: tom
"""

import os
import pandas as pd

# od_path = "C:\\Users\\Thomas Ball\\OneDrive - University of Cambridge"
od_path = "E:\\OneDrive\\OneDrive - University of Cambridge"
results_path = os.path.join(od_path, "Work\\P_curve_shape\\dat\\results_log_ABC")
# =============================================================================
# Load data
# =============================================================================
f = []
for path, subdirs, files in os.walk(results_path):
    for name in files:
        f.append(os.path.join(path, name))
f = [file for file in f if ".csv" in file]
for m in ["A", "B", "C"]:
    df = pd.DataFrame()
    of = f"model{m}_data.csv"
    ff = [file for file in f if f"LogGrowth{m}" in file]
    for i, file in enumerate(ff[:]):
        dat = pd.read_csv(file, index_col = 0)
        df = pd.concat([df, dat])
    #     runName = dat.runName.unique().item()
    #     model = runName.split("_")[0]
    #     Q = dat.Q.unique().item()
    #     rmax = dat.Rmax.unique().item()
    #     try:
    #         Sa = dat.Sa.unique().item()
    #     except AttributeError:
    #         Sa = None
    #     B = dat.B.unique().item()
    #     x = dat.K
    #     y = dat.P
    #     df["K"] = x
    #     if Sa == None:
    #         st = f"Q{round(Q, 3)}_Rmax{round(rmax, 3)}_SA_nan"
    #     else:
    #         st = f"Q{round(Q, 3)}_Rmax{round(rmax, 3)}_SA{round(Sa, 3)}"
    #     df.loc[:, st] = y.values
    
    
    xdf = pd.DataFrame()
    
    d = {"Model" : [runName.split("_")[0].strip("LogGrowth") for runName in df.runName],
         "Rmax" : df.Rmax,
         "Q" : df.Q,
         "Sa": df.Sa,
         "K" : df.K,
         "P" : df.P}
         
         
    xdf = pd.DataFrame(d)
    xdf.to_csv(os.path.join(od_path, "Work\\P_curve_shape\\dat", of), index=False)