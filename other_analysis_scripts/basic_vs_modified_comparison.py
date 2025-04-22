# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 15:08:18 2025

@author: Thomas Ball
"""

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt



# my onedrive path, computer dependent..
od_path = "C:\\Users\\Thomas Ball\\OneDrive - University of Cambridge"
# od_path = "E:\\OneDrive\\OneDrive - University of Cambridge"

# 
df_basic_path = os.path.join(od_path, "Work\\P_curve_shape\\version2\\dat\\data_fits_A_basic_gompertz.csv")
df_mod_path = os.path.join(od_path, "Work\\P_curve_shape\\version2\\data\\data_fits_A.csv")

df_basic = pd.read_csv(df_basic_path, index_col =0)
df_mod = pd.read_csv(df_mod_path, index_col = 0)

df = pd.DataFrame()

df["QSD"] = df_basic.QSD
df["RMAX"] = df_basic.RMAX
df["MAX_Y"] = df_basic.MAX_Y
df["basic_R2"] = df_basic.R2
df["mod_R2"] = df_mod.R2
df["inflection"] = df_basic.dPdK_tp

df["R2_diff"] = df.mod_R2 - df.basic_R2
df["R2_perc_diff"] = df.mod_R2 / df.basic_R2

MEAN_R2_diff = df.R2_diff.mean()
RMSE = np.sqrt( np.sum(df.R2_diff ** 2) / len(df))

fig, axs = plt.subplots(1, 3)

dat = df.R2_diff

dat = dat[df.MAX_Y == 1]
dat = dat[~np.isnan(dat)]

# ax.boxplot(dat)

# # ax.hist(dat, bins = 200)


axs[1].scatter(df.inflection, df.R2_diff)
axs[1].set_ylabel("Absolute difference in R2")
# ax.set_xlabel("Inflection point (K)")

fig.tight_layout()

