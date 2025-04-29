# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 15:08:18 2025

@author: Thomas Ball
"""

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# 
model_name = "C"
df_basic_path = f"..\\results\data_fits\\data_fits_{model_name}_basic_gompertz.csv"
df_mod_path = f"..\\results\\data_fits\\data_fits_{model_name}.csv"

df_basic = pd.read_csv(df_basic_path, index_col =0)
df_mod = pd.read_csv(df_mod_path, index_col = 0)

df = pd.DataFrame()

df["QSD"] = df_basic.QSD
df["RMAX"] = df_basic.RMAX
df["QREV"] = df_basic.QREV
df["SA"] = df_basic.SA

df["MAX_Y"] = df_basic.MAX_Y
df["basic_R2"] = df_basic.R2
df["mod_R2"] = df_mod.R2
df["inflection"] = df_basic.dPdK_tp
df["R2_diff"] = df.mod_R2 - df.basic_R2
df["R2_perc_diff"] = df.mod_R2 / df.basic_R2

MEAN_R2_diff = df.R2_diff.mean()
RMSE = np.sqrt( np.sum(df.R2_diff ** 2) / len(df))

fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [0.8, 1.8]}, 
                        sharey=True)
# fig, axs = plt.subplots()
df = df[df.MAX_Y > 0.9999]

dat = df.R2_diff
dat = dat[~np.isnan(dat)]

axs[0].boxplot(dat)

# axs[0].hist(dat, bins = 200)

axs[0].set_xticks([])
axs[1].scatter(df.inflection, df.R2_diff, color = "k", alpha = 0.4, label = f"Model {model_name}")
axs[0].set_ylabel("Absolute difference in R2")
axs[1].set_xlabel("Inflection point (K)")
axs[1].legend()
fig.tight_layout()

