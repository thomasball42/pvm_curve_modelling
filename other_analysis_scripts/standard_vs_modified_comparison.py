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
model_name = "A"
df_basic_path = f"..\\results\data_fits\\data_fits_{model_name}_basic_gompertz.csv"
df_mod_path = f"..\\results\\data_fits\\data_fits_{model_name}.csv"

df_basic = pd.read_csv(df_basic_path, index_col =0)
df_mod = pd.read_csv(df_mod_path, index_col = 0)

df = pd.DataFrame()

var = "R2"

mod_min = df_mod.R2.min()

basic_outside_range = df_basic[df_basic.R2 < mod_min]
perc_outside_range = len(basic_outside_range) / len(df_basic)

df["QSD"] = df_basic.QSD
df["RMAX"] = df_basic.RMAX
df["QREV"] = df_basic.QREV
df["SA"] = df_basic.SA

df["MAX_Y"] = df_basic.MAX_Y
df["inflection"] = df_basic.dPdK_tp
df["k10"] = df_basic.k90

df[f"basic_{var}"] = df_basic[var]
df[f"mod_{var}"] = df_mod[var]
df[f"{var}_diff"] = df[f"mod_{var}"] - df[f"basic_{var}"]
df[f"{var}_perc_diff"] = df[f"mod_{var}"] / df[f"basic_{var}"]

# print(model_name, df_basic[df_basic.MAX_Y == 1].R2.min())

MEAN_var_diff = df[f"{var}_diff"].mean()

fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [0.8, 1.8]}, 
                        sharey=True)
# fig, axs = plt.subplots()
df = df[df.MAX_Y > 0.9999]

dat = df[f"{var}_diff"]
dat = dat[~np.isnan(dat)]

dat2 = df.inflection
# dat2 = df.k10
axs[0].boxplot(dat)

# axs[0].hist(dat, bins = 200)

axs[0].set_xticks([])
axs[1].scatter(dat2, df[f"{var}_diff"], color = "k", alpha = 0.4, label = f"Model {model_name}")
axs[0].set_ylabel(f"Absolute difference in {var}")
axs[1].set_xlabel("Inflection point (K)")
# axs[1].set_xlabel("$K_{10}$")
axs[1].legend()
fig.tight_layout()

