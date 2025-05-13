# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:47:34 2024

@author: Thomas Ball
"""

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker

import _curve_fit

scale_1_0 = False
plot_pspace = False
plot_curves = True

results_path = "..\\results\\simulation_results\\results_main"

sims_to_plot = [   
     "LogGrowthA_QSD0.11_QREV1.0_RMAX0.158_SAnan_N00.csv",
     "LogGrowthB_QSD0.11_QREV1.0_RMAX0.158_SAnan_N00.csv",
     "LogGrowthC2_QSD0.11_QREVnan_RMAX0.158_SA0.35_N00.csv",
     "LogGrowthD2_QSD0.11_QREV0.258_RMAX0.158_SA0.35_N00.csv",
    ]

# =============================================================================
# Find data
# =============================================================================
f = []
for path, subdirs, files in os.walk(results_path):
    for name in files:
        f.append(os.path.join(path, name))
f = [file for file in f if os.path.split(file)[-1] in sims_to_plot]

fig, ax = plt.subplots()

for i, file in enumerate(f[:]):
    
    dat = pd.read_csv(file)
    runName = dat.runName.unique().item()
    
    ## SETUP
    ddf = pd.DataFrame()

    model = runName.split("_")[0]
    qsd = dat.QSD.unique().item()
    N = dat.N.unique().item()
    
    try:
        qrev = dat.QREV.unique().item()
        if not model == "LogGrowthD2":
            qrev = np.nan
    except AttributeError:
        qrev = np.nan
        
    rmax = dat.RMAX.unique().item()
    B = dat.B.unique().item()
    x = dat.K 
    y = dat.P
    
    max_y = y.max()
    
    try:
        sa = dat.SA.unique().item()
    except AttributeError:
        sa = None
        
    if scale_1_0:
        reversed_arr = y[::-1]
        first_non_one_index = np.argmax(reversed_arr != 1)
        tail_start_index = len(y) - first_non_one_index
        y = y[:tail_start_index]
        if len(x) == tail_start_index:
            x = x/ x.max()
        else:
            x = x[:tail_start_index] / x[tail_start_index]
    
    # initialise fitting
    fit = False
    model_name = np.nan
    R2 = np.nan
    resids = np.nan
        
    # TRY GOMPERTZ
    func = _curve_fit.mod_gompertz
    param_names = ("param_a", "param_b", "param_alpha")
    params = tuple([np.nan for _ in param_names])
    ret = _curve_fit.betterfit_gompertz(func, x, y, 
                            alpha_space = np.arange(0, 5, 0.001), 
                            ylim=(0.05, 0.95), 
                            plot_lins=False,)
    if ret == None:
        ret = _curve_fit.betterfit_gompertz(func, x, y, 
                                alpha_space = np.arange(-3, 0, 0.001),
                                ylim=(0.05, 0.95), 
                                plot_lins=False)
    if not fit and not ret == None:
        fit = True
        params, y_predicted, R2, resids = ret
        model_name = func.__name__
           

    # calc k50, rsd, dPdK_max
    xff = np.geomspace(dat.K.min(), dat.K.max(), num = 100000)
    yff = func(xff, *params)
    kXs = np.arange(0.1, 1.0, 0.1)
    def get_kX(X, xff, yff):
        gtX = xff[yff >= X]
        if len(gtX) > 0: kX = gtX[0]
        else: kX = np.nan
        return kX
    def get_kX2(X, a, b, alpha):
        """analytical"""
        if np.isnan(a):
            kX = np.nan
        else: kX = ((np.log( -np.log(X)) - a) / b ) ** (1/alpha)
        return kX
    kX_vals = [get_kX(X, xff, yff) for X in kXs]
    kX2_vals = [get_kX2(X, *params) for X in kXs]
    kX_names = [f"k{int(X*100)}" for X in kXs]
    kX_diff = np.array([kX_vals[i] - kX2_vals[i] for i in range(len(kX_vals))])
    kX_diff_sd = np.sqrt((kX_diff**2).sum() / len(kX_diff))
    
    if not np.isnan(yff).all():
        dPdK = np.diff(yff) / np.diff(xff)
        dPdK_max = xff[np.argmax(dPdK) + 1]
    else:
        dPdK_max = np.nan
    
    if not ret == None:
        rsd = np.sqrt(np.sum(resids ** 2) / (len(resids) - 0))
    else:
        rsd = np.nan
    
    ddf.loc[len(ddf), ["model", "runName", "RMAX", "QSD", "QREV", "B", "SA", 
                        "model_name", *param_names, "R2", "RSD", "MAX_Y", *kX_names, "dPdK_tp"]] = [
                        model, runName, rmax, qsd, qrev, B, sa, 
                        model_name, *params, R2, rsd, max_y, *kX_vals, dPdK_max]
                            
    print(R2)
    # # PLOT CURVES AND FITS
    if plot_curves:
        label = f"Model {model.strip('LogGrowth')}"
        nnnn = 0 #batman
        while round(R2, nnnn) == 1:
            nnnn += 1
        
        if np.isnan(R2):
            c = "r"
            marker = "x"
        elif params[-1] < 0:
            c = "m"
            marker = "o"
        else:
            c = matplotlib.colormaps['viridis']((R2 - 0.990)/(1-0.990))
            marker = "o"
        mmm = {"LogGrowthA":0.05,
               "LogGrowthB":0.33,
               "LogGrowthC2":0.66,
               "LogGrowthD2":0.95}
        
        
        c = matplotlib.cm.get_cmap("viridis")(mmm[model])
        
        mod = model.strip("LogGrowth").strip("2")
        
        # label = f"{runName}_(R2:{round(R2, nnnn+1)})"
        if mod == "A" or mod == "B":
            label = f"Model {mod}; $r_{{max}}$=0.158; $\\sigma$=0.11"
        elif mod == "C":
            label = f"Model {mod}; $r_{{max}}$=0.158; $\\sigma$=0.11; $S_a$=0.35"
        elif mod == "D":
            label = f"Model {mod}; $r_{{max}}$=0.158; $\\sigma$=0.11; $S_a$=0.35; $Z$=0.258"
        
        ax.scatter(x, 1 - y, color=c, alpha = 0.7, marker = marker)
        xff = np.geomspace(x.min(), x.max(), num = 100000)
        scatter_color = ax.collections[-1].get_facecolor()
        ax.plot(xff, 1 - func(xff, *params), color = scatter_color, label = label)
        if not scale_1_0:
            ax.set_xscale("log", base = 2)
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=10))
            def custom_formatter(x, pos):
                return f'$10^{{{int(np.log10(x))}}}$'
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(custom_formatter))
        ax.set_ylabel(f"Probability of extinction $P_E$")
        ax.set_xlabel("Carrying capacity $K$")
        
if plot_pspace:
    norm = plt.Normalize(0.990, 1)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('R2')

ax.legend(fontsize = 9)
fig.set_size_inches(8, 4)
fig.tight_layout()



