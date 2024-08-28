# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:55:22 2024

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
plot_curves = False

all_out = True

R2_cutoff = 0.99
od_path = "C:\\Users\\Thomas Ball\\OneDrive - University of Cambridge"
# od_path = "E:\\OneDrive\\OneDrive - University of Cambridge"

results_path = os.path.join(od_path, "Work\\P_curve_shape\\dat\\results2_ABCD")
data_fits_path = os.path.join(od_path, "Work\\P_curve_shape\\dat\\test.csv")

# =============================================================================
# Load data
# =============================================================================
f = []
for path, subdirs, files in os.walk(results_path):
    for name in files:
        f.append(os.path.join(path, name))
f = [file for file in f if ".csv" in file]
f = [k for k in f if "LogGrowthA" in k]

#%%
n = int(plot_curves)+int(plot_pspace)
if n == 0:
    n = 1
fig, axs = plt.subplots(1, n)

fx = []
fy = []

for i, file in enumerate(f[:]):
    
    data_fits = pd.DataFrame()
        
    dat = pd.read_csv(file)
    runName = dat.runName.unique().item()
    
    ## SETUP
    ddf = pd.DataFrame()

    model = runName.split("_")[0]
    qsd = dat.QSD.unique().item()
    N = dat.N.unique().item()
    qrev = dat.QREV.unique().item()
    if not model == "LogGrowthD":
        qrev = np.nan
    rmax = dat.RMAX.unique().item()
    B = dat.B.unique().item()
    x = dat.K 
    y = dat.P
    
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
    
    # initialise
    fit = False
    model_name = np.nan
    R2 = np.nan
        
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
        if ret[-2] > R2_cutoff:
            params, y_predicted, R2, resids = ret
            model_name = func.__name__
        # else:
            
        
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
    
    rsd = np.sqrt(np.sum(resids ** 2) / (len(resids) - len(params)))
    
    ddf.loc[len(ddf), ["model", "runName", "RMAX", "QSD", "QREV", "B", "SA", 
                        "model_name", *param_names, "R2", "RSD", *kX_names, "dPdK_tp"]] = [
                        model, runName, rmax, qsd, qrev, B, sa, 
                        model_name, *params, R2, rsd, *kX_vals, dPdK_max]
                            
    print(R2)
    # # PLOT CURVES AND FITS
    if plot_curves:
        if n>1:
            ax = axs[-1]
        else:
            ax = axs
        label = f"Model {model.strip('LogGrowth')}"
        nnnn = 0
        while round(R2, nnnn) == 1:
            nnnn += 1
        
        if np.isnan(R2):
            c = "r"
            marker = "x"
        elif params[-1] < 0:
            c = "m"
            marker = "o"
        else:
            c = matplotlib.cm.get_cmap('viridis')((R2 - 0.990)/(1-0.990))
            marker = "o"
            
        label = f"{runName}_(R2:{round(R2, nnnn+1)})"
        fx.append(x)
        fy.append(y)
        ax.scatter(x, 1 - y, c=c, alpha = 0.7, marker = marker)
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
        ax.set_ylabel(f"Probability of persistence (N={round(N)})")
        ax.set_xlabel("Carrying capacity K")
        
    if plot_pspace:
        if n>1:
            ax = axs[0]
        else:
            ax = axs
        if np.isnan(R2):
            c = "r"
            marker = "x"
        elif params[-1] < 0:
            c = "m"
            marker = "o"
        else:
            c = matplotlib.cm.get_cmap('viridis')((R2 - 0.990)/(1-0.990))
            marker = "o"
        ax.scatter(qsd, rmax, color = c, s = 70, marker = marker)
        ax.set_ylabel("R$_{max}$")
        ax.set_xlabel("Stochasicity SD")
        
    data_fits = pd.concat([data_fits, ddf])
    data_fits.to_csv(data_fits_path)

if plot_pspace:
    if n>1:
        ax = axs[0]
    norm = plt.Normalize(0.990, 1)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('R2')
    
fig.set_size_inches(12, 7)
fig.tight_layout()



