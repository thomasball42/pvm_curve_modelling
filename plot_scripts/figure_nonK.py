# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:36:27 2024

@author: Thomas Ball
"""

import os
import pandas as pd
import numpy as np
import scipy.stats
import math

import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.stats import gaussian_kde as kde
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

import _curve_fit

plot_curves = True

# od_path = "C:\\Users\\Thomas Ball\\OneDrive - University of Cambridge"
od_path = "E:\\OneDrive\\OneDrive - University of Cambridge"

fig, axs = plt.subplots(1, 2)

for r, rpath in enumerate(["results_propN0", 
                           "results_fixedN0", ]):
    ax = axs[r]
        
    results_path = os.path.join(od_path, f"Work\\P_curve_shape\\dat\\{rpath}")
    # =============================================================================
    # Load data
    # =============================================================================
    f = []
    for path, subdirs, files in os.walk(results_path):
        for name in files:
            f.append(os.path.join(path, name))
    f = [file for file in f if ".csv" in file]
    f = [k for k in f if "LGA" in k]
    if r == 1:
        f = [f[i] for i in [5, 0, 1, 2, 3, 4]]

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
                c = matplotlib.cm.get_cmap('viridis')((R2 - 0.990)/(1-0.990))
                marker = "o"
     
            c = matplotlib.cm.get_cmap("viridis")((i+0.9)/(len(f)))
            
            mod = model.strip("LogGrowth").strip("2")
            

            if "over" in runName:
                fac = runName.split("over")[-1]
                prop = round(1/float(fac), 3)
                
                # label = f"$N_0$={prop}$K$; $r^2$: {round(R2, nnnn+1)}"
                
                label = f"$N_0$={prop}$K$"
                
                if prop == 1:
                    label = f"$N_0$=$K$"
                    
            elif "fixed" in runName:
                if "over" in runName:
                    label = f"$N_0$=K; $r^2$: {round(R2, nnnn+1)}"
                else:
                    val = float(runName.split("_")[-1])
                    sn = f"$10^{int(math.log10(abs(val)))}$"
                    # label = f"$N_0$={sn}; $r^2$: {round(R2, nnnn+1)}"
                    label = f"$N_0$={sn}"
                
            ax.scatter(x, 1 - y, color=c, alpha = 0.9, marker = marker, label = label)
            xff = np.geomspace(x.min(), x.max(), num = 100000)
            scatter_color = ax.collections[-1].get_facecolor()
            ax.plot(xff, 1 - func(xff, *params), color = scatter_color, )
            
            ax.set_xscale("log", base = 2)
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=10))
            def custom_formatter(x, pos):
                return f'$10^{{{int(np.log10(x))}}}$'
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(custom_formatter))
    
    ax.legend(fontsize=11)
    
axs[0].text(0.05, 0.95, "a", transform=axs[0].transAxes, ha='left', va='top', fontsize=12)
axs[1].text(0.05, 0.95, "b", transform=axs[1].transAxes, ha='left', va='top', fontsize=12)
axs[0].set_ylabel(f"Probability of extinction $P_E$")

fig.text(0.45, 0.019, 'Carrying capacity $K$', va='center', rotation='horizontal')

fig.set_size_inches(8, 4.3)
fig.tight_layout()



