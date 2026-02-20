# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 13:37:41 2025

@author: tom

This script is a bit of a mess. Aim to tidy at some point.

"""

import os
import pandas as pd
import numpy as np
import sys 

import matplotlib.pyplot as plt
import matplotlib.ticker

sys.path.append("..")
import _curve_fit

scale_1_0 = False
plot_pspace = False
plot_curves = False

# dir that the simulation outputs are in
results_path = "..\\results\\simulation_results\\results_main"

for mm in ["D"]:
    # path to output fitted data
    data_fits_path = f"..\\results\\data_fits\\data_fits_{mm}_basic_gompertz.csv"
    
    
    # =============================================================================
    # Find data
    # =============================================================================
    f = []
    for path, subdirs, files in os.walk(results_path):
        for name in files:
            f.append(os.path.join(path, name))
    f = [file for file in f if ".csv" in file and f"LogGrowth{mm}" in file]
  
    #%%
    n = int(plot_curves)+int(plot_pspace)
    if n > 0:
        fig, axs = plt.subplots(1, n)
    
    first_entry = True
    for i, file in enumerate(f[:]):
        
        print(mm, i / len(f))
        if os.path.isfile(data_fits_path) and not first_entry:
            data_fits = pd.read_csv(data_fits_path, index_col=0)
        else:
            data_fits = pd.DataFrame()
            first_entry = False
            
        dat = pd.read_csv(file)
        runName = dat.runName.unique().item()
        
        ## SETUP
        ddf = pd.DataFrame()
    
        model = runName.split("_")[0]
        qsd = dat.QSD.unique().item()
        N = dat.N.unique().item()
        
        try:
            qrev = dat.QREV.unique().item()
            if "LogGrowthD" not in model:
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
            
        func = _curve_fit.basic_gomp
        param_names = ("bg_param_a", "bg_param_b")
        params = tuple([np.nan for _ in param_names])
        try:    
            ret = _curve_fit.fit(func, x, y)
        except RuntimeError:
            pass
        
        if not fit and not ret == None:
            
            fit = True
            params, y_predicted, R2, resids = ret
            model_name = func.__name__
        
        # NOTE THAT kX is 1-X due to reframing of P_S(K) -> P_E(K)
        # calc kX, rsd, dPdK_max
        xff = np.geomspace(dat.K.min(), dat.K.max(), num = 100000)
        yff = func(xff, *params)
        kXs = np.arange(0.1, 1.0, 0.1)
        
        
        def get_kX2(X, a, b, alpha = 1):
            """analytical"""
            if np.isnan(a):
                kX = np.nan
            else: kX = ((np.log( -np.log(X)) - a) / b ) ** (1/alpha)
            return kX
        
        kX_vals = [get_kX2(X, *params) for X in kXs]
        kX_names = [f"k{int(X*100)}" for X in kXs]
        
        if not np.isnan(yff).all():
            dPdK = np.diff(yff) / np.diff(xff)
            dPdK_max = xff[np.argmax(dPdK) + 1]
        else:
            dPdK_max = np.nan
        
        if not ret == None:
            rsd = np.sqrt(np.sum(resids ** 2) / (len(resids) - 0))
            rmse = np.sqrt(np.sum(resids ** 2) / (len(resids) - len(params) + 1))
        else:
            rsd = np.nan
            rmse = np.nan
        
        ddf.loc[len(ddf), ["model", "runName", "RMAX", "QSD", "QREV", "B", "SA", 
                            "model_name", *param_names, "R2", "RSD", "RMSE", "MAX_Y", *kX_names, "dPdK_tp"]] = [
                            model, runName, rmax, qsd, qrev, B, sa, 
                            model_name, *params, R2, rsd, rmse, max_y, *kX_vals, dPdK_max]
                                
        # # PLOT CURVES AND FITS
        if plot_curves and max_y == 1:
            if n>1:
                ax = axs[-1]
            else:
                ax = axs
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
                
            label = f"{runName}_(R2:{round(R2, nnnn+1)})"
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
            ax.set_ylabel(f"Probability of extinction (N={round(N)})")
            ax.set_xlabel("Carrying capacity K")
            fig.set_size_inches(12, 7)
            fig.tight_layout()
            
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
        
        print(file)

        data_fits = pd.concat([data_fits, ddf])
        data_fits.to_csv(data_fits_path)

    if plot_pspace:
        if n>1:
            ax = axs[0]
        norm = plt.Normalize(0.990, 1)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('R2')
    
        # ax.legend()
        fig.set_size_inches(12, 7)
        fig.tight_layout()
    
    
    
