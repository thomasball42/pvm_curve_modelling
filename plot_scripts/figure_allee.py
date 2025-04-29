# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:42:56 2025

@author: Thomas Ball
"""

import pandas as pd
import numpy as np
import os 
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker

sys.path.append("..")
import _curve_fit

# my onedrive path, computer dependent..
# od_path = "C:\\Users\\Thomas Ball\\OneDrive - University of Cambridge"
# # od_path = "E:\\OneDrive\\OneDrive - University of Cambridge"

params = ["theta", "upsil"]
together = False

if together:
    fig, axs = plt.subplots(1, 2)

for p, param in enumerate(params):
    
    # dir that the simulation outputs are in
    dat_path = f"..\\results\\simulation_results\\results_ALLEE_{param}"

    # path to output fitted data
    data_fits_path = f"..\\results\\data_fits\\data_fits_ALLEE_{param}.csv"
    
    if together:
        ax = axs[p]
    else:
        fig, ax = plt.subplots()
        
    # =============================================================================
    # Find data
    # =============================================================================
    f = []
    for path, subdirs, files in os.walk(dat_path):
        for name in files:
            f.append(os.path.join(path, name))
    
    
    
    first_entry = True
    for i, file in enumerate(f[:]):
        
        
        
        if os.path.isfile(data_fits_path) and not first_entry:
            data_fits = pd.read_csv(data_fits_path, index_col=0)
        else:
            data_fits = pd.DataFrame()
            first_entry = False
            
        dat = pd.read_csv(file)
        runName = dat.runName.unique().item()
        
        ddf = pd.DataFrame()
    
    
        # get run info
        model = runName.split("_")[0]
        allee_theta = [_ for _ in runName.split("_") if "theta" in _][0].lower().strip("qwertyuiopasdfghjklzxcvbnm")
        allee_upsil = [_ for _ in runName.split("_") if "upsil" in _][0].lower().strip("qwertyuiopasdfghjklzxcvbnm") # old naming
        allee_upsil = float(allee_upsil[0] + "." + allee_upsil[1:])
        
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
                                    alpha_space = np.arange(-5, 0, 0.001),
                                    ylim=(0.05, 0.95), 
                                    plot_lins=False)
        if not fit and not ret == None:
            fit = True
            params, y_predicted, R2, resids = ret
            model_name = func.__name__
        
        
        
        # NOTE THAT kX is 1-X due to reframing of P_S(K) -> P_E(K)
        # calc kX, rsd, dPdK_max
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
        kX_vals = [get_kX2(X, *params) for X in kXs]
        kX_names = [f"k{int(X*100)}" for X in kXs]
        if not np.isnan(yff).all():
            dPdK = np.diff(yff) / np.diff(xff)
            dPdK_max = xff[np.argmax(dPdK) + 1]
        else:
            dPdK_max = np.nan
        
        if not ret == None:
            rsd = np.sqrt(np.sum(resids ** 2) / (len(resids) - 0))
        else:
            rsd = np.nan
         
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
            c = matplotlib.cm.get_cmap('viridis')(i/(len(f)-1))
            # c = matplotlib.cm.get_cmap('viridis')((R2 - 0.990)/(1-0.990))
            marker = "o"
            
        # label = f"{runName}_(R2:{round(R2, nnnn+1)})"
        
        
        label = f"$\\theta$: {allee_theta}, $\\upsilon$: {allee_upsil} (R2:{round(R2, nnnn+1)})"
        
        if fit:
            ax.scatter(x, 1 - y, color=c, alpha = 0.7, marker = marker)
            xff = np.geomspace(x.min(), x.max(), num = 100000)
            scatter_color = ax.collections[-1].get_facecolor()
            ax.plot(xff, 1 - func(xff, *params), color = scatter_color, label = label)
            
            ax.set_xscale("log", base = 2)
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=10))
            def custom_formatter(x, pos):
                return f'$10^{{{int(np.log10(x))}}}$'
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(custom_formatter))
            ax.set_ylabel(f"Probability of extinction (N={round(N)})")
            ax.set_xlabel("Carrying capacity K")
            
        ddf.loc[len(ddf), ["model", "runName", "RMAX", "QSD", "QREV", "B", "SA", 
                            "model_name", *param_names, "R2", "RSD", "MAX_Y", *kX_names, "dPdK_tp"]] = [
                            model, runName, rmax, qsd, qrev, B, sa, 
                            model_name, *params, R2, rsd, max_y, *kX_vals, dPdK_max]
        data_fits = pd.concat([data_fits, ddf])
        data_fits.to_csv(data_fits_path)                        
                                
        ax.legend()
    fig.set_size_inches(8, 5.5)
    fig.tight_layout()
                
