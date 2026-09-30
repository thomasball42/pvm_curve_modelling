# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:36:27 2024

@author: Thomas Ball
"""

import os
import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.ticker

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import _curve_fit
import _analysis_utils as utils
from _figure_io import (AXIS_LABEL_SIZE, PANEL_LABEL_SIZE, RESULTS_DIR,
                        save_figure, use_main_fig_style)

SIM_RESULTS_DIR = RESULTS_DIR / "simulation_results"

plot_curves = True
FIG_SUBDIR = "nonK"

# main-text figure: width is capped at 210 mm by save_figure()
MAIN_FIG = True
use_main_fig_style()


fig, axs = plt.subplots(1, 2)

clip_x = True

for r, rpath in enumerate(["results_propN0", 
                           "results_fixedN0", ]):
    ax = axs[r]
        
    results_path = SIM_RESULTS_DIR / rpath
    data_fits_dir = RESULTS_DIR / 'data_fits' / 'data_fits_nonK'
    data_fits_dir.mkdir(parents=True, exist_ok=True)
    data_fits_path = data_fits_dir / f"data_fits_{rpath.split('_')[-1]}.csv"
    
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
        f = sorted(f, key=lambda x: float(x.split('N0')[-1].split('.csv')[0]))

    # start fresh each run so re-running overwrites rather than appends
    ddf = pd.DataFrame()

    for i, file in enumerate(f[:]):

        dat = pd.read_csv(file)
        runName = dat.runName.unique().item()

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
        min_y = y.min()
            
        try:
            sa = dat.SA.unique().item()
        except AttributeError:
            sa = None
        
        # TRY GOMPERTZ - same fitter/settings as the main analysis
        func = _curve_fit.mod_gompertz
        param_names = ("param_a", "param_b", "param_alpha")
        # main-analysis fitter, but with the fitting window these data support
        # and enough iterations for the large-N0 runs to converge
        fit_result = utils.fit_gompertz_curve(x, y, ylim=(0.05, 0.95),
                                             maxfev=100000)
        params = fit_result["params"]
        R2 = fit_result["R2"]
        resids = fit_result["resids"]
        rsd = fit_result["rsd"]
        alpha_ci = fit_result["alpha_ci"]
        model_name = fit_result["model_name"]

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
        
        ddf.loc[len(ddf), ["model", "runName", "RMAX", "QSD", "QREV", "B", "SA", 
                            "model_name", *param_names, "alpha_ci_5", "alpha_ci_95", "R2", "RSD", "MAX_Y", *kX_names, "dPdK_tp"]] = [
                            model, runName, rmax, qsd, qrev, B, sa,
                            model_name, *params, *alpha_ci, R2, rsd, max_y, *kX_vals, dPdK_max]
         
        ddf.to_csv(data_fits_path)
        
        if max_y < 1 or min_y > 0:
            print(min_y, max_y, file)
            continue 
        
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
                c = plt.get_cmap('viridis')((R2 - 0.990)/(1-0.990))
                marker = "o"
     
            c = plt.get_cmap("viridis")((i+0.9)/(len(f)))
            
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
                    val = float(runName.split("_")[-1].split("N0")[-1])
                    # sn = f"$10^{int(math.log10(abs(val)))}$"
                    # label = f"$N_0$={sn}; $r^2$: {round(R2, nnnn+1)}"
                    label = f"$N_0$=$2^{{{int(np.log2(val))}}}$"

            # label += f"; $\\alpha$={params[-1]:.2f} [{alpha_ci[0]:.2f}, {alpha_ci[1]:.2f}]"
            label += f" [fitted $\\gamma$={params[-1]:.3f}]"

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
    
    ax.legend()

if clip_x:
    axs[0].set_xlim(1E1, 0.2E3)
    # axs[1].set_xlim(1E1, 0.2E3)
axs[0].text(0.05, 0.95, "a", transform=axs[0].transAxes, ha='left', va='top',
            fontsize=PANEL_LABEL_SIZE, fontweight="bold")
axs[1].text(0.05, 0.95, "b", transform=axs[1].transAxes, ha='left', va='top',
            fontsize=PANEL_LABEL_SIZE, fontweight="bold")
axs[0].set_ylabel(f"Probability of extinction $P_E$")

fig.text(0.45, 0.019, 'Carrying capacity $K$', va='center', rotation='horizontal',
         fontsize=AXIS_LABEL_SIZE)

fig.set_size_inches(8, 4.3)
fig.tight_layout()

fig.show()
save_figure(fig, "figure_nonK", subdir=FIG_SUBDIR, main_fig=MAIN_FIG)

