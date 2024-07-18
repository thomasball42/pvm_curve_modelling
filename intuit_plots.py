# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:50:48 2024

@author: tom
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import _models
import _population

from scipy.stats import gaussian_kde as kde

results_path = ""
save_path = os.path.join("..", "figs", "intuit_plots")

num_runs = 10000
# Ks = np.linspace(10, 50, num = 5)
Ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25]
# Ks = [50, 500, 5000]
num_years = 100

Q_space = [0.1]
Rmax_space = [0.1]
Sa_space = [0.35]
N0_space = [1]

# =============================================================================
# SETUP
# =============================================================================
years = np.arange(0, num_years, 1)
Ks = np.unique(np.round(Ks))

# set up some runs
runs = {
    "fixedN0_LGA": {
        "modelR": _models.Ri_model_I,
        "modelN" : _models.Ni_realnum_intuit,
        "num_runs": num_runs,
        "kwargs": {}
    },
#     "fixedN0_LGB": {
#         "modelR": _models.Ri_model_B,
#         "modelN" : _models.Ni_log,
#         "num_runs": num_runs,
#         "kwargs": {}
#     },
#     "fixedN0_LGC": {
#         "modelR": _models.Ri_model_C,
#         "modelN" : _models.Ni_log,
#         "num_runs": num_runs,
#         "kwargs": {}
#     },
    }
   
# check that the results path exists:
try:
    assert os.path.isdir(results_path)
except AssertionError:
    print("results_path must be a valid directory")
    
# =============================================================================
# Main
# =============================================================================
def gen_name(run_name, S, Sa, Rmax):
    if Sa == None:
        st = f"{run_name}_S{round(S, 3)}_Rmax{round(Rmax, 3)}_SA_nan"
    else:
        st = f"{run_name}_S{round(S, 3)}_Rmax{round(Rmax, 3)}_SA{round(Sa, 3)}"
    return st

odf = pd.DataFrame()
for run_name, run_params in runs.items():
    for Q in Q_space:
        for Rmax in Rmax_space:
            for N0 in N0_space:
                SA_ITERATOR = Sa_space if run_params["modelR"] == _models.Ri_model_C else [None]
                for Sa in SA_ITERATOR:
                    modelR = run_params["modelR"]
                    modelN = run_params["modelN"]
                    num_runs = run_params["num_runs"]
                    kwargs = run_params["kwargs"]
                    odf = pd.DataFrame()
                    if modelR == _models.Ri_model_C:
                        B = _models.getB(Rmax, Sa)
                        if B is None: 
                            pass
                    else:
                        B = None
                    run_label = gen_name(run_name, Q, Sa, Rmax)
                    q_pars = (0, Q)
                    for k, K in enumerate(Ks):
                        
                        print(f"{run_label}, {k} / {len(Ks)}")

                        for _ in range(num_runs):
                            sp = _population.Population(K, B, Rmax, Sa, K)

                            for y in years:
                                sp.iterate(modelR, modelN, _models.normal_dist(*q_pars), **kwargs)
                            
                            minN = np.min(sp.Nf)
                            
                            odf.loc[len(odf), ["runName", "K", "B", "Q", "Rmax", "N", "Sa", "minN"]] = [
                                run_label, K, B, Q, Rmax, 1, Sa, minN]
                            
                            
# %%
percs = (0, 90)
base_cmap = plt.get_cmap('viridis')
colors = base_cmap(np.linspace(0, 1, len(odf.K.unique())))

plot_range = np.percentile(odf.minN, percs)
plot_range = (0, 500)

bins = np.linspace(*plot_range, 500)
# bins = np.unique([round(b) for b in bins])
# bins = np.arange(plot_range[0], plot_range[1] + 1)

def pdf(x, y):
    return kde(x, weights = y)

for k, K in enumerate(odf.K.unique()):
    
    fig, ax = plt.subplots()
    
    color = colors[k]
    
    dat = odf[odf.K==K]
    
    minN = dat.minN
    
    counts, bin_edges = np.histogram(dat.minN, range=plot_range, bins=bins)
    
    curve = pdf(bin_edges[:-1], counts)(np.linspace(*plot_range, 100))
    
    bin_width = bin_edges[1] - bin_edges[0]
    bar_cont = ax.bar(bin_edges[:-1], counts, width=bin_width*1, align='edge', alpha=0.7, label=f"", color=color)
    
    # counts, _ = np.histogram(dat.minN, range = plot_range, bins = bins)
    
    # curve = pdf(bins[:-1], counts)(np.linspace(*plot_range, 100))
    
    
    # ax.hist(bins[:-1], bins, weights=counts, alpha = 0.9, label = f"K = {K}", color = color, align = "mid", edgecolor = None)
    # ax.plot(np.linspace(*plot_range, 100), num_runs * curve / curve.sum(), label = f"Prob density K = {K}",
    #         linewidth = 3, color = bar_cont.patches[0].get_facecolor(), alpha = 1)
    ax.axvline(2, color = "k", linestyle = "--")
    ax.set_ylabel(f"Counts ({num_runs} runs)")
    ax.set_xlabel("min N")
    fig.set_size_inches(8, 6)
    # fig.suptitle(f"", fontsize=14)
    ax.legend()
    
    fig.savefig(os.path.join(save_path, f"realnums_K{int(K)}.png"))
    
                        

