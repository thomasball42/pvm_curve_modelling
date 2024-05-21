# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:21:29 2024

@author: tom
"""

import numpy as np
import pandas as pd
import os

import _models
import _population


# =============================================================================
# PARAMS
# =============================================================================

results_path = "E:\\OneDrive\\OneDrive - University of Cambridge\\Work\\P_curve_shape\\results"
num_runs = 10000
Ks = np.geomspace(1, 20000, num = 150)
num_years = 100

Sa_space = np.arange(0, 1, 0.05)
Rmax_space = np.arange(0, 1, 0.05)

# =============================================================================
# SETUP
# =============================================================================
years = np.arange(0, num_years, 1)
Ks = np.unique(np.round(Ks))

# set up some runs
runs = {
    "LogGrowthA_PoissonDraw": {
        "modelR": _models.Ri_model_LogisticGrowthA,
        "modelN" : _models.Ni_log_poisson,
        "num_runs": num_runs,
        "kwargs": {}
    },
    # "LogGrowthB_PoissonDraw": {
    #     "modelR": _models.Ri_model_LogisticGrowthB,
    #     "modelN" : _models.Ni_log_poisson,
    #     "num_runs": num_runs,
    #     "kwargs": {}
    # },
    # "LogGrowthC_PoissonDraw": {
    #     "modelR": _models.Ri_model_LogisticGrowthC,
    #     "modelN" : _models.Ni_log_poisson,
    #     "num_runs": num_runs,
    #     "kwargs": {}
    # },
}
    
# =============================================================================
# Main
# =============================================================================
def gen_name(run_name, Sa, Rmax):
    return f"{run_name}_Sa{round(Sa, 3)}_Rmax{round(Rmax, 3)}"
# Loop through run parameters
for run_name, run_params in runs.items(): 
    modelR = run_params["modelR"]
    modelN = run_params["modelN"]
    num_runs = run_params["num_runs"]
    kwargs = run_params["kwargs"]
    for s, Sa in enumerate(Sa_space):
        for r, Rmax in enumerate(Rmax_space):
            if modelR ==_models.Ri_model_LogisticGrowthC:
                B = _models.getB(Rmax, Sa)
                if np.isinf(B):
                    continue
            else: B = None
            odf = pd.DataFrame()
            run_label = gen_name(run_name, Sa, Rmax)
            q_pars = (0, Sa)
            for k, K in enumerate(Ks):
                extinctions = 0
                for _ in range(num_runs):
                    sp = _population.Population(K, B, Rmax)
                    for y in years:
                        sp.iterate(modelR, modelN, _models.normal_dist(*q_pars), **kwargs)
                        # record extinction and stop run
                        if sp.EXTANT == False:
                            extinctions += 1
                            break
                        # For cases where something goes wrong
                        if sp.RUNABORT == True:
                            break
                P = 1 - extinctions / num_runs
                # Store results
                odf.loc[len(odf), ["runName", "K", "B", "Sa", "Rmax", "N", "P"]] = [
                    run_label, K, B, Sa, Rmax, num_runs, P]
            odf.to_csv(os.path.join(results_path, run_label + ".csv"))
            