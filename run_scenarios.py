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

import matplotlib.pyplot as plt
import _curve_fit
import matplotlib.ticker

# =============================================================================
# PARAMS
# =============================================================================

results_path = "/maps/tsb42/pvm_curve/results_C_0_09"

num_runs = 10000
Ks = np.geomspace(1, 20000, num = 150)
num_years = 100

S_space = np.arange(0, 1, 0.05)
Rmax_space = np.arange(0, 1, 0.05)
Sa_space = np.arange(0.4, 0.9, 0.1)

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
    "LogGrowthB_PoissonDraw": {
        "modelR": _models.Ri_model_LogisticGrowthB,
        "modelN" : _models.Ni_log_poisson,
        "num_runs": num_runs,
        "kwargs": {}
    },
    "LogGrowthC_PoissonDraw": {
        "modelR": _models.Ri_model_LogisticGrowthC,
        "modelN" : _models.Ni_log_poisson,
        "num_runs": num_runs,
        "kwargs": {}
    },
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

# Loop through run parameters
for run_name, run_params in runs.items(): 
    modelR = run_params["modelR"]
    modelN = run_params["modelN"]
    num_runs = run_params["num_runs"]
    kwargs = run_params["kwargs"]
    
    for s, S in enumerate(S_space):
        
        for r, Rmax in enumerate(Rmax_space):
            
            if modelR ==_models.Ri_model_LogisticGrowthC:
                SA_ITERATOR = Sa_space
            else: SA_ITERATOR = [None]
            
            for sa, Sa in enumerate(SA_ITERATOR):
                odf = pd.DataFrame()
                if modelR ==_models.Ri_model_LogisticGrowthC:
                    B = _models.getB(Rmax, Sa)
                    if B == None: continue
                else: B = None
                run_label = gen_name(run_name, S, Sa, Rmax)
                q_pars = (0, S)
                
                for k, K in enumerate(Ks):
                    print(f"{run_label}, {k} / {len(Ks)}")
                    extinctions = 0
                    for _ in range(num_runs):
                        sp = _population.Population(K, B, Rmax, Sa)
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
                    odf.loc[len(odf), ["runName", "K", "B", "S", "Rmax", "N", "P", "Sa"]] = [
                        run_label, K, B, S, Rmax, num_runs, P, Sa]
                odf.to_csv(os.path.join(results_path, run_label + ".csv"))