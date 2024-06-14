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

results_path = "/maps/tsb42/pvm_curve/results_log_ABC"

num_runs = 10000
Ks = np.geomspace(1, 30000, num = 200)
num_years = 100

Q_space = np.arange(0.1, 0.55, 0.05)
Rmax_space = [0.265, 0.373, 0.447, 0.509, 0.56, 0.619, 0.644, 0.71, 0.774]
Sa_space = np.arange(0.35, 0.95, 0.05)

# =============================================================================
# SETUP
# =============================================================================
years = np.arange(0, num_years, 1)
Ks = np.unique(np.round(Ks))

# set up some runs
runs = {
    "LogGrowthA": {
        "modelR": _models.Ri_model_GompertzGrowthA,
        "modelN" : _models.Ni_log,
        "num_runs": num_runs,
        "kwargs": {}
    },
    "LogGrowthB": {
        "modelR": _models.Ri_model_GompertzGrowthB,
        "modelN" : _models.Ni_log,
        "num_runs": num_runs,
        "kwargs": {}
    },
    "LogGrowthC": {
        "modelR": _models.Ri_model_GompertzGrowthC,
        "modelN" : _models.Ni_log,
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
        st = f"{run_name}_Q{round(S, 3)}_Rmax{round(Rmax, 3)}_SA_nan"
    else:
        st = f"{run_name}_Q{round(S, 3)}_Rmax{round(Rmax, 3)}_SA{round(Sa, 3)}"
    return st

# Loop through run parameters
for run_name, run_params in runs.items(): 
    modelR = run_params["modelR"]
    modelN = run_params["modelN"]
    num_runs = run_params["num_runs"]
    kwargs = run_params["kwargs"]
    
    for q, Q in enumerate(Q_space):
        
        for r, Rmax in enumerate(Rmax_space):
            
            if modelR ==_models.Ri_model_GompertzGrowthC:
                SA_ITERATOR = Sa_space
            else: SA_ITERATOR = [None]
            
            for sa, Sa in enumerate(SA_ITERATOR):
                odf = pd.DataFrame()
                if modelR ==_models.Ri_model_GompertzGrowthC:
                    B = _models.getB(Rmax, Sa)
                    if B == None: continue
                else: B = None
                run_label = gen_name(run_name, Q, Sa, Rmax)
                q_pars = (0, Q)
                
                for k, K in enumerate(Ks):
                    print(f"{run_label}, {k} / {len(Ks)}")
                    extinctions = 0
                    run_counter = 0
                    for _ in range(num_runs):
                        sp = _population.Population(K, B, Rmax, Sa)
                        for y in years:
                            sp.iterate(modelR, modelN, _models.normal_dist(*q_pars), **kwargs)
                            # record extinction and stop run
                            if sp.EXTANT == False:
                                extinctions += 1
                                run_counter += 1
                                break
                            # For cases where something goes wrong 
                            if sp.RUNABORT == True:
                                break
                        
                        if sp.EXTANT and not sp.RUNABORT:
                            extinctions += 0 # for clarity
                            run_counter += 1
                            
                    if run_counter > 0:
                        P = 1 - extinctions / run_counter
                        # Store results
                        odf.loc[len(odf), ["runName", "K", "B", "Q", "Rmax", "N", "P", "Sa"]] = [
                            run_label, K, B, Q, Rmax, num_runs, P, Sa]
                        
                odf.to_csv(os.path.join(results_path, run_label + ".csv"))