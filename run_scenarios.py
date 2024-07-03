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

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

NUM_WORKERS = 80
multi = False

# =============================================================================
# PARAMS
# =============================================================================

results_path = "/maps/tsb42/pvm_curve/results_fixed_N0"

num_runs = 10000
Ks = np.geomspace(1, 30000, num = 200)
num_years = 100

# Q_space = np.arange(0.1, 0.55, 0.05)
# Rmax_space = [0.265, 0.373, 0.447, 0.509, 0.56, 0.619, 0.644, 0.71, 0.774]
# Sa_space = np.arange(0.35, 0.95, 0.05)

Q_space = [0.15]
Rmax_space = [0.56]
Sa_space = [0.75]

N0_space = [100, 1000, 10000]

# =============================================================================
# SETUP
# =============================================================================
years = np.arange(0, num_years, 1)
Ks = np.unique(np.round(Ks))

# set up some runs
runs = {
    "fixedN0_LGA": {
        "modelR": _models.Ri_model_A,
        "modelN" : _models.Ni_log,
        "num_runs": num_runs,
        "kwargs": {}
    },
    "fixedN0_LGB": {
        "modelR": _models.Ri_model_B,
        "modelN" : _models.Ni_log,
        "num_runs": num_runs,
        "kwargs": {}
    },
    "fixedN0_LGC": {
        "modelR": _models.Ri_model_C,
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
    # quit()
    
# =============================================================================
# Main
# =============================================================================
def gen_name(run_name, S, Sa, Rmax, N0):
    if Sa == None:
        st = f"{run_name}_Q{round(S, 3)}_Rmax{round(Rmax, 3)}_SA_nan_N0{round(N0, 3)}"
    else:
        st = f"{run_name}_Q{round(S, 3)}_Rmax{round(Rmax, 3)}_SA{round(Sa, 3)}_N0{round(N0, 3)}"
    return st

# run a set of sims across Ks and iters
def simulate(run_name, run_params, Q, Rmax, Sa, N0):
    modelR = run_params["modelR"]
    modelN = run_params["modelN"]
    num_runs = run_params["num_runs"]
    kwargs = run_params["kwargs"]
    global odf
    odf = pd.DataFrame()
    if modelR == _models.Ri_model_C:
        B = _models.getB(Rmax, Sa)
        if B is None: 
            return
    else:
        B = None
    run_label = gen_name(run_name, Q, Sa, Rmax, N0)
    q_pars = (0, Q)
    for k, K in enumerate(Ks):
        if not multi:
            print(f"{run_label}, {k} / {len(Ks)}")
        extinctions = 0
        run_counter = 0
        for _ in range(num_runs):
            sp = _population.Population(K, B, Rmax, Sa, N0)
            for y in years:
                sp.iterate(modelR, modelN, _models.normal_dist(*q_pars), **kwargs)
                if not sp.EXTANT:
                    extinctions += 1
                    run_counter += 1
                    break
                if sp.RUNABORT:
                    break

            if sp.EXTANT and not sp.RUNABORT:
                extinctions += 0
                run_counter += 1
        if run_counter > 0:
            P = 1 - extinctions / run_counter
            odf.loc[len(odf), ["runName", "K", "B", "Q", "Rmax", "N", "P", "Sa", "N0"]] = [
                run_label, K, B, Q, Rmax, num_runs, P, Sa, N0]
    odf.to_csv(os.path.join(results_path, run_label + ".csv"))
    
# run sims
if __name__ == '__main__':
    if multi:
        tasks = len(runs) * len(Q_space) * len(Rmax_space) * len(Sa_space)
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            with tqdm(total=tasks) as progressor:
                for run_name, run_params in runs.items():
                    for Q in Q_space:
                        for Rmax in Rmax_space:
                            for N0 in N0_space:
                                SA_ITERATOR = Sa_space if run_params["modelR"] == _models.Ri_model_C else [None]
                                for Sa in SA_ITERATOR:
                                    futures.append(executor.submit(simulate, run_name, run_params, Q, Rmax, Sa, N0))
            
                for future in as_completed(futures):
                    future.result()
                    progressor.update(1)
    else:
        for run_name, run_params in runs.items():
            for Q in Q_space:
                for Rmax in Rmax_space:
                    for N0 in N0_space:
                        SA_ITERATOR = Sa_space if run_params["modelR"] == _models.Ri_model_C else [None]
                        for Sa in SA_ITERATOR:
                            simulate(run_name, run_params, Q, Rmax, Sa, N0)