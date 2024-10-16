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

# Configuration
NUM_WORKERS = 200
MULTIPROCESSING_ENABLED = False
OVERWRITE_EXISTING_FILES = False

# Paths
RESULTS_PATH = ""

# Simulation Parameters
NUM_RUNS = 10000
NUM_YEARS = 100
CARRYING_CAPACITIES = np.geomspace(1, 3000000, num=200)
CARRYING_CAPACITIES = np.unique(np.round(CARRYING_CAPACITIES))
YEARS = np.arange(0, NUM_YEARS, 1)

# Parameter Spaces
QSD_SPACE = np.arange(0.05, 0.55, 0.03)
RMAX_SPACE = np.array([round(z, 3) for z in np.linspace(0.055, 0.774, 15)])
SA_SPACE = np.arange(0.35, 0.95 + 0.15, 0.15)

QREV_SPACE = np.linspace(1, 100, 5) / 100
N0_SPACE = [0]  # MODIFY THE CODE TO CHANGE N0 TO ANYTHING OTHER THAN K

# Run Configuration
RUNS = {
    "LogGrowthA": {
        "modelR": _models.Ri_model_A,
        "modelN": _models.Ni_log,
        "modelQ": _models.Q_normal_dist,
        "num_runs": NUM_RUNS,
        "kwargs": {}
    },
    "LogGrowthB": {
        "modelR": _models.Ri_model_B,
        "modelN": _models.Ni_log,
        "modelQ": _models.Q_normal_dist,
        "num_runs": NUM_RUNS,
        "kwargs": {}
    },
    "LogGrowthC2": {
        "modelR": _models.Ri_model_C,
        "modelN": _models.Ni_log,
        "modelQ": _models.Q_normal_dist,
        "num_runs": NUM_RUNS,
        "kwargs": {}
    },
    "LogGrowthD2": {
        "modelR": _models.Ri_model_C,
        "modelN": _models.Ni_log,
        "modelQ": _models.Q_ornstein_uhlenbeck,
        "num_runs": NUM_RUNS,
        "kwargs": {}
    },
}

# Ensure results path exists
if not os.path.isdir(RESULTS_PATH):
    raise ValueError("RESULTS_PATH must be a valid directory")

def generate_filename(run_name, qsd, qrev, Sa, Rmax, N0):
    return f"{run_name}_QSD{round(qsd, 3)}_QREV{round(qrev, 3) if qrev else 'nan'}_RMAX{round(Rmax, 3)}_SA{round(Sa, 3) if Sa else 'nan'}_N0{round(N0, 3)}"

def simulate(run_name, run_params, qsd, qrev, Rmax, Sa, N0):
    modelR = run_params["modelR"]
    modelN = run_params["modelN"]
    modelQ = run_params["modelQ"]
    num_runs = run_params["num_runs"]
    kwargs = run_params["kwargs"]
    if modelR == _models.Ri_model_C:
        B = _models.getB(Rmax, Sa)
        if B is None:
            return
    else:
        B = None

    filename = generate_filename(run_name, qsd, qrev, Sa, Rmax, N0)
    filepath = os.path.join(RESULTS_PATH, filename + ".csv")
    
    if os.path.isfile(filepath) and not OVERWRITE_EXISTING_FILES:
        return
    q_params = (0, qsd, qrev)
    kwargs["q_params"] = q_params
    results_df = pd.DataFrame()
    for idx, K in enumerate(CARRYING_CAPACITIES):
        N0 = K  # MODIFY AS NEEDED
        if not MULTIPROCESSING_ENABLED:
            print(f"{filename}, {idx + 1} / {len(CARRYING_CAPACITIES)}")
        extinctions = 0
        run_count = 0
        for _ in range(num_runs):
            population = _population.Population(K, B, Rmax, Sa, N0)
            for year in YEARS:
                population.iterate(modelR, modelN, modelQ, **kwargs)
                if not population.EXTANT:
                    extinctions += 1
                    run_count += 1
                    break
                if population.RUNABORT:
                    break
            if population.EXTANT and not population.RUNABORT:
                run_count += 1
        if run_count > 0:
            survival_probability = 1 - extinctions / run_count
            results_df.loc[len(results_df),
                           ["runName", "K", "B", "QSD", "QREV", "RMAX", "N", "P", "SA", "N0"]] = [
                               filename, K, B, qsd, qrev, Rmax, num_runs, survival_probability, Sa, N0]
    results_df.to_csv(filepath)

def main():
    if MULTIPROCESSING_ENABLED:
        task_count=0
        for run_name, run_params in RUNS.items():
            for qsd in QSD_SPACE:
                for Rmax in RMAX_SPACE:
                    for N0 in N0_SPACE:
                        qrev_iterator = QREV_SPACE if run_params["modelQ"] == _models.Q_ornstein_uhlenbeck else [None]
                        for qrev in qrev_iterator:
                            sa_iterator = SA_SPACE if run_params["modelR"] == _models.Ri_model_C else [None]
                            for Sa in sa_iterator:
                                task_count += 1
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            with tqdm(total=task_count) as progress_bar:
                for run_name, run_params in RUNS.items():
                    for qsd in QSD_SPACE:
                        for Rmax in RMAX_SPACE:
                            for N0 in N0_SPACE:
                                sa_iterator = SA_SPACE if run_params["modelR"] == _models.Ri_model_C else [None]
                                qrev_iterator = QREV_SPACE if run_params["modelQ"] == _models.Q_ornstein_uhlenbeck else [None]
                                for qrev in qrev_iterator:
                                    for Sa in sa_iterator:
                                        futures.append(executor.submit(simulate, run_name, run_params, qsd, qrev, Rmax, Sa, N0))
                for future in as_completed(futures):
                    future.result()
                    progress_bar.update(1)
    else:
        for run_name, run_params in RUNS.items():
            for qsd in QSD_SPACE:
                for qrev in QREV_SPACE:
                    for Rmax in RMAX_SPACE:
                        for N0 in N0_SPACE:
                            qrev_iterator = QREV_SPACE if run_params["modelQ"] == _models.Q_ornstein_uhlenbeck else [None]
                            for qrev in qrev_iterator:
                                sa_iterator = SA_SPACE if run_params["modelR"] == _models.Ri_model_C else [None]
                                for Sa in sa_iterator:
                                    simulate(run_name, run_params, qsd, qrev, Rmax, Sa, N0)
                                
if __name__ == '__main__':
    main()
