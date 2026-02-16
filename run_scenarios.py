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
import _simulate

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
NUM_WORKERS = 10
MULTIPROCESSING_ENABLED = True
OVERWRITE_EXISTING_FILES = True

# Paths
RESULTS_PATH = "..\\results\\simulation_results\\results_tte"

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
YEARS_SPACE = [100]

N0_SPACE = [0] # MODIFY THE CODE TO CHANGE N0 TO ANYTHING OTHER THAN K

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
    "LogGrowthC": {
        "modelR": _models.Ri_model_C,
        "modelN": _models.Ni_log,
        "modelQ": _models.Q_normal_dist,
        "num_runs": NUM_RUNS,
        "kwargs": {}
    },
    "LogGrowthD": {
        "modelR": _models.Ri_model_C,
        "modelN": _models.Ni_log,
        "modelQ": _models.Q_ornstein_uhlenbeck,
        "num_runs": NUM_RUNS,
        "kwargs": {}
    },
}

# Ensure results path exists
if not os.path.isdir(RESULTS_PATH):
    try:
        print(f"Creating results directory at: {RESULTS_PATH}")
        os.makedirs(RESULTS_PATH, exist_ok=True) 
    except Exception as e: 
        print(f"Error creating RESULTS_PATH: {e}")
        quit()

def generate_tasks():
        """Generator that yields all parameter combinations."""
        for run_name, run_params in RUNS.items():
            for year_threshold in YEARS_SPACE:
                for qsd in QSD_SPACE:
                    for Rmax in RMAX_SPACE:
                        for N0 in N0_SPACE:
                            qrev_iterator = QREV_SPACE if run_params["modelQ"] == _models.Q_ornstein_uhlenbeck else [None]
                            sa_iterator = SA_SPACE if run_params["modelR"] == _models.Ri_model_C else [None]
                            for qrev in qrev_iterator:
                                for Sa in sa_iterator:
                                    yield (run_name, run_params, qsd, qrev, Rmax, Sa, N0, year_threshold)

def main():
    
    tasks = list(generate_tasks())

    if MULTIPROCESSING_ENABLED:

        global NUM_WORKERS
        if NUM_WORKERS > len(tasks):
            print(f"Warning: NUM_WORKERS ({NUM_WORKERS}) is greater than the number of tasks ({len(tasks)}). Reducing to {len(tasks)}.")
            NUM_WORKERS = len(tasks)

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [
                executor.submit(_simulate.simulate,
                            RESULTS_PATH, OVERWRITE_EXISTING_FILES, MULTIPROCESSING_ENABLED,
                            CARRYING_CAPACITIES,
                            *task_params)
                for task_params in tasks
            ]
            
            with tqdm(total=len(futures)) as progress_bar:
                for future in as_completed(futures):
                    future.result()
                    progress_bar.update(1)
    else:
        for task_params in tasks:
            _simulate.simulate(RESULTS_PATH, OVERWRITE_EXISTING_FILES, MULTIPROCESSING_ENABLED,
                            CARRYING_CAPACITIES,
                            *task_params)
                                
if __name__ == '__main__':
    main()
