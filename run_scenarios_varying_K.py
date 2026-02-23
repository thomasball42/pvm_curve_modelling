# -*- coding: utf-8 -*-

import numpy as np
import os
from pathlib import Path

import _models
import _simulate

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
NUM_WORKERS = 20
MULTIPROCESSING_ENABLED = True
OVERWRITE_EXISTING_FILES = False

# Paths
RESULTS_PATH = Path("..", "results", "simulation_results", "results_varying_K")

# Simulation Parameters
NUM_RUNS = 10000
CARRYING_CAPACITIES = np.unique(np.round(np.geomspace(1, 3000000, num=200)))

# Parameter Spaces
QSD_SPACE = np.arange(0.05, 0.55, 0.03)
RMAX_SPACE = np.array([round(z, 3) for z in np.linspace(0.055, 0.774, 15)])
YEARS_SPACE = [100]

K_SCHEDULE_SPACE = ["linear_increase", "linear_decrease", "random_walk"]
STRENGTH_SPACE = [0.1, 0.25, 0.5, 1.0]

# Run Configuration
RUNS = {
    "LogGrowthA_VaryingK": {
        "modelR": _models.Ri_model_A,
        "modelN": _models.Ni_log,
        "modelQ": _models.Q_normal_dist,
        "num_runs": NUM_RUNS,
        "kwargs": {}
    }
}


def generate_tasks_varying_K(RUNS):
    """Generator that yields all parameter combinations for varying K runs."""
    for run_name, run_params in RUNS.items():
        qsd_space = run_params["kwargs"].get("QSD_SPACE", QSD_SPACE)
        rmax_space = run_params["kwargs"].get("RMAX_SPACE", RMAX_SPACE)
        years_space = run_params["kwargs"].get("YEARS_SPACE", YEARS_SPACE)

        for year_threshold in years_space:
            for qsd in qsd_space:
                for rmax in rmax_space:
                    for K_schedule in K_SCHEDULE_SPACE:
                        for strength in STRENGTH_SPACE:
                            yield (run_name, run_params, qsd, None, rmax, None, None, None,
                                   year_threshold, K_schedule, strength)


def main(RUNS,
         RESULTS_PATH,
         MULTIPROCESSING_ENABLED=True,
         NUM_WORKERS=10,
         OVERWRITE_EXISTING_FILES=True,
         CARRYING_CAPACITIES=np.unique(np.round(np.geomspace(1, 3000000, num=200)))):

    tasks = list(generate_tasks_varying_K(RUNS))

    # Ensure results path exists
    if not os.path.isdir(RESULTS_PATH):
        try:
            print(f"Creating results directory at: {RESULTS_PATH}")
            os.makedirs(RESULTS_PATH, exist_ok=True)
        except Exception as e:
            print(f"Error creating RESULTS_PATH: {e}")
            quit()

    if MULTIPROCESSING_ENABLED:

        if NUM_WORKERS > len(tasks):
            print(f"Warning: NUM_WORKERS ({NUM_WORKERS}) is greater than the number of tasks ({len(tasks)}). Reducing to {len(tasks)}.")
            NUM_WORKERS = len(tasks)

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [
                executor.submit(_simulate.simulate_varying_K,
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
            _simulate.simulate_varying_K(RESULTS_PATH, OVERWRITE_EXISTING_FILES, MULTIPROCESSING_ENABLED,
                                         CARRYING_CAPACITIES,
                                         *task_params)


if __name__ == '__main__':
    main(RUNS, RESULTS_PATH,
         MULTIPROCESSING_ENABLED=MULTIPROCESSING_ENABLED,
         NUM_WORKERS=NUM_WORKERS,
         OVERWRITE_EXISTING_FILES=OVERWRITE_EXISTING_FILES,
         CARRYING_CAPACITIES=CARRYING_CAPACITIES)
