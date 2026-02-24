# -*- coding: utf-8 -*-

import numpy as np
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _models
import _simulate
import run_scenarios_main

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
NUM_WORKERS = 32
MULTIPROCESSING_ENABLED = True
OVERWRITE_EXISTING_FILES = False

# Paths
RESULTS_PATH = Path("..", "results", "simulation_results", "results_varying_K")

NUM_RUNS = 2
CARRYING_CAPACITIES = np.unique(np.round(np.geomspace(1, 300000, num=100)))

QSD_SPACE = [0.05]
RMAX_SPACE = [0.3]
YEARS_SPACE = [100]

K_SCHEDULE_SPACE = [None, "increase", "decrease", "random_walk"]
K_STRENGTH_SPACE = [0.05]

# Run Configuration
RUNS = {
    "LogGrowthA_VaryingK": {
        "modelR": _models.Ri_model_A,
        "modelN": _models.Ni_log,
        "modelQ": _models.Q_normal_dist,
        "num_runs": NUM_RUNS,
        "kwargs": {"K_SCHEDULE_SPACE": K_SCHEDULE_SPACE, 
                   "K_STRENGTH_SPACE": K_STRENGTH_SPACE, 
                   "QSD_SPACE": QSD_SPACE, 
                   "RMAX_SPACE": RMAX_SPACE, 
                   "YEARS_SPACE": YEARS_SPACE}
    }
}

if __name__ == '__main__':
    run_scenarios_main.main(RUNS, 
                            RESULTS_PATH, 
                            MULTIPROCESSING_ENABLED=True, 
                            NUM_WORKERS=NUM_WORKERS,
                            OVERWRITE_EXISTING_FILES=False, 
                            CARRYING_CAPACITIES=CARRYING_CAPACITIES)
