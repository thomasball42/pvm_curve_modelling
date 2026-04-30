import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import os
import run_scenarios_main

# to allow model AlleeB
import _models

RESULTS_PATH = Path("..", "results", "simulation_results", "results_griffon_vulture")

QSD_SPACE = np.arange(0.01, 0.30, 0.04)
THETA_SPACE = [round(_) for _ in np.geomspace(1, 100000, 10)]

UPSIL_SPACE = np.linspace(0, 1, 9)

N_REPEATS = 1000

bird_demogr_data_file = Path("manuscript_inputs", "niel_lebreton2005_bird_demographics.csv")
bird_growth_rate_data_file = Path("manuscript_inputs", "niel_lebreton2005_bird_growth_rates.csv")
bird_demogr_df = pd.read_csv(bird_demogr_data_file, index_col=0)
bird_growth_rate_df = pd.read_csv(bird_growth_rate_data_file, index_col=0)

RUNS = {}

for index, row in bird_demogr_df.iterrows():

    bird = bird_demogr_df.loc[index]

    if "vulture" not in bird.name.lower():
        continue

    bird_B = bird["s_i>="]
    bird_Sa = bird["si"]

    bird_Rmax1 = bird_growth_rate_df.loc[index, "lambda_max (MM)"] - 1
    bird_Rmax2 = bird_growth_rate_df.loc[index, "lambda_max (DIM)"] - 1 
    rmax_space = [bird_Rmax1, bird_Rmax2]
    
    for theta in THETA_SPACE:
        for upsil in UPSIL_SPACE:

            task_params = {
                "modelR": run_scenarios_main.RUNS["LogGrowthD"]["modelR"],
                "modelN": run_scenarios_main.RUNS["LogGrowthD"]["modelN"],
                "modelQ": run_scenarios_main.RUNS["LogGrowthD"]["modelQ"],
                "num_runs": N_REPEATS,
                "kwargs": {"RMAX_SPACE": rmax_space, 
                        "SA_SPACE": [bird_Sa], 
                        "B_SPACE": [bird_B],
                        "QSD_SPACE": QSD_SPACE,
                        "Rgen_model": _models.Ri_model_alleeB,
                        "allee_params_theta_upsil": (theta, upsil)},
            }

            runName = f"{index}_theta{theta}_upsil{upsil}"

            RUNS[runName] = task_params

if __name__ == '__main__':
    run_scenarios_main.main(RUNS, 
                            RESULTS_PATH, 
                            MULTIPROCESSING_ENABLED=True, 
                            NUM_WORKERS=10,
                            OVERWRITE_EXISTING_FILES=False, 
                            CARRYING_CAPACITIES=run_scenarios_main.CARRYING_CAPACITIES)
