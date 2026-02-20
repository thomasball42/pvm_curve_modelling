import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import os
import run_scenarios_main

RESULTS_PATH = Path("..", "results", "simulation_results", "results_N_L_2005")

bird_demogr_data_file = Path("manuscript_inputs\\niel_lebreton2005_bird_demographics.csv")
bird_growth_rate_data_file = Path("manuscript_inputs\\niel_lebreton2005_bird_growth_rates.csv")

bird_demogr_df = pd.read_csv(bird_demogr_data_file, index_col=0)

bird_growth_rate_df = pd.read_csv(bird_growth_rate_data_file, index_col=0)

RUNS = {}

for index, row in bird_demogr_df.iterrows():
    
    bird = bird_demogr_df.loc[index]
    
    runName = f"{index}_{bird["Binomial"]}"

    bird_B = bird["s_i>="]
    bird_Sa = bird["si"]

    bird_Rmax1 = bird_growth_rate_df.loc[index, "lambda_max (MM)"] - 1
    bird_Rmax2 = bird_growth_rate_df.loc[index, "lambda_max (DIM)"] - 1 
    rmax_space = [bird_Rmax1, bird_Rmax2]
    
    task_params = {
        "modelR": run_scenarios_main.RUNS["LogGrowthC"]["modelR"],
        "modelN": run_scenarios_main.RUNS["LogGrowthC"]["modelN"],
        "modelQ": run_scenarios_main.RUNS["LogGrowthC"]["modelQ"],
        "num_runs": 10000,
        "kwargs": {"RMAX_SPACE": rmax_space, 
                   "SA_SPACE": [bird_Sa], 
                   "B_SPACE": [bird_B]},
    }

    RUNS[runName] = task_params

if __name__ == '__main__':
    run_scenarios_main.main(RUNS, 
                            RESULTS_PATH, 
                            MULTIPROCESSING_ENABLED=True, 
                            NUM_WORKERS=10,
                            OVERWRITE_EXISTING_FILES=True, 
                            CARRYING_CAPACITIES=run_scenarios_main.CARRYING_CAPACITIES)
