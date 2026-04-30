import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import os
import run_scenarios_main

RESULTS_PATH = Path("..", "results", "simulation_results", "results_tte")

NUM_RUNS = 50
QSD_SPACE = [0.11]
RMAX_SPACE = [0.209]
SA_SPACE = [0.35]
QREV_SPACE = [0.5]
YEARS_SPACE = [20000]

RUNS = {
    "LogGrowthA": {
        "modelR": run_scenarios_main.RUNS["LogGrowthA"]["modelR"],
        "modelN": run_scenarios_main.RUNS["LogGrowthA"]["modelN"],
        "modelQ": run_scenarios_main.RUNS["LogGrowthA"]["modelQ"],
        "num_runs": NUM_RUNS,
            "kwargs": {"YEARS_SPACE": YEARS_SPACE,
                    "QSD_SPACE": QSD_SPACE,
                    "RMAX_SPACE": RMAX_SPACE,
                    "SA_SPACE": SA_SPACE,
                    "QREV_SPACE": QREV_SPACE}
    },
    "LogGrowthB": {
        "modelR": run_scenarios_main.RUNS["LogGrowthB"]["modelR"],
        "modelN": run_scenarios_main.RUNS["LogGrowthB"]["modelN"],
        "modelQ": run_scenarios_main.RUNS["LogGrowthB"]["modelQ"],
        "num_runs": NUM_RUNS,
        "kwargs": {"YEARS_SPACE": YEARS_SPACE,
                    "QSD_SPACE": QSD_SPACE,
                    "RMAX_SPACE": RMAX_SPACE,
                    "SA_SPACE": SA_SPACE,
                    "QREV_SPACE": QREV_SPACE}
    },
    "LogGrowthC": {
        "modelR": run_scenarios_main.RUNS["LogGrowthC"]["modelR"],
        "modelN": run_scenarios_main.RUNS["LogGrowthC"]["modelN"],
        "modelQ": run_scenarios_main.RUNS["LogGrowthC"]["modelQ"],
        "num_runs": NUM_RUNS,
        "kwargs": {"YEARS_SPACE": YEARS_SPACE,
                    "QSD_SPACE": QSD_SPACE,
                    "RMAX_SPACE": RMAX_SPACE,
                    "SA_SPACE": SA_SPACE,
                    "QREV_SPACE": QREV_SPACE}
    },
    "LogGrowthD": {
        "modelR": run_scenarios_main.RUNS["LogGrowthD"]["modelR"],
        "modelN": run_scenarios_main.RUNS["LogGrowthD"]["modelN"],
        "modelQ": run_scenarios_main.RUNS["LogGrowthD"]["modelQ"],
        "num_runs": NUM_RUNS,
        "kwargs": {"YEARS_SPACE": YEARS_SPACE,
                    "QSD_SPACE": QSD_SPACE,
                    "RMAX_SPACE": RMAX_SPACE,
                    "SA_SPACE": SA_SPACE,
                    "QREV_SPACE": QREV_SPACE}
    },
}

if __name__ == '__main__':
    run_scenarios_main.main(RUNS, 
                            RESULTS_PATH, 
                            MULTIPROCESSING_ENABLED=True, 
                            NUM_WORKERS=32,
                            OVERWRITE_EXISTING_FILES=False, 
                            CARRYING_CAPACITIES=run_scenarios_main.CARRYING_CAPACITIES)
