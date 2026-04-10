import _simulate_individuals as sim
import pandas as pd
import numpy as np
from pathlib import Path
import tqdm

RESULTS_DIR = Path("..", "results", "individual_simulation_results", "individual_simulation_examples")

CARRYING_CAPACITY_SPACE = [int(x) for x in np.geomspace(1, 10000, 50)]

num_runs = 2
year_threshold = 100
mortality_space = np.arange(0.15, 0.4, 0.05)

if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True)

df = pd.DataFrame(columns=["K", "N", "P", "P_SEM", "mean_TTE", "mean_TTE_SEM", "mortality_rate"])

total_runs = len(CARRYING_CAPACITY_SPACE) * num_runs * len(mortality_space)

with tqdm.tqdm(total=total_runs, desc="Running simulations") as pbar:

    for mortality_rate in mortality_space:

        for carrying_capacity in CARRYING_CAPACITY_SPACE:

            extinctions = 0
            run_count = 0
            year_extinct = []

            for _ in range(num_runs):

                year, extant = sim.main(carrying_capacity=carrying_capacity, 
                                        plot=False, years = year_threshold, P_DEATH=mortality_rate)

                if not extant:
                    year_extinct.append(year)
                    extinctions += 1

                run_count += 1

                pbar.update(1)

            P = 1 - extinctions / run_count
            P_SEM = np.sqrt((P * (1 - P)) / run_count)

            mean_TTE = np.mean(year_extinct) if extinctions > 0 else np.nan
            mean_TTE_SEM = np.std(year_extinct) / np.sqrt(extinctions) if extinctions > 0 else np.nan

            dat = {
                "K": carrying_capacity,
                "N": num_runs,
                "P": P,
                "P_SEM": P_SEM,
                "mean_TTE": mean_TTE,
                "mean_TTE_SEM": mean_TTE_SEM,
                "mortality_rate": mortality_rate
            }

            df = pd.concat([df, pd.DataFrame([dat])])

        df.to_csv(RESULTS_DIR / f"individual_simulations_mort_{mortality_rate:.2f}.csv", index=False)
