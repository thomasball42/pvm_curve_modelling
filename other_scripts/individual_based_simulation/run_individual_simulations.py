import _simulate_individuals as sim
import pandas as pd
import numpy as np
from pathlib import Path
import tqdm
import multiprocessing as mp
from functools import partial

RESULTS_DIR = Path("..", "results", "individual_simulation_results", "individual_simulation_examples_2 ")

CARRYING_CAPACITY_SPACE = [int(x) for x in np.geomspace(1, 1000000, 50)]
MP_THREADS = 32

num_runs = 200
year_threshold = 100
mortality_space = np.arange(0.08, 0.2, 0.02)

if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True)

def run_single(_, carrying_capacity, year_threshold, mortality_rate):
    return sim.main(
        carrying_capacity=carrying_capacity,
        plot=False,
        years=year_threshold,
        P_DEATH=mortality_rate,
    )

def run_batch(carrying_capacity, mortality_rate, num_runs, year_threshold, pool):
    """Run num_runs simulations in parallel and aggregate results."""
    worker = partial(
        run_single,
        carrying_capacity=carrying_capacity,
        year_threshold=year_threshold,
        mortality_rate=mortality_rate,
    )

    results = pool.map(worker, range(num_runs))

    extinctions = 0
    year_extinct = []
    for year, extant in results:
        if not extant:
            year_extinct.append(year)
            extinctions += 1

    P = 1 - extinctions / num_runs
    P_SEM = np.sqrt((P * (1 - P)) / num_runs)
    mean_TTE = np.mean(year_extinct) if extinctions > 0 else np.nan
    mean_TTE_SEM = np.std(year_extinct) / np.sqrt(extinctions) if extinctions > 0 else np.nan

    return {
        "K": carrying_capacity,
        "N": num_runs,
        "P": P,
        "P_SEM": P_SEM,
        "mean_TTE": mean_TTE,
        "mean_TTE_SEM": mean_TTE_SEM,
        "mortality_rate": mortality_rate,
    }


if __name__ == "__main__":
    total_runs = len(CARRYING_CAPACITY_SPACE) * num_runs * len(mortality_space)

    with mp.Pool(processes=MP_THREADS) as pool:
        with tqdm.tqdm(total=total_runs, desc="Running simulations") as pbar:

            for mortality_rate in mortality_space:
                df = pd.DataFrame(columns=["K", "N", "P", "P_SEM", "mean_TTE", "mean_TTE_SEM", "mortality_rate"])

                for carrying_capacity in CARRYING_CAPACITY_SPACE:

                    dat = run_batch(carrying_capacity, mortality_rate, num_runs, year_threshold, pool)
                    df = pd.concat([df, pd.DataFrame([dat])])
                    pbar.update(num_runs)

                df.to_csv(
                    RESULTS_DIR / f"individual_simulations_mort{mortality_rate:.2f}.csv",
                    index=False,
                )