import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
import pandas as pd
import numpy as np

import _models
import _population

def generate_filename(run_name, qsd, qrev=None, Sa=None, Rmax=None, N0=None, year_threshold=100, **kwargs):
    parts = [
        f"{run_name}",
        f"QSD{round(qsd, 3)}",
        f"YT{year_threshold}",
        f"RMAX{round(Rmax, 3)}" if Rmax is not None else None,
        f"SA{round(Sa, 3)}" if Sa is not None else None,
        f"QREV{round(qrev, 3)}" if qrev is not None else None,
        f"N0{round(N0, 3)}" if N0 is not None else None,    
    ]

    if "allee_params_theta_upsil" in kwargs.keys():
        ALLEE_param = kwargs["allee_params_theta_upsil"]
        if isinstance(ALLEE_param, tuple) and len(ALLEE_param) == 2:
            theta, upsil = ALLEE_param
            parts.append(f"ALLEEtheta{round(theta, 3)}_ALLEEupsil{round(upsil, 3)}")
    return "_".join(filter(None, parts))  # Remove None values

def simulate(RESULTS_PATH, OVERWRITE_EXISTING_FILES, MULTIPROCESSING_ENABLED,
             CARRYING_CAPACITIES,
             run_name, run_params, qsd, qrev, Rmax=None, Sa=None, N0=None, B=None, year_threshold=100, 
             **kwargs):
    
    modelR = run_params["modelR"]
    modelN = run_params["modelN"]
    modelQ = run_params["modelQ"]
    num_runs = run_params["num_runs"]
    kwargs = {**run_params["kwargs"], **kwargs}

    if B is None:
        if modelR == _models.Ri_model_C and Sa is not None and Rmax is not None:
            B = _models.getB(Rmax, Sa)
            if B is None:
                return  # Skip this simulation if B is invalid

    filename = generate_filename(run_name, qsd, qrev, Sa, Rmax, N0, year_threshold, **kwargs)
    filepath = os.path.join(RESULTS_PATH, filename + ".csv")

    
    if os.path.isfile(filepath) and not OVERWRITE_EXISTING_FILES:
        return None

    q_params = (0, qsd, qrev)
    kwargs["q_params"] = q_params

    results_df = pd.DataFrame()
    for idx, K in enumerate(CARRYING_CAPACITIES):

        N0_run = K if N0 is None else N0  # Modify as needed for non-K initialisations

        if not MULTIPROCESSING_ENABLED:
            print(f"{filename}, {idx + 1} / {len(CARRYING_CAPACITIES)}")

        extinctions = 0
        run_count = 0
        year_extinct = []
        for _ in range(num_runs):

            population = _population.Population(K, B, Rmax, Sa, N0_run)
            year = 0
            while year < year_threshold:
                population.iterate(modelR, modelN, modelQ, **kwargs)
                if not population.EXTANT:
                    year_extinct.append(year)
                    extinctions += 1
                    run_count += 1
                    break
                if population.RUNABORT:
                    break
                year += 1
            if population.EXTANT and not population.RUNABORT:
                run_count += 1

        if run_count > 0:
            mean_tte = np.mean(year_extinct) if extinctions > 0 else np.nan
            mean_tte_sem = np.std(year_extinct) / np.sqrt(extinctions) if extinctions > 0 else np.nan

            survival_probability = 1 - extinctions / run_count
            survival_probability_sem = np.sqrt((survival_probability * (1 - survival_probability)) / run_count) if run_count > 0 else np.nan # standard error for a proportion

            results_df.loc[len(results_df), [
                "runName", "K", "B", "QSD", "QREV", "RMAX", "N", 
                "P", "P_SEM", "SA", "N0", "YEAR_THRESHOLD", "mean_TTE", "mean_TTE_SEM"
            ]] = [
                filename, K, B, qsd, qrev, Rmax, num_runs, 
                survival_probability, survival_probability_sem, Sa, N0_run, year_threshold, mean_tte, mean_tte_sem
            ]

    results_df.to_csv(filepath, index=False)


def generate_filename_varying_K(run_name, qsd, qrev=None, Sa=None, Rmax=None, N0=None, year_threshold=100,
                                 K_schedule=None, strength=None, **kwargs):
    parts = [
        f"{run_name}",
        f"QSD{round(qsd, 3)}",
        f"YT{year_threshold}",
        f"RMAX{round(Rmax, 3)}" if Rmax is not None else None,
        f"SA{round(Sa, 3)}" if Sa is not None else None,
        f"QREV{round(qrev, 3)}" if qrev is not None else None,
        f"N0{round(N0, 3)}" if N0 is not None else None,
        f"KSCHED_{K_schedule}" if K_schedule is not None else None,
        f"KSTR{strength}" if strength is not None else None,
    ]
    return "_".join(filter(None, parts))


def simulate_varying_K(RESULTS_PATH, OVERWRITE_EXISTING_FILES, MULTIPROCESSING_ENABLED,
                       CARRYING_CAPACITIES,
                       run_name, run_params, qsd, qrev, Rmax=None, Sa=None, N0=None, B=None, year_threshold=100,
                       K_schedule="linear_increase", strength=0.1, K_schedule_fn=None,
                       **kwargs):
    """
    Simulates population dynamics with a time-varying carrying capacity K.

    Mirrors the structure of simulate(), but accepts a K schedule that updates
    population.K, population.Km and population.Kf each year before calling iterate().

    Parameters
    ----------
    K_schedule : str
        One of "linear_increase", "linear_decrease", "random_walk", or "custom".
    strength : float
        Magnitude of K change per year_threshold units.
    K_schedule_fn : callable, optional
        Used when K_schedule == "custom". Signature: K_schedule_fn(t, K0, strength) -> float.
    """
    modelR = run_params["modelR"]
    modelN = run_params["modelN"]
    modelQ = run_params["modelQ"]
    num_runs = run_params["num_runs"]
    kwargs = {**run_params["kwargs"], **kwargs}

    if B is None:
        if modelR == _models.Ri_model_C and Sa is not None and Rmax is not None:
            B = _models.getB(Rmax, Sa)
            if B is None:
                return

    filename = generate_filename_varying_K(run_name, qsd, qrev, Sa, Rmax, N0, year_threshold,
                                           K_schedule, strength, **kwargs)
    filepath = os.path.join(RESULTS_PATH, filename + ".csv")

    if os.path.isfile(filepath) and not OVERWRITE_EXISTING_FILES:
        return None

    q_params = (0, qsd, qrev)
    kwargs["q_params"] = q_params

    results_df = pd.DataFrame()
    for idx, K0 in enumerate(CARRYING_CAPACITIES):

        N0_run = K0 if N0 is None else N0

        if not MULTIPROCESSING_ENABLED:
            print(f"{filename}, {idx + 1} / {len(CARRYING_CAPACITIES)}")

        extinctions = 0
        run_count = 0
        year_extinct = []
        for _ in range(num_runs):

            population = _population.Population(K0, B, Rmax, Sa, N0_run)
            current_K = K0
            year = 0
            while year < year_threshold:
                # Compute new K for this timestep from the schedule
                if K_schedule == "linear_increase":
                    new_K = K0 * (1 + strength * year / year_threshold)
                elif K_schedule == "linear_decrease":
                    new_K = K0 * max(0, 1 - strength * year / year_threshold)
                elif K_schedule == "random_walk":
                    new_K = max(1, current_K * np.exp(np.random.normal(0, strength)))
                elif K_schedule == "custom":
                    new_K = K_schedule_fn(year, K0, strength)
                else:
                    raise ValueError(f"Unknown K_schedule: {K_schedule}")

                current_K = new_K
                population.K = new_K
                population.Km = new_K / 2
                population.Kf = new_K / 2

                population.iterate(modelR, modelN, modelQ, **kwargs)
                if not population.EXTANT:
                    year_extinct.append(year)
                    extinctions += 1
                    run_count += 1
                    break
                if population.RUNABORT:
                    break
                year += 1
            if population.EXTANT and not population.RUNABORT:
                run_count += 1

        if run_count > 0:
            mean_tte = np.mean(year_extinct) if extinctions > 0 else np.nan
            mean_tte_sem = np.std(year_extinct) / np.sqrt(extinctions) if extinctions > 0 else np.nan

            survival_probability = 1 - extinctions / run_count
            survival_probability_sem = np.sqrt((survival_probability * (1 - survival_probability)) / run_count) if run_count > 0 else np.nan

            results_df.loc[len(results_df), [
                "runName", "K", "B", "QSD", "QREV", "RMAX", "N",
                "P", "P_SEM", "SA", "N0", "YEAR_THRESHOLD", "mean_TTE", "mean_TTE_SEM",
                "K_SCHEDULE", "STRENGTH"
            ]] = [
                filename, K0, B, qsd, qrev, Rmax, num_runs,
                survival_probability, survival_probability_sem, Sa, N0_run, year_threshold, mean_tte, mean_tte_sem,
                K_schedule, strength
            ]

    results_df.to_csv(filepath, index=False)
