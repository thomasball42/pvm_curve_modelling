# -*- coding: utf-8 -*-
"""
Fit basic Gompertz curves to extinction probability data from simulation results.

@author: Thomas Ball
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
import _curve_fit
import _analysis_utils as utils


# Configuration
SCALE_1_0 = False
PLOT_PSPACE = False
PLOT_CURVES = False
OVERWRITE = False

# Paths
RESULTS_PATH = "..\\results\\simulation_results\\results_main"
DATA_FITS_PATH = "..\\results\\data_fits\\data_fits_main"
MODELS = ["A", "B", "C", "D"]

FUNC = _curve_fit.basic_gomp
PARAM_NAMES = ("bg_param_a", "bg_param_b")


def process_simulation_file(file, existing_runs,
                            scale_1_0=False, plot_curves=False,
                            plot_pspace=False, axs=None):
    # Load data
    dat = pd.read_csv(file)
    run_params = utils.extract_run_parameters(dat)

    # Skip if already processed
    if run_params['runName'] in existing_runs:
        print(f"Skipping {run_params['runName']} - already processed")
        return None

    # Extract data
    x = dat.K
    y = dat.P

    # Fit basic Gompertz curve
    params = tuple([np.nan] * len(PARAM_NAMES))
    R2 = np.nan
    resids = np.nan
    rsd = np.nan
    rmse = np.nan
    model_name = np.nan

    try:
        ret = _curve_fit.fit(FUNC, x, y)
    except RuntimeError:
        ret = None

    if ret is not None:
        params, y_predicted, R2, resids = ret
        model_name = FUNC.__name__
        rsd  = np.sqrt(np.sum(resids ** 2) / len(resids))
        rmse = np.sqrt(np.sum(resids ** 2) / (len(resids) - len(params) + 1))

    # Calculate extinction metrics
    # basic_gomp has no alpha; pad with alpha=1 for calculate_kx_analytical
    xff = np.geomspace(dat.K.min(), dat.K.max(), num=100_000)
    yff = FUNC(xff, *params) if not np.isnan(params[0]) else np.full_like(xff, np.nan)
    metrics = utils.calculate_extinction_metrics(xff, yff, (*params, 1))

    # Create results DataFrame
    kX_names = [f"k{int(X*100)}" for X in np.arange(0.1, 1.0, 0.1)]

    ddf = pd.DataFrame()
    ddf.loc[0, ["model", "runName", "RMAX", "QSD", "QREV", "B", "SA",
                "model_name", *PARAM_NAMES, "R2", "RSD", "RMSE", "MAX_Y",
                *kX_names, "dPdK_tp"]] = [
        run_params['model'], run_params['runName'], run_params['rmax'],
        run_params['qsd'], run_params['qrev'], run_params['b'], run_params['sa'],
        model_name, *params, R2, rsd, rmse, run_params['max_p'],
        *[metrics[name] for name in kX_names], metrics['dPdK_tp']
    ]

    # Plotting
    if plot_curves and axs is not None:
        n = int(plot_curves) + int(plot_pspace)
        ax = axs[-1] if n > 1 else axs
        utils.plot_curve_fit(ax, x, y, FUNC, params, R2,
                             run_params['runName'], scale_1_0)
        ax.set_ylabel(f"Probability of extinction (N={round(run_params['n'])})")
        ax.set_xlabel("Carrying capacity K")

    if plot_pspace and axs is not None:
        n = int(plot_curves) + int(plot_pspace)
        ax = axs[0] if n > 1 else axs
        utils.plot_parameter_space(ax, run_params['qsd'], run_params['rmax'],
                                   R2, params)
        ax.set_ylabel("R$_{max}$")
        ax.set_xlabel("Stochasticity SD")

    return ddf


def main():
    for mm in MODELS:
        print(f"\n{'='*50}")
        print(f"Processing Model {mm}")
        print(f"{'='*50}")

        # Setup paths
        data_fits_path = Path(DATA_FITS_PATH) / f"data_fits_{mm}_basic_gompertz.csv"

        # Find simulation files
        f = utils.find_simulation_files(RESULTS_PATH, mm)
        print(f"Found {len(f)} files for model {mm}")

        # Setup plotting
        n = int(PLOT_CURVES) + int(PLOT_PSPACE)
        if n > 0:
            fig, axs = plt.subplots(1, n)
            fig.set_size_inches(12, 7)
        else:
            axs = None

        # Load existing fits
        data_fits, existing_runs = utils.load_existing_fits(data_fits_path, OVERWRITE)

        # Process each file
        for i, file in enumerate(f):
            print(f"\nModel {mm}: {i+1}/{len(f)} ({100*(i+1)/len(f):.1f}%)")

            ddf = process_simulation_file(
                file, existing_runs,
                scale_1_0=SCALE_1_0,
                plot_curves=PLOT_CURVES,
                plot_pspace=PLOT_PSPACE,
                axs=axs
            )

            # Save results incrementally
            if ddf is not None:
                os.makedirs(data_fits_path.parent, exist_ok=True)
                data_fits = pd.concat([data_fits, ddf])
                data_fits.to_csv(data_fits_path)
                existing_runs.add(ddf.loc[0, 'runName'])

        # Finalise plots
        if PLOT_PSPACE and n > 0:
            ax = axs[0] if n > 1 else axs
            norm = plt.Normalize(0.990, 1)
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('R2')

        if n > 0:
            fig.tight_layout()
            plt.show()

        print(f"\nModel {mm} complete. Results saved to {data_fits_path}")


if __name__ == "__main__":
    main()