# -*- coding: utf-8 -*-
"""
Fit Gompertz curves to extinction probability data from simulation results.

@author: Thomas Ball
Created on Wed May 22 13:55:22 2024
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import _curve_fit
import _analysis_utils as utils


# Configuration
SCALE_1_0 = False
PLOT_PSPACE = False
PLOT_CURVES = False
OVERWRITE = True

# Paths
RESULTS_PATH = "..\\results\\simulation_results\\results_tte"
DATA_FITS_PATH = "..\\results\\data_fits\\data_fits_tte"
MODELS = ["A", "B", "C", "D"]


def process_simulation_file(file, data_fits_path, existing_runs, 
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
    x = run_params['x']
    y = run_params['y']
    
    # Scale if requested
    if scale_1_0:
        x, y = utils.scale_to_unit_interval(x, y)
    
    # Fit Gompertz curve
    fit_result = utils.fit_gompertz_curve(x, y)
    params = fit_result['params']
    R2 = fit_result['R2']
    
    print(f"R2: {R2}")
    
    # Calculate extinction metrics
    xff = np.geomspace(dat.K.min(), dat.K.max(), num=100000)
    yff = _curve_fit.mod_gompertz(xff, *params)
    metrics = utils.calculate_extinction_metrics(xff, yff, params)
    
    # Create results DataFrame
    param_names = ("param_a", "param_b", "param_alpha")
    kX_names = [f"k{int(X*100)}" for X in np.arange(0.1, 1.0, 0.1)]
    
    ddf = pd.DataFrame()
    ddf.loc[0, ["model", "runName", "RMAX", "QSD", "QREV", "B", "SA", 
                "model_name", *param_names, "R2", "RSD", "RMSE", "MAX_Y", 
                *kX_names, "dPdK_tp"]] = [
        run_params['model'], run_params['runName'], run_params['rmax'], 
        run_params['qsd'], run_params['qrev'], run_params['B'], run_params['sa'],
        fit_result['model_name'], *params, R2, fit_result['rsd'], 
        fit_result['rmse'], run_params['max_y'], 
        *[metrics[name] for name in kX_names], metrics['dPdK_tp']
    ]
    
    # Plotting
    if plot_curves and axs is not None:
        n = int(plot_curves) + int(plot_pspace)
        ax = axs[-1] if n > 1 else axs
        utils.plot_curve_fit(ax, x, y, _curve_fit.mod_gompertz, params, R2, 
                           run_params['runName'], scale_1_0)
        ax.set_ylabel(f"Probability of extinction (N={round(run_params['N'])})")
        ax.set_xlabel("Carrying capacity K")
    
    if plot_pspace and axs is not None:
        n = int(plot_curves) + int(plot_pspace)
        ax = axs[0] if n > 1 else axs
        utils.plot_parameter_space(ax, run_params['qsd'], run_params['rmax'], 
                                  R2, params)
        ax.set_ylabel("R$_{max}$")
        ax.set_xlabel("Stochasicity SD")
    
    return ddf


def main():
    """Main processing loop."""
    for mm in MODELS:
        print(f"\n{'='*50}")
        print(f"Processing Model {mm}")
        print(f"{'='*50}")
        
        # Setup paths
        data_fits_path = Path(DATA_FITS_PATH) / f"data_fits_model_{mm}.csv"
        
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
        
        print(data_fits)
        # Process each file
        for i, file in enumerate(f):
            print(f"\nModel {mm}: {i+1}/{len(f)} ({100*(i+1)/len(f):.1f}%)")
            
            ddf = process_simulation_file(
                file, data_fits_path, existing_runs,
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
        
        # Finalize plots
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