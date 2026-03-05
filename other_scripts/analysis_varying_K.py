import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import re

sys.path.insert(0, str(Path(__file__).parent.parent))
import _curve_fit
import _analysis_utils

OVERWRITE = True
FIGSIZE = (6, 8)

### varying K
DATA_FITS_PATH = Path("..", "results", "data_fits", "data_fits_varying_K")

figs_dir = Path("..", "figs", "figs_varying_K")
data_path = Path("..", "results", "simulation_results", "results_varying_K")

list_of_files = []
for path, subdirs, files in os.walk(data_path):
    for name in files:
        list_of_files.append(os.path.join(path, name)) 

files = [file for file in list_of_files if ".csv" in file and "LogGrowth" in file]

base_file = [file for file in files if "RMAX0.3.csv" in file]


models = ["increase", "decrease", "random_walk"]

fig, axs = plt.subplots(3, figsize=FIGSIZE, sharex=True, sharey=True)

for m, run_model in enumerate(models):

    model_files = [file for file in files if f"{run_model}_KSTR" in file]
    
    model_files = base_file + model_files

    ax = axs.flatten()[models.index(run_model)]
    
    data_fits_path = Path(DATA_FITS_PATH) / f"data_fits_model_{run_model}.csv"

    data_fits, existing_runs = _analysis_utils.load_existing_fits(data_fits_path, OVERWRITE)

    for f, file in enumerate(model_files):

        dat = pd.read_csv(file)
        run_params = _analysis_utils.extract_run_parameters(dat)
        run_name = run_params["runName"]
        
        print(f"{f}/{len(files)}: {run_name}")
        
        x = dat.K
        y = dat.P
        
        max_y = np.max(y)

        # if max_y != 1:
        #     continue 

        fit_result = _analysis_utils.fit_gompertz_curve(x, y, 
                                                alpha_space=np.arange(0, 1, 0.05), 
                                                ylim=(0.05, 0.95),
                                                iteration_depth= 4
                                                )
        
        fit_params = fit_result["params"]
        R2 = fit_result["R2"]
        
        try:
            KS = dat.K_SCHEDULE.unique()[0]
        except:
            KS = "Static"

        try:
            KSTR = re.search(r'KSTR(\d+(?:\.\d+)?)', file).group(1)
        except:
            KSTR = "0.00"

        if KSTR == "0.1": KSTR = "0.10"

        label = f"K mvmt strength:{KSTR}  R2:{_analysis_utils.format_R2_str(R2)}"
        color = f / len(model_files)

        _analysis_utils.plot_curve_fit( ax, x, y, _curve_fit.mod_gompertz, fit_params, R2, run_model,
                                        color =  color,
                                        label = label,
                                        )

        # Calculate extinction metrics
        xff = np.geomspace(dat.K.min(), dat.K.max(), num=100000)
        yff = _curve_fit.mod_gompertz(xff, *fit_params)
        metrics = _analysis_utils.calculate_extinction_metrics(xff, yff, fit_params)
        
        # Create results DataFrame
        param_names = ("param_a", "param_b", "param_alpha")
        kX_names = [f"k{int(X*100)}" for X in np.arange(0.1, 1.0, 0.1)]
        
        ddf = pd.DataFrame()
        ddf.loc[0, ["model", "runName", "RMAX", "QSD", "QREV", "B", "SA", 
                    "year_threshold",
                    "model_name", *param_names, "R2", "RSD", "RMSE", "MAX_Y", 
                    *kX_names, "dPdK_tp"]] = [
            run_params['model'], run_params['runName'], run_params['rmax'], 
            run_params['qsd'], run_params['qrev'], run_params['b'], run_params['sa'],
            run_params['year_threshold'],
            fit_result['model_name'], *fit_params, R2, fit_result['rsd'], 
            fit_result['rmse'], max_y, 
            *[metrics[name] for name in kX_names], metrics['dPdK_tp']
        ]

        if not os.path.isdir(data_fits_path.parent):
            os.makedirs(data_fits_path.parent, exist_ok=True)
            
        data_fits = pd.concat([data_fits, ddf])

        data_fits.to_csv(data_fits_path, index=False)

        existing_runs.add(ddf.loc[0, 'runName'])

    ax.legend(fontsize = 9, ncol = 1)
    ax.set_title(f"K {KS}s")
    ax.set_ylabel("P(E)")
ax.set_xlabel("Initial K")


figs_dir.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(figs_dir / f"PE_vs_K_varying_K.png", dpi=300)
plt.show()

