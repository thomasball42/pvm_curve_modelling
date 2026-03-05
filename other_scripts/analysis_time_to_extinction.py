import matplotlib
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import re
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
import _curve_fit
import _analysis_utils

OVERWRITE = True
FIGSIZE = (8, 5)

### time to extinction vs K
DATA_FITS_PATH = Path("..", "results", "data_fits", "data_fits_tte")

figs_dir = Path("..", "figs", "figs_tte")
data_path = Path("..", "results", "simulation_results", "results_tte")

list_of_files = []
for path, subdirs, files in os.walk(data_path):
    for name in files:
        list_of_files.append(os.path.join(path, name)) 

files = [file for file in list_of_files if ".csv" in file and "LogGrowth" in file]

models = ["A", "B", "C", "D"]

fig, axs = plt.subplots(2,2, figsize=FIGSIZE, sharex=True, sharey=True)
fig2, axs2 = plt.subplots(figsize=FIGSIZE, sharex=True, sharey=True)

for m, run_model in enumerate(models):

    model_files = [file for file in files if f"LogGrowth{run_model}" in file]
    
    def extract_yt(path):
        match = re.search(r'YT(\d+)', path) # thanks Claude :)
        return int(match.group(1)) if match else -1
    
    model_files = sorted(model_files, key=extract_yt)

    ax = axs.flatten()[models.index(run_model)]

    ax2 = axs2
    # ax2 = axs2.flatten()[models.index(run_model)]
    
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

        if max_y != 1:
            continue 

        fit_result = _analysis_utils.fit_gompertz_curve(x, y, 
                                                alpha_space=np.arange(0, 1, 0.05), 
                                                ylim=(0.05, 0.95),
                                                iteration_depth= 4
                                                )
        
        fit_params = fit_result["params"]
        R2 = fit_result["R2"]
        
        YT = dat.YEAR_THRESHOLD.unique()[0]
        label = f"YT: {YT}"

        _analysis_utils.plot_curve_fit( ax, x, y, _curve_fit.mod_gompertz, fit_params, R2, run_model,
                                        color =  (YT - 1) / 250,
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

    ax.legend(fontsize = 7, ncol = 2)
    ax.set_title(f"Model {run_model}")

    yt_vals = data_fits.year_threshold.astype(float)
    r2_vals = data_fits.R2.astype(float)

    ## plot second ax
    ax2_color =  matplotlib.colormaps["viridis"]((m/4)+0.2)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        yt_vals, r2_vals
    )
    x_line = np.linspace(yt_vals.min(), yt_vals.max(), 200)
    y_line = slope * x_line + intercept
    ax2.plot(x_line, y_line, color=ax2_color, linewidth=1.5, linestyle="--", alpha = 0.5)

    ax2.scatter(yt_vals, r2_vals, 
                color = matplotlib.colormaps["viridis"]((m/4)+0.2),
                alpha = 0.9, 
                label = f"Model {run_model} linear fit R$^2$: {r_value**2:.4f}")

    ax2.ticklabel_format(axis='y', useOffset=False, style='plain')
    ax2.set_ylim(0.9999, 1.0)
    ax2.set_title(f"Model {run_model}")
    ax2.set_xlabel("Year Threshold")
    ax2.set_ylabel("R$^2$")

    ax.set_xlabel("K")
    ax.set_ylabel("P(E)")

ax2.legend(fontsize=11)

figs_dir.mkdir(parents=True, exist_ok=True)

fig.tight_layout()
fig2.tight_layout()

fig.savefig(figs_dir / "tte_PE_vs_K_varYT.png", dpi=300)
fig2.savefig(figs_dir / "R2_vs_YT.png", dpi=300)

plt.show()
