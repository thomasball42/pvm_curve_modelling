import matplotlib
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import re
from scipy import stats
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent.parent))
import _curve_fit
import _analysis_utils

OVERWRITE = True
FIGSIZE = (8, 5)

# ── Fit function: easy to swap ──────────────────────────────────────────────
# Current: y = c * x^a  (two-param power law)
# To use a single-param version just change to: lambda x, a: x**a
def FIT_FUNC(x, a, c):
    return c * x ** a

FIT_P0 = [1.0, 1.0]           # initial guesses, match length to FIT_FUNC params
FIT_LABEL = r"$c \cdot x^a$"  # shown in panel titles
# ────────────────────────────────────────────────────────────────────────────

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

fig, axs = plt.subplots(2, 2, figsize=FIGSIZE, sharey=True)

for m, run_model in enumerate(models):

    model_files = [file for file in files if f"LogGrowth{run_model}" in file]

    def extract_yt(path):
        match = re.search(r'YT(\d+)', path)
        return int(match.group(1)) if match else -1

    model_files = sorted(model_files, key=extract_yt)

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
        mean_tte = dat.mean_TTE

        mask = y == 0

        x_scatter = x[mask].values
        y_scatter = mean_tte[mask].values
        y_scatter_err = dat.mean_TTE_SEM[mask].values

        YT = run_params['year_threshold']
        color = f"C{f}"

        # ── Fit FIT_FUNC to the P==0 data ───────────────────────────────────
        label = f"YT: {YT}"
        if len(x_scatter) >= len(FIT_P0):
            try:
                popt, _ = curve_fit(FIT_FUNC, x_scatter, y_scatter, p0=FIT_P0,
                                    maxfev=10000)

                y_pred = FIT_FUNC(x_scatter, *popt)
                ss_res = np.sum((y_scatter - y_pred) ** 2)
                ss_tot = np.sum((y_scatter - np.mean(y_scatter)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

                sem = dat.mean_TTE_SEM[mask]
                ss_noise = np.sum(sem ** 2)
                ss_tot = np.sum((y_scatter - np.mean(y_scatter)) ** 2)
                r2_ceiling = 1 - ss_noise / ss_tot

                # Plot fit line
                x_fit = np.linspace(x_scatter.min(), x_scatter.max(), 300)
                y_fit = FIT_FUNC(x_fit, *popt)
                ax.plot(x_fit, y_fit, color=color, linewidth=1.2)

                label = f"(fit: {FIT_LABEL}) $R^2$={_analysis_utils.format_R2_str(r2)} (ceil: {_analysis_utils.format_R2_str(r2_ceiling)})"

            except (RuntimeError, ValueError):
                pass  # fit failed – scatter still plotted, just no fit line

        # ax.scatter(x_scatter, y_scatter, label=label, color=color, s=18, zorder=3)
        ax.errorbar(x_scatter, y_scatter, yerr=y_scatter_err, fmt="o",
                    label=label, color=color, markersize=5, zorder=3)

        # ────────────────────────────────────────────────────────────────────

        max_y = np.max(y)
        if max_y != 1:
            continue

        fit_result = _analysis_utils.fit_gompertz_curve(
            x, y,
            alpha_space=np.arange(0, 1, 0.05),
            ylim=(0.05, 0.95),
            iteration_depth=4
        )

        N = run_params['n']
        fit_params = fit_result["params"]
        R2 = fit_result["R2"]

        xff = np.geomspace(dat.K.min(), dat.K.max(), num=100000)
        yff = _curve_fit.mod_gompertz(xff, *fit_params)
        kXs = list(np.arange(0.1, 1.0, 0.1))

        metrics = _analysis_utils.calculate_extinction_metrics(xff, yff, fit_params, kXs=kXs)

        param_names = ("param_a", "param_b", "param_alpha")
        kX_names = [f"k{int(X*100)}" for X in kXs]

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

    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"Model {run_model}")
    ax.set_xlabel("K")
    ax.set_ylabel(f"Mean time to extinction (n={N})")

figs_dir.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(figs_dir / "K_vs_TTE.png", dpi=300)
plt.show()