import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.append(str(Path(__file__).resolve().parents[1]))
from other_scripts import process_mammal_data
import _curve_fit
import _analysis_utils

data_fits_path = Path("..", "results", "data_fits", "data_fits_main")
figs_dir = Path("..", "figs", "figs_stand_vs_mod_gompertz")
models = ["A", 
          "B", 
          "C", 
          "D"
        ]

files = data_fits_path.glob("*.csv")

# fig, axs = plt.subplots(2, 2, figsize=(8, 6),
#                         sharey="row",
#                         sharex=True)

# for m, model in enumerate(models):

#     ax = axs.flatten()[m]

#     df_mod = pd.read_csv(data_fits_path / f"data_fits_model_{model}.csv")
#     df_bas = pd.read_csv(data_fits_path / f"data_fits_{model}_basic_gompertz.csv")

#     df = df_mod.merge(df_bas, on="runName", suffixes=("_mod", "_bas"))
#     df = df[df.MAX_Y_mod == 1]
#     xff = np.unique(np.round(np.geomspace(1, 3E6, num=3000)))

#     perc_diffs = []
#     abs_diffs = []

#     for i, (idx, row) in enumerate(df.iterrows()):
        
#         print(i/len(df))
#         mod_params = row[["param_a", "param_b", "param_alpha"]]
#         bas_params = row[["bg_param_a", "bg_param_b"]]

#         yff_mod = 1 - _curve_fit.mod_gompertz(xff, *mod_params)
#         yff_bas = 1 - _curve_fit.basic_gomp(xff, *bas_params)

#         # y_index = (dat.P > 0) & (dat.P < 1)

#         # yy = dat.P[y_index]
#         # xx = dat.K[y_index]
#         # yyerr = dat.P_SEM[y_index]

#         # K_norm = (xx - xx.min()) / (xx.max() - xx.min())

#         abs_diff = np.abs(yff_mod - yff_bas)
#         abs_diff = np.where(yff_mod > 0, abs_diff, 0)
#         abs_diffs.append(abs_diff)
#         perc_diff = np.where(yff_mod > 0, abs_diff / yff_mod, 0)
#         perc_diffs.append(perc_diff)

#     mean_perc_diff = np.nanmean(perc_diffs, axis=0)
#     mean_perc_diff_std = np.nanstd(perc_diffs, axis=0)

#     mean_abs_diff = np.nanmean(abs_diffs, axis=0)
#     mean_abs_diff_std = np.nanstd(abs_diffs, axis=0)

#     ax.scatter(xff, mean_perc_diff, label=f"Model {model}", color=f"C{m}")#

#     ax = _analysis_utils.ax_log2_scale(ax)
    
#     ax.set_xlabel("K")
#     ax.set_ylabel("Mean % difference in P" if m in [0, 2] else None)

#     legend_patch = [
#         Patch(facecolor="white", alpha=0.0, label=f"Model {model}")
#         ]
#     ax.legend(handles=legend_patch, fontsize=10)

# fig.tight_layout()
# figs_dir.mkdir(parents=True, exist_ok=True)
# fig.savefig(figs_dir / f"standard_vs_mod_gompertz_PE.png", dpi=300)
# plt.show()

# ── Second figure: % difference vs normalised x (clipped to y in (0,1)) ──────
fig2, axs2 = plt.subplots(2, 2, figsize=(8, 6),
                           sharey="row",
                           sharex=True)

N_NORM = 200  # resolution of the common normalised x grid
x_norm_grid = np.linspace(0, 1, N_NORM)

xff = np.unique(np.round(np.geomspace(1, 3E6, num=3000)))

for m, model in enumerate(models):

    ax = axs2.flatten()[m]

    df_mod = pd.read_csv(data_fits_path / f"data_fits_model_{model}.csv")
    df_bas = pd.read_csv(data_fits_path / f"data_fits_{model}_basic_gompertz.csv")

    df = df_mod.merge(df_bas, on="runName", suffixes=("_mod", "_bas"))
    df = df[df.MAX_Y_mod == 1]

    perc_diffs_norm = []

    for i, (idx, row) in enumerate(df.iterrows()):

        print(f"Model {model} – {i / len(df):.1%}")

        mod_params = row[["param_a", "param_b", "param_alpha"]]
        bas_params = row[["bg_param_a", "bg_param_b"]]

        yff_mod = 1 - _curve_fit.mod_gompertz(xff, *mod_params)
        yff_bas = 1 - _curve_fit.basic_gomp(xff, *bas_params)

        thresh = 0.99
        valid = (yff_mod > 1-thresh) & (yff_mod < thresh)
        
        if valid.sum() < 2:
            continue

        x_valid     = xff[valid]
        y_mod_valid = yff_mod[valid]
        y_bas_valid = yff_bas[valid]

        x_min, x_max = x_valid.min(), x_valid.max()
        x_norm = (x_valid - x_min) / (x_max - x_min)

        perc_diff = np.where(y_mod_valid > 0,
                             100* (y_mod_valid - y_bas_valid) / y_mod_valid,
                             np.nan)

        perc_diff_interp = np.interp(x_norm_grid, x_norm, perc_diff)
        perc_diffs_norm.append(perc_diff_interp)

    if not perc_diffs_norm:
        continue

    perc_diffs_norm = np.array(perc_diffs_norm)

    mean_pd  = np.nanmean(perc_diffs_norm, axis=0)
    std_pd   = np.nanstd(perc_diffs_norm,  axis=0)

    ax.plot(x_norm_grid, mean_pd, color=f"C{m}", lw=1.5)
    ax.fill_between(x_norm_grid,
                    mean_pd - std_pd,
                    mean_pd + std_pd,
                    color=f"C{m}", alpha=0.25)

    ax.set_xlabel("Normalised K  (0 = P(E)→1,  1 = P(E)→0)")
    ax.set_ylabel("Mean % difference in P" if m in [0, 2] else None)
    ax.set_xlim(0, 1)

    legend_patch = [Patch(facecolor="white", alpha=0.0, label=f"Model {model} (n = {len(perc_diffs_norm)})")]
    ax.legend(handles=legend_patch, fontsize=10)

fig2.tight_layout()
fig2.savefig(figs_dir / "gross_standard_vs_mod_gompertz_PE_norm_K.png", dpi=300)
plt.show()

fig2, axs2 = plt.subplots(2, 2, figsize=(8, 6),
                           sharey="row",
                           sharex=True)

N_NORM = 200  # resolution of the common normalised x grid
x_norm_grid = np.linspace(0, 1, N_NORM)

xff = np.unique(np.round(np.geomspace(1, 3E6, num=3000)))

for m, model in enumerate(models):

    ax = axs2.flatten()[m]

    df_mod = pd.read_csv(data_fits_path / f"data_fits_model_{model}.csv")
    df_bas = pd.read_csv(data_fits_path / f"data_fits_{model}_basic_gompertz.csv")

    df = df_mod.merge(df_bas, on="runName", suffixes=("_mod", "_bas"))
    df = df[df.MAX_Y_mod == 1]

    perc_diffs_norm = []

    for i, (idx, row) in enumerate(df.iterrows()):

        print(f"Model {model} – {i / len(df):.1%}")

        mod_params = row[["param_a", "param_b", "param_alpha"]]
        bas_params = row[["bg_param_a", "bg_param_b"]]

        yff_mod = 1 - _curve_fit.mod_gompertz(xff, *mod_params)
        yff_bas = 1 - _curve_fit.basic_gomp(xff, *bas_params)

        thresh = 0.99
        valid = (yff_mod > 1-thresh) & (yff_mod < thresh)
        
        if valid.sum() < 2:
            continue

        x_valid     = xff[valid]
        y_mod_valid = yff_mod[valid]
        y_bas_valid = yff_bas[valid]

        x_min, x_max = x_valid.min(), x_valid.max()
        x_norm = (x_valid - x_min) / (x_max - x_min)

        perc_diff = np.where(y_mod_valid > 0,
                             100* np.abs(y_mod_valid - y_bas_valid) / y_mod_valid,
                             np.nan)

        perc_diff_interp = np.interp(x_norm_grid, x_norm, perc_diff)
        perc_diffs_norm.append(perc_diff_interp)

    if not perc_diffs_norm:
        continue

    perc_diffs_norm = np.array(perc_diffs_norm)

    mean_pd  = np.nanmean(perc_diffs_norm, axis=0)
    std_pd   = np.nanstd(perc_diffs_norm,  axis=0)

    ax.plot(x_norm_grid, mean_pd, color=f"C{m}", lw=1.5)
    ax.fill_between(x_norm_grid,
                    mean_pd - std_pd,
                    mean_pd + std_pd,
                    color=f"C{m}", alpha=0.25)

    ax.set_xlabel("Normalised K  (0 = P(E)→1,  1 = P(E)→0)")
    ax.set_ylabel("Absolute mean % difference in P" if m in [0, 2] else None)
    ax.set_xlim(0, 1)

    legend_patch = [Patch(facecolor="white", alpha=0.0, label=f"Model {model} (n = {len(perc_diffs_norm)})")]
    ax.legend(handles=legend_patch, fontsize=10)

fig2.tight_layout()
fig2.savefig(figs_dir / "standard_vs_mod_gompertz_PE_norm_K.png", dpi=300)
plt.show()