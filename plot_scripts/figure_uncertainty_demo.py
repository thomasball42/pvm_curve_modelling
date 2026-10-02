"""TB 26. Note Fig 2 takes a while to generate"""

from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import _analysis_utils
import _curve_fit
from _figure_io import save_figure
import _overlay

data_path = Path("..\\results\\simulation_results\\results_main")
FIG_SUBDIR = "uncertainty"

list_of_files = []
for path, subdirs, files in os.walk(data_path):
    for name in files:
        list_of_files.append(os.path.join(path, name))


## fig 1
example_files = ["LogGrowthA_QSD0.17_YT100_RMAX0.158.csv",
                 "LogGrowthB_QSD0.17_YT100_RMAX0.158.csv",
                 "LogGrowthC_QSD0.17_YT100_RMAX0.158_SA0.5.csv",
                 "LogGrowthD_QSD0.17_YT100_RMAX0.158_SA0.5_QREV0.505.csv"
                 ]

fig, ax = plt.subplots(figsize=(7, 6))

# inset ax
ax_inset = zoomed_inset_axes(ax, zoom=2.6, loc="center right")

for f, file in enumerate(example_files):

    dat = pd.read_csv(os.path.join(data_path, file))
    run_params = _analysis_utils.extract_run_parameters(dat)
    run_name = run_params["runName"]

    x = dat.K
    y = dat.P
    yerr = dat.P_SEM

    max_y = np.max(y)

    fit_result = _analysis_utils.fit_gompertz_curve(x, y,
                                            alpha_space=np.arange(0, 1, 0.05),
                                            ylim=(0.05, 0.95),
                                            iteration_depth=4
                                            )

    fit_params = fit_result["params"]
    R2 = fit_result["R2"]

    color = (f) / len(example_files)

    _analysis_utils.plot_curve_fit( ax, x, y, _curve_fit.mod_gompertz, fit_params, R2, run_name.split("_")[0],
                                        color =  color,
                                        label = None
                                        )
    
    # Plot on main ax
    ax.errorbar(x, 1-y, yerr=yerr, fmt="o",
                markersize=0,
                color=plt.cm.viridis(color), label=f"{run_name.split("_")[0]} (R2={_analysis_utils.format_R2_str(R2)})")

    # Plot the SAME data on the inset ax
    ax_inset.errorbar(x, 1-y, yerr=yerr, fmt="o",
                      markersize=2,
                      color=plt.cm.viridis(color))
    _analysis_utils.plot_curve_fit( ax_inset, x, y, _curve_fit.mod_gompertz, fit_params, R2, run_name.split("_")[0],
                                        color =  color,
                                        label = None,
                                        alpha = 0.35
                                        )
    
    xff = np.geomspace(dat.K.min(), dat.K.max(), num=100000)
    yff = _curve_fit.mod_gompertz(xff, *fit_params)
    metrics = _analysis_utils.calculate_extinction_metrics(xff, yff, fit_params)

    _analysis_utils.ax_log2_scale(ax)

x_mid_min = 50
x_mid_max = 230  
y_mid_min = 0.32  
y_mid_max = 0.58  

ax_inset.set_xlim(x_mid_min, x_mid_max)
ax_inset.set_ylim(y_mid_min, y_mid_max)

_analysis_utils.ax_log2_scale(ax_inset)

ax_inset.tick_params(labelleft=True, labelbottom=True)

mark_inset(ax, ax_inset, loc1=2, loc2=3, fc="none", ec="0.5", lw=0.8, linestyle="--")

ax.legend()

ax.set_ylabel("Probability of extinction P(E) (n = 10000)")
ax.set_xlabel("Carrying Capacity K")

fig.tight_layout()
save_figure(fig, "uncertainty_examples", subdir=FIG_SUBDIR)

# fig 2

models = [  "LogGrowthA", 
            "LogGrowthB",
            "LogGrowthC",
            "LogGrowthD"]

fig, axs = plt.subplots(2,2, figsize=(7, 6))

panels = {}

for m, model in enumerate(models):

    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    binned_data = [[] for _ in range(n_bins)]
    ax = axs.flatten()[models.index(model)]

    model_files = [file for file in list_of_files if model in file]

    for f, file in enumerate(model_files):

        print(f/ len(model_files), end="\r")
        dat = pd.read_csv(file)

        run_params = _analysis_utils.extract_run_parameters(dat)
        run_name = run_params["runName"]
        model_name = run_params["model"]

        x = dat.K
        y = dat.P

        y_index = (dat.P > 0) & (dat.P < 1)

        yy = dat.P[y_index]
        xx = dat.K[y_index]
        yyerr = dat.P_SEM[y_index]

        K_norm = (xx - xx.min()) / (xx.max() - xx.min())

        bin_indices = np.digitize(K_norm, bins) - 1  
        bin_indices = np.clip(bin_indices, 0, n_bins - 1) 

        for i, val in zip(bin_indices, 100 * yyerr / yy):
            binned_data[i].append(val)

        ax.scatter(K_norm, 100 * yyerr/yy, s = 1, alpha =0.1, color = "steelblue")

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    ax.boxplot(
        binned_data,
        positions=bin_centers,
        widths=(bins[1] - bins[0]) * 0.8,
        patch_artist=True,
        boxprops=dict(facecolor='steelblue', alpha=0.6),
        medianprops=dict(color='black'),
        flierprops=dict(marker='o', markersize=0, alpha=0.4),
        manage_ticks=False
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 13)
    ax.set_xlabel("K (normalised)")
    ax.set_ylabel("%uncertainty in P")

    legend_patch = [
        Patch(facecolor="white", alpha=0.0, label=f"{model.replace('LogGrowth', 'Model ')}")
        ]
    ax.legend(handles=legend_patch, fontsize=10)

    panels[m] = (ax, bin_centers, [len(b) for b in binned_data], legend_patch)

fig.tight_layout()
save_figure(fig, "uncertainty_percentage_distribution", subdir=FIG_SUBDIR)

# --- "_wdots" variant: per-bin n for the raw point cloud --------------------
# Every individual value is ALREADY plotted as the steelblue scatter above, so
# nothing is added here; what was missing is the sample size behind each box.
# The scatters are rasterized so the vector file stays workable (the all-vector
# version is ~14 MB).  boxplot(patch_artist=True) puts its boxes in ax.patches
# and its whiskers in ax.lines, so ax.collections holds ONLY those scatters.
# use_raster_dpi() is required because save_figure() does not pass dpi to the
# PDF.  Do NOT call tight_layout() again.
_overlay.use_raster_dpi()
for m, model in enumerate(models):
    pax, centres, bin_ns, patch = panels[m]
    for coll in pax.collections:
        coll.set_rasterized(True)
    # ten bins across a ~3.4 inch panel leaves ~8.6 mm each, so the counts have
    # to be set vertically.  No data falls below zero, so drop the lower limit
    # to open up a clear band for them rather than running them over the boxes.
    pax.set_ylim(-3.2, 13)
    # the original legend uses loc="best", which relocates into that new band
    # once the limits change -- pin it where it sits in the original
    pax.legend(handles=patch, fontsize=10, loc="upper right")
    for centre, n_bin in zip(centres, bin_ns):
        _overlay.annotate_n(pax, centre, n_bin, fontsize=5, rotation=90)
    print(f"{model}: total n={sum(bin_ns)}, per-bin {bin_ns}")

save_figure(fig, "uncertainty_percentage_distribution" + _overlay.SUFFIX,
            subdir=FIG_SUBDIR)

# plt.show()
