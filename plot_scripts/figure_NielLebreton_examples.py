import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import _analysis_utils
from _figure_io import save_figure
import _overlay
# import _curve_fit

data_path = Path("..\\results\\simulation_results\\results_N_L_2005")
FIG_SUBDIR = "NielLebreton_examples"

bird_demo_df = pd.read_csv(Path("manuscript_inputs", "niel_lebreton2005_bird_demographics.csv"))
bird_growth_df = pd.read_csv(Path("manuscript_inputs", "niel_lebreton2005_bird_growth_rates.csv"))

num_sp = len(bird_demo_df.Species.unique())
cols = 5
rows = int(np.ceil(num_sp / cols))

fig, axs = plt.subplots(rows, cols, figsize=(8, 6),
                        # sharex=True, 
                        sharey=True)

panels = {}

for b, bird in enumerate(bird_demo_df.Species.unique()):

    print(bird)

    ax = axs.flatten()[b]

    bird_demo = bird_demo_df[bird_demo_df.Species == bird]
    bird_growth = bird_growth_df[bird_growth_df.Species == bird]
    
    bird_files = [ f for f in os.listdir(data_path) if bird in f]

    all_runs = []
    for f, file in enumerate(bird_files):

        dat = pd.read_csv(os.path.join(data_path, file))

        if np.max(dat.P) < 1:
            continue

        all_runs.append(dat.set_index("K")["P"])

    if len(all_runs) == 0:
        ax.set_title(bird)
        ax.set_xlabel("K")
        ax.set_ylabel("P(E)")
        continue

    runs_df = pd.concat(all_runs, axis=1)
    x = runs_df.index.values
    mean_y = 1 - runs_df.mean(axis=1).values
    std_y = runs_df.std(axis=1).values
    max_y = 1 - runs_df.min(axis=1).values
    min_y = 1 - runs_df.max(axis=1).values

    color = plt.cm.viridis((b) / len(bird_demo_df.Species.unique()))

    # kept for the "_wdots" pass: one row per individual simulation run, as
    # 1 - P to match the plotted quantity
    panels[b] = (ax, x, 1 - runs_df.to_numpy().T, bird)

    ax.fill_between(x, min_y, max_y, color=color, alpha=0.4)
    
    ax.set_xlim(x[runs_df.mean(axis=1).values > 0.0041].min(), 
                x[runs_df.mean(axis=1).values < 0.9998].max()
                )

    ax.plot(x, mean_y, color=color, linewidth=1.0)
    ax.set_title(bird)
    ax.set_xlabel("K")
    ax.set_ylabel("P(E)")
    _analysis_utils.ax_log2_scale(ax)

for ax in axs.flatten()[num_sp:]:
    ax.set_visible(False)

fig.tight_layout()
save_figure(fig, "niel_lebreton_examples", subdir=FIG_SUBDIR)

# --- "_wdots" variant: overlay every individual simulation run --------------
# The shaded band is the min-max envelope of exactly these runs, so the runs are
# drawn ON TOP of it (zorder 1.5) rather than buried underneath.  n is small
# (11-42 per species) so every run is shown and none is subsampled.
# Do NOT call tight_layout() again -- the n= labels sit inside the axes.
for b in sorted(panels):
    pax, px, curves, bird = panels[b]
    _, n_tot, n_shown = _overlay.faint_curves(
        pax, px, curves, cap=None, glyph=b,
        zorder=1.5, color="#333333", alpha=0.45, lw=0.3)
    # bottom-left: every curve starts at P(E)=1 on the left and falls to the
    # right, so that corner is the only one clear in all 14 panels
    _overlay.annotate_n_corner(pax, n_tot, n_shown, loc=(0.04, 0.05),
                               va="bottom", fontsize=6)
    print(f"{bird} : n_runs={n_tot}")

save_figure(fig, "niel_lebreton_examples" + _overlay.SUFFIX, subdir=FIG_SUBDIR)
plt.show()
