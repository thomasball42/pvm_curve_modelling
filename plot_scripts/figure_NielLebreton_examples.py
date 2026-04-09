import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import _analysis_utils
# import _curve_fit

data_path = Path("..\\results\\simulation_results\\results_N_L_2005")
figs_dir = Path("..", "figs", "figs_NielLebreton_examples")

bird_demo_df = pd.read_csv(Path("manuscript_inputs", "niel_lebreton2005_bird_demographics.csv"))
bird_growth_df = pd.read_csv(Path("manuscript_inputs", "niel_lebreton2005_bird_growth_rates.csv"))

num_sp = len(bird_demo_df.Species.unique())
cols = 5
rows = int(np.ceil(num_sp / cols))

fig, axs = plt.subplots(rows, cols, figsize=(8, 6),
                        # sharex=True, 
                        sharey=True)

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

    ax.fill_between(x, min_y, max_y, color=color, alpha=0.4)
    
    ax.set_xlim(x[runs_df.mean(axis=1).values > 0.0041].min(), 
                x[runs_df.mean(axis=1).values < 0.997].max()
                )

    ax.plot(x, mean_y, color=color, linewidth=1.0)
    ax.set_title(bird)
    ax.set_xlabel("K")
    ax.set_ylabel("P(E)")
    _analysis_utils.ax_log2_scale(ax)

for ax in axs.flatten()[num_sp:]:
    ax.set_visible(False)

fig.tight_layout()
figs_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(figs_dir / f"niel_lebreton_examples.png", dpi=300)
plt.show()
