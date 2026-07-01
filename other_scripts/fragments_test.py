import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import random

sys.path.append(str(Path(__file__).resolve().parents[1]))
from other_scripts import process_mammal_data
import _curve_fit
import _analysis_utils 

FRAGMENT_SPACE = sorted(set([int(_) for _ in np.geomspace(1, 3000, num = 15)]))

cmap = plt.get_cmap("viridis")

model_names = [
                # "model_A", 
                # "model_B", 
                # "model_C", 
                "model_D"
                ]

number_of_closest_X = 4

figs_dir = Path("..", "figs_non_ms", "figs_fragment_test")

all_data_fits = {}
for model_name in model_names:
    path = Path("..", "results", "data_fits", "data_fits_main",
                f"data_fits_{model_name}.csv")
    df = pd.read_csv(path, index_col=0).dropna(
        subset=["param_a", "param_b", "param_alpha"]
    )
    all_data_fits[model_name] = df[(df.MAX_Y == 1) & (df.R2 > 0.9999)]


def fragment_gompertz(x, a, b, alpha, fragments):
    return (1 - _curve_fit.mod_gompertz(x/fragments, a, b, alpha)) ** fragments


fig, axs = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)

for model_name, df in all_data_fits.items():
    mean_alpha = df.param_alpha.mean()
    # closest_X = df.iloc[(df.param_alpha - mean_alpha).abs().argsort()[:number_of_closest_X]]

    pool = df.iloc[(df.param_alpha - mean_alpha).abs().argsort()[:50]].sort_values("param_a")
    closest_X = pool.iloc[np.linspace(0, len(pool) - 1, number_of_closest_X, dtype=int)]

    for i, (idx, row) in enumerate(closest_X.iterrows()):

        print(i)
        ax = axs.flatten()[i]
        ax = _analysis_utils.ax_log2_scale(ax)

        params = row[["param_a", "param_b", "param_alpha"]].values
        
        print(params, row.R2)
        xff = np.geomspace(1, 3000000, num=1000)

        for f, fragments in enumerate(FRAGMENT_SPACE):

            yff = fragment_gompertz(xff, *params, fragments=fragments)
            color = cmap(f / (len(FRAGMENT_SPACE) - 1))

            label = f"gamma={round(row.param_alpha, 3)},\nsigma={round(row.QSD, 2)},rmax={round(row.RMAX, 2)}"
            
            ax.plot(xff, yff, color=color, alpha=1, label = f"F={fragments}")

        model_legend = [Patch(facecolor="white", alpha=0.0, label=label)]

        all_handles = model_legend  + ax.get_legend_handles_labels()[0]

        ax.legend(handles=all_handles, fontsize=8,
                  ncols=2)

        ax.set_xlabel("K")
        ax.set_ylabel("P(E)")

fig.tight_layout()
figs_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(figs_dir / f"fragment_examples.png", dpi=300)
plt.show()
