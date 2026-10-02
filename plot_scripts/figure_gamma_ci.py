# -*- coding: utf-8 -*-
"""
Distribution of the gamma confidence interval widths, expressed as a
percentage of the fitted gamma, for each of the four main models.

@author: Thomas Ball
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker

sys.path.insert(0, str(Path(__file__).parent.parent))
from _figure_io import save_figure
import _overlay

FIG_SUBDIR = "gamma_ci"

model_names = ["model_A", "model_B", "model_C", "model_D"]

# dir that the main data fits are in
dat_fits_path = Path("..", "results", "data_fits", "data_fits_main")

ci_widths = {}

for model_name in model_names:

    path = dat_fits_path / f"data_fits_{model_name}.csv"
    df = pd.read_csv(path, index_col=0).dropna(
        subset=["param_alpha", "alpha_ci_025", "alpha_ci_975"])
    df = df[df.MAX_Y > 0.999]
    df = df[df.param_alpha > 0]

    # CI width relative to the point estimate of gamma
    width = 100 * (df.alpha_ci_975 - df.alpha_ci_025) / df.param_alpha
    width = width[np.isfinite(width)]
    ci_widths[model_name] = width.to_numpy()

    q25, q50, q75 = np.percentile(width, [25, 50, 75])
    print(f"{model_name} : n={len(width)}, median={q50:.2f}%, "
          f"IQR=[{q25:.2f}, {q75:.2f}]%, max={width.max():.1f}%")

fig, axs = plt.subplots()

positions = np.arange(len(model_names))
data = [ci_widths[m] for m in model_names]

vp = axs.violinplot(
    [np.log10(d) for d in data],
    positions=positions,
    widths=0.7,
    showmeans=False,
    showmedians=False,
    showextrema=False,
)
for body in vp["bodies"]:
    body.set_facecolor(plt.get_cmap('viridis')(0.3))
    body.set_edgecolor("#555555")
    body.set_alpha(0.4)
    body.set_linewidth(1.0)

axs.boxplot([np.log10(d) for d in data], positions=positions, widths=0.15,
            showfliers=False, medianprops={"color": "k"})

n_texts = {}
for m, model_name in enumerate(model_names):
    med = np.median(ci_widths[model_name])
    axs.text(m + 0.12, np.log10(med), f"{med:.1f}%", ha="left", va="center", fontsize=9)
    # handle is kept so the "_wdots" pass can extend this label in place rather
    # than adding a second one; the text written here is unchanged
    n_texts[m] = axs.text(m, 0.02, f"n={len(ci_widths[model_name])}",
                          transform=axs.get_xaxis_transform(),
                          ha="center", va="bottom", fontsize=8, color="#555555")

axs.axhline(np.log10(10), linestyle="--", color="k", alpha=0.5)

axs.set_xticks(positions, labels=[f"Model {m.split('_')[-1]}" for m in model_names])
axs.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda y, pos: f"{10**y:g}"))
axs.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
axs.set_ylabel(r"95% CI width as % of $\gamma$")

fig.set_size_inches(6, 4.5)
fig.tight_layout()
save_figure(fig, "gamma_ci", subdir=FIG_SUBDIR)

# --- "_wdots" variant: overlay the individual CI widths ---------------------
# The violin and the box are both built from log10 of the data, so the overlaid
# values have to be log10 too.  Violin widths are 0.7 on a linear position axis.
# Do NOT call tight_layout() again -- the n= labels sit inside the axes.
for m, model_name in enumerate(model_names):
    _, n_tot, n_shown = _overlay.jittered_dots(
        axs, m, np.log10(ci_widths[model_name]), width=0.7, glyph=m)
    n_texts[m].set_text(_overlay.n_text(n_tot, n_shown))
    n_texts[m].set_fontsize(7)
    print(f"{model_name} : n={n_tot}, shown={n_shown}")

save_figure(fig, "gamma_ci" + _overlay.SUFFIX, subdir=FIG_SUBDIR)
