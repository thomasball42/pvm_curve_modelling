# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 17:27:04 2025

@author: tom
"""

import pandas as pd
import numpy as np
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from _figure_io import save_figure
import _overlay

FIG_SUBDIR = "gamma_dist"

# dir that the main data fits are in (same path as figure_gamma_ci.py)
dat_fits_path = Path("..", "results", "data_fits", "data_fits_main")

models = ["A", "B", "C", "D"]

fig, axs = plt.subplots()

gammas = {}

for m, model in enumerate(models):

    dat = pd.read_csv(dat_fits_path / f"data_fits_model_{model}.csv", index_col = 0)

    g = dat.param_alpha

    g = g[(~np.isnan(g)) & (dat.MAX_Y == 1)]
    g = g[g > 0]
    gammas[model] = g.to_numpy()

    axs.boxplot(g, positions = [m])
    print(f"{model} : n={len(g)}, median={np.median(g)}")
    # mean = g.mean()
    # stde = 3*np.std(g) / np.sqrt(len(g))
    # axs.scatter(m, mean, marker = "x", color = "k")
    # axs.errorbar(m, mean, yerr = stde, color = "k")
    
axs.set_ylim(-0.5, 1.25)
axs.set_xticks(np.arange(len(models)), labels = [f"Model_{x}" for x in models])
axs.axhline(1, linestyle = "--", color = "k", alpha = 0.5)
axs.set_ylabel("Gamma parameter")
fig.tight_layout()
save_figure(fig, "gamma_dist", subdir=FIG_SUBDIR)

# --- "_wdots" variant: overlay the individual fitted gamma values -----------
# Boxplot default widths are 0.5 and positions are 0..3 on a linear axis.
# The original ylim of (-0.5, 1.25) clips the upper tail, so extend it here so
# that every overlaid point is actually visible; this block runs after the
# original has been written, so that file is unaffected.  Do NOT call
# tight_layout() again -- the n= labels are placed inside the axes.
axs.set_ylim(-0.5, 1.45)

for m, model in enumerate(models):
    _, n_tot, n_shown = _overlay.jittered_dots(axs, m, gammas[model],
                                               width=0.5, glyph=m)
    _overlay.annotate_n(axs, m, n_tot, n_shown)
    clipped = _overlay.count_clipped(axs, gammas[model])
    print(f"{model} : n={n_tot}, shown={n_shown}, clipped={clipped}")

save_figure(fig, "gamma_dist" + _overlay.SUFFIX, subdir=FIG_SUBDIR)