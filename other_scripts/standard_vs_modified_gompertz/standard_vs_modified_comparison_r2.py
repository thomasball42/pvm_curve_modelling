# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 15:08:18 2025

@author: Thomas Ball

Produces the per-model comparison of fit quality between the modified and the
standard (unmodified) Gompertz curve: the distribution of R2 differences (left)
against the inflection point of the fitted curve (right).

Run from the pvm_curve_modelling directory, as with the other plot scripts.
"""

import sys
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path
# this script sits two levels below the package root
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
# _overlay lives alongside the figure scripts, which are not on the path when
# this script is run from its own directory
sys.path.insert(0, str(_root / "plot_scripts"))
from _figure_io import save_figure
import _overlay

FIG_SUBDIR = "stand_vs_mod_gompertz"

# dir that the main data fits are in
data_fits_path = Path("..", "results", "data_fits", "data_fits_main")

models = ["A", "B", "C", "D"]
var = "R2"

for m, model_name in enumerate(models):

    df_bas = pd.read_csv(
        data_fits_path / f"data_fits_{model_name}_basic_gompertz.csv", index_col=0)
    df_mod = pd.read_csv(
        data_fits_path / f"data_fits_model_{model_name}.csv", index_col=0)

    # Join on runName rather than on position: model A has 255 basic fits but
    # only 254 modified ones, so assigning the columns across by index silently
    # paired up mismatched runs.
    df = df_bas.merge(df_mod, on="runName", suffixes=("_bas", "_mod"))

    mod_min = df_mod[var].min()
    perc_outside_range = (df_bas[var] < mod_min).mean()

    # MAX_Y is taken from the basic fits, as before
    df = df[df.MAX_Y_bas > 0.9999]

    df[f"{var}_diff"] = df[f"{var}_mod"] - df[f"{var}_bas"]
    df["inflection"] = df["dPdK_tp_bas"]

    dat = df[f"{var}_diff"].to_numpy()
    dat = dat[~np.isnan(dat)]

    print(f"Model {model_name}: merged {len(df_bas.merge(df_mod, on='runName'))} runs, "
          f"{len(df)} after MAX_Y filter, {len(dat)} with non-NaN {var} difference; "
          f"min modified {var}={mod_min:.4f}, "
          f"{100 * perc_outside_range:.1f}% of basic fits fall below it; "
          f"mean difference={np.mean(dat):.4f}")

    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [0.8, 1.8]},
                            sharey=True)

    axs[0].boxplot(dat)
    axs[0].set_xticks([])
    sc = axs[1].scatter(df["inflection"], df[f"{var}_diff"],
                        color="k", alpha=0.4, label=f"Model {model_name}")
    axs[0].set_ylabel(f"Absolute difference in {var}")
    axs[1].set_xlabel("Inflection point (K)")
    axs[1].legend()
    fig.tight_layout()

    # name must not collide with standard_vs_mod_gompertz_r2, which is written
    # into this same subdir by figure_standard_vs_modified_gompertz_r2.py
    stem = f"standard_vs_mod_gompertz_{var}_diff_model_{model_name}"
    save_figure(fig, stem, subdir=FIG_SUBDIR)

    # --- "_wdots" variant --------------------------------------------------
    # The right panel already shows every individual point, so it needs only
    # the n; the left panel is the box plot that needs the dot overlay.
    # boxplot() here uses its default position of 1 and default width of 0.5.
    _, n_tot, n_shown = _overlay.jittered_dots(axs[0], 1, dat, width=0.5, glyph=m)
    _overlay.annotate_n(axs[0], 1, n_tot, n_shown)
    sc.set_label(f"Model {model_name} (n={n_tot:,})")
    axs[1].legend()
    print(f"  overlay: n={n_tot}, shown={n_shown}")

    save_figure(fig, stem + _overlay.SUFFIX, subdir=FIG_SUBDIR)
    plt.close(fig)
