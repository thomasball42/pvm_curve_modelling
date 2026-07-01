# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:31:42 2025
Refactored to multi-panel plot (one panel per result file)

@author: Thomas Ball
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import math

import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib

sys.path.insert(0, str(Path(__file__).parent.parent))
import _analysis_utils  # noqa: F401
import _curve_fit


# dir that the simulation outputs are in
dat_path = "..\\results\\simulation_results\\results_curve_comparison"

# collect files
f = []
for path, subdirs, files in os.walk(dat_path):
    for name in files:
        f.append(os.path.join(path, name))

# optional: stable ordering
f = sorted(f)


def n_decimals_for_r2(r2):
    """Choose decimals so values very close to 1 are still distinguishable."""
    if pd.isna(r2):
        return 2
    n = 0
    while n < 10 and round(r2, n) == 1:
        n += 1
    return n + 1


def gompertz(x, a, b, c):
    return a * np.exp(-b * np.exp(-c * x))


def powerX2(x, x0, z0):
    return z0 * ((x - x0) ** 0.25)


def lande(x, a, b0, c0):
    """
    P(EXT) = (a / K)^( Abs[(rmax - s/2)/s] - (d - s/2)/s )
    """
    poww = (np.abs((b0 - (0.5 * c0)) / c0) + ((b0 - (0.5 * c0)) / c0))
    y = 1 - (a / x) ** poww
    y = np.asarray(y, dtype=float)
    y[y < 0] = np.nan
    return y


def plot_one_file(ax, filepath, i):
    dat = pd.read_csv(filepath)

    if dat.empty:
        ax.set_title(f"{Path(filepath).name}\n(empty file)")
        ax.axis("off")
        return

    # pull metadata
    runName = dat.runName.unique().item() if "runName" in dat else Path(filepath).stem
    model = runName.split("_")[0] if isinstance(runName, str) and "_" in runName else str(runName)
    qsd = dat.QSD.unique().item() if "QSD" in dat else np.nan
    rmax = dat.RMAX.unique().item() if "RMAX" in dat else np.nan
    N = dat.N.unique().item() if "N" in dat else np.nan

    sa = dat.SA.unique().item() if "SA" in dat else np.nan
    qrev = dat.QREV.unique().item() if "QREV" in dat else np.nan

    x = dat.K
    y = dat.P

    # initialise fitting
    fit = False
    model_name = np.nan
    R2 = np.nan
    params = (np.nan, np.nan, np.nan)

    # TRY MODIFIED GOMPERTZ
    func = _curve_fit.mod_gompertz
    ret = _curve_fit.betterfit_gompertz(
        func, x, y,
        alpha_space=np.arange(0, 5, 0.001),
        ylim=(0.05, 0.95),
        plot_lins=False,
    )
    if ret is None:
        ret = _curve_fit.betterfit_gompertz(
            func, x, y,
            alpha_space=np.arange(-5, 0, 0.001),
            ylim=(0.05, 0.95),
            plot_lins=False,
        )

    if (not fit) and (ret is not None):
        fit = True
        params, y_predicted, R2, resids = ret
        model_name = func.__name__

    dat_offset = 0
    do2 = 0
    ygt0 = (y > 0).argmax()
    ylt1 = (y < 1).argmin()

    lo = ygt0 - dat_offset - do2 if dat_offset + do2 < ygt0 else 0
    hi = ylt1 + dat_offset if ylt1 + dat_offset < len(y) else len(y)

    yy = y[lo:hi]
    xx = x[yy.index].to_numpy()

    if len(xx) == 0:
        ax.set_title(f"{Path(filepath).name}\n(no valid plotting data)")
        ax.axis("off")
        return

    kwargs = {"linewidth": 2, "alpha": 0.99, "linestyle": "-"}

    # line colors
    line_colors = {
        "std_gompertz": "#648FFF",  
        "mod_gompertz": "#785EF0",   
        "logistic":     "#DC267F",  
        "power":        "#FE6100",  
        "lande":        "#FFB000",  
    }

    # plot data points
    ax.scatter(xx, 1 - yy, color="k", alpha=0.4, marker="o", s=30)

    xff = np.geomspace(max(np.min(xx), 1e-12), np.max(xx), num=2000)

    # Standard gompertz
    try:
        params_g, y_predicted_g, R2_g, residuals_g = _curve_fit.fit(
            gompertz, x, y, init_guess=[1, 0.5, 0.01]
        )
        label = f"Std Gompertz (R²:{round(R2_g, n_decimals_for_r2(R2_g))})"
        ax.plot(xff, 1 - gompertz(xff, *params_g), color=line_colors["std_gompertz"], label=label, **kwargs)
    except Exception:
        pass

    # Modified gompertz
    if ret is not None:
        try:
            label = f"Mod Gompertz (R²:{round(R2, n_decimals_for_r2(R2))})"
            ax.plot(xff, 1 - func(xff, *params), color=line_colors["mod_gompertz"], label=label, **kwargs)
        except Exception:
            pass

    # Logistic
    try:
        func2 = _curve_fit.logistic
        params2, y_predicted2, R2_2, residuals_2 = _curve_fit.fit(
            func2, x, y, init_guess=[1, 0.5, 200]
        )
        label = f"Logistic (R²:{round(R2_2, n_decimals_for_r2(R2_2))})"
        ax.plot(xff, 1 - func2(xff, *params2), color=line_colors["logistic"], label=label, **kwargs)
    except Exception:
        pass

    # Power law
    try:
        start = max(0, len(xx) // 6)
        stop = max(start + 5, len(xx) - 1)
        x_fit_pw, y_fit_pw = xx[start:stop], yy.iloc[start:stop]

        params3, y_predicted3, R2_3, residuals_3 = _curve_fit.fit(
            powerX2, x_fit_pw, y_fit_pw, init_guess=[20, 0.26]
        )

        label = f"Power law [0.25] (R²:{round(R2_3, n_decimals_for_r2(R2_3))})"

        y_pw = powerX2(xff, *params3)
        y_plot = 1 - y_pw
        mask = np.isfinite(y_plot)

        x_curve = xff[mask]
        y_curve = y_plot[mask]

        if len(x_curve) > 0:
            x_anchor = x_curve[0]
            y_anchor = 1.0

            x_plot = np.r_[x_anchor, x_curve]
            y_plot2 = np.r_[y_anchor, y_curve]

            ax.plot(x_plot, y_plot2, color=line_colors["power"], label=label, **kwargs)

    except Exception:
        pass

    # Lande
    try:
        start = max(0, len(xx) // 6)
        stop = max(start + 5, len(xx) - 1)
        x_fit_l, y_fit_l = xx[start:stop], yy.iloc[start:stop]

        params5, y_predicted5, R2_5, residuals_5 = _curve_fit.fit(
            lande, x_fit_l, y_fit_l, init_guess=[10, rmax, qsd]
        )
        label = f"Lande (R²:{round(R2_5, n_decimals_for_r2(R2_5))})"
        ax.plot(xff, 1 - lande(xff, *params5), color=line_colors["lande"], label=label, **kwargs)
    except Exception:
        pass

    model_name = runName.split("_")[0].strip("LogGrowth")

    def title(model_name, rmax, qsd, sa, qrev):
        if model_name == "A":
            title_str = f"Model {model_name} \n($r_{{max}}$={rmax:.2f}, $\\sigma$={qsd:.2f})"
        elif model_name == "D":
            title_str = f"Model {model_name} \n($r_{{max}}$={rmax:.2f}, $\\sigma$={qsd:.2f}, $S_A$={sa}, $Z$={qrev:.2f})"
        else:
            title_str = f"Model {model_name}"
        return title_str

    # formatting
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=2.0, numticks=10))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Carrying capacity K")
    ax.set_ylabel(f"P(extinction) (N={round(N) if pd.notna(N) else 'NA'})" if i == 0 else None)
    ax.set_title(title(model_name, rmax, qsd, sa, qrev), fontsize=10)

    ax.legend(fontsize=8, loc="best")


n_files = len(f)
if n_files == 0:
    raise FileNotFoundError(f"No files found in: {dat_path}")

ncols = min(3, n_files)
nrows = math.ceil(n_files / ncols)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(4 * ncols, 4.5 * nrows),
    sharex=False,
    squeeze=False
)

axes_flat = axes.flatten()

for i, filepath in enumerate(f):
    try:
        plot_one_file(axes_flat[i], filepath, i)
    except Exception as e:
        axes_flat[i].set_title(f"{Path(filepath).name}\nERROR: {type(e).__name__}")
        axes_flat[i].text(0.5, 0.5, str(e), ha="center", va="center", wrap=True, fontsize=8)
        axes_flat[i].axis("off")

# hide unused panels
for j in range(n_files, len(axes_flat)):
    axes_flat[j].axis("off")

fig.tight_layout()
plt.show()