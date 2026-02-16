# -*- coding: utf-8 -*-
"""
@author: Thomas Ball
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker

import _curve_fit


def find_simulation_files(results_path, model):

    f = []
    for path, subdirs, files in os.walk(results_path):
        for name in files:
            f.append(os.path.join(path, name))
    return [file for file in f if ".csv" in file and f"LogGrowth{model}" in file]


def scale_to_unit_interval(x, y):

    reversed_arr = y[::-1]
    first_non_one_index = np.argmax(reversed_arr != 1)
    tail_start_index = len(y) - first_non_one_index
    y = y[:tail_start_index]
    
    if len(x) == tail_start_index:
        x = x / x.max()
    else:
        x = x[:tail_start_index] / x[tail_start_index]
    
    return x, y


def fit_gompertz_curve(x, y):
    """
    Fit a modified Gompertz curve to the data.
    
    Parameters
    ----------
    x : array-like
        Independent variable
    y : array-like
        Dependent variable
    
    Returns
    -------
    dict
        Dictionary containing:
        - params: tuple of fitted parameters (a, b, alpha)
        - y_predicted: predicted y values
        - R2: R-squared value
        - resids: residuals
        - model_name: name of the fitted model
        - rsd: residual standard deviation
        - rmse: root mean squared error
    """
    func = _curve_fit.mod_gompertz
    param_names = ("param_a", "param_b", "param_alpha")
    
    # Try first parameter space
    ret = _curve_fit.betterfit_gompertz(
        func, x, y, 
        alpha_space=np.arange(0, 5, 0.001), 
        ylim=(0.0001, 0.9999), 
        plot_lins=False
    )
    
    # Try alternative parameter space if first fails
    if ret is None:
        ret = _curve_fit.betterfit_gompertz(
            func, x, y, 
            alpha_space=np.arange(-5, 0, 0.001),
            ylim=(0.05, 0.95), 
            plot_lins=False
        )
    
    if ret is None:
        return {
            'params': tuple([np.nan for _ in param_names]),
            'y_predicted': None,
            'R2': np.nan,
            'resids': np.nan,
            'model_name': np.nan,
            'rsd': np.nan,
            'rmse': np.nan
        }
    
    params, y_predicted, R2, resids = ret
    
    # Calculate error metrics
    rsd = np.sqrt(np.sum(resids ** 2) / len(resids))
    rmse = np.sqrt(np.sum(resids ** 2) / (len(resids) - len(params) + 1))
    
    return {
        'params': params,
        'y_predicted': y_predicted,
        'R2': R2,
        'resids': resids,
        'model_name': func.__name__,
        'rsd': rsd,
        'rmse': rmse
    }


def calculate_kx_analytical(X, a, b, alpha):
    if np.isnan(a):
        return np.nan
    return ((np.log(-np.log(X)) - a) / b) ** (1 / alpha)


def calculate_extinction_metrics(xff, yff, params):
    """
    Calculate extinction probability metrics from fitted curve.
    
    Parameters
    ----------
    xff : array-like
        Fine-grained x values
    yff : array-like
        Fitted y values
    params : tuple
        Fitted Gompertz parameters (a, b, alpha)
    
    Returns
    -------
    dict
        Dictionary containing:
        - kX values (k10, k20, ..., k90): K at different probability thresholds
        - dPdK_tp: turning point of dP/dK (maximum gradient)
    """
    # Calculate kX values for different thresholds
    kXs = np.arange(0.1, 1.0, 0.1)
    kX_vals = [calculate_kx_analytical(X, *params) for X in kXs]
    kX_names = [f"k{int(X*100)}" for X in kXs]
    
    # Calculate maximum gradient point
    if not np.isnan(yff).all():
        dPdK = np.diff(yff) / np.diff(xff)
        dPdK_max = xff[np.argmax(dPdK) + 1]
    else:
        dPdK_max = np.nan
    
    result = {name: val for name, val in zip(kX_names, kX_vals)}
    result['dPdK_tp'] = dPdK_max
    
    return result


def plot_curve_fit(ax, x, y, func, params, R2, runName, scale_1_0=False):
    # Determine color and marker based on fit quality
    if np.isnan(R2):
        c = "r"
        marker = "x"
    elif params[-1] < 0:
        c = "m"
        marker = "o"
    else:
        c = plt.cm.get_cmap('viridis')((R2 - 0.990) / (1 - 0.990))
        marker = "o"
    
    # Determine number of decimal places for R2
    nnnn = 0
    while round(R2, nnnn) == 1:
        nnnn += 1
    
    label = f"{runName}_(R2:{round(R2, nnnn + 1)})"
    
    # Plot scatter
    ax.scatter(x, 1 - y, color=c, alpha=0.7, marker=marker)
    
    # Plot fitted curve
    xff = np.geomspace(x.min(), x.max(), num=100000)
    scatter_color = ax.collections[-1].get_facecolor()
    ax.plot(xff, 1 - func(xff, *params), color=scatter_color, label=label)
    
    # Format axes
    if not scale_1_0:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=10))
        
        def custom_formatter(x, pos):
            return f'$10^{{{int(np.log10(x))}}}$'
        
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(custom_formatter))


def plot_parameter_space(ax, qsd, rmax, R2, params):
    if np.isnan(R2):
        c = "r"
        marker = "x"
    elif params[-1] < 0:
        c = "m"
        marker = "o"
    else:
        c = plt.cm.get_cmap('viridis')((R2 - 0.990) / (1 - 0.990))
        marker = "o"
    
    ax.scatter(qsd, rmax, color=c, s=70, marker=marker)


def load_existing_fits(data_fits_path, overwrite):
    if os.path.isfile(data_fits_path) and not overwrite:
        data_fits = pd.read_csv(data_fits_path, index_col=0)
        existing_runs = set(data_fits['runName'].unique())
    else:
        data_fits = pd.DataFrame()
        existing_runs = set()
    
    return data_fits, existing_runs


def extract_run_parameters(dat):

    runName = dat.runName.unique()
    if len(runName) != 1:
        raise ValueError("Expected exactly one unique runName in the data.") 
    runName = runName[0]

    model = runName.split("_")[0]
    qsd = dat.QSD.unique().item()
    N = dat.N.unique().item()
    rmax = dat.RMAX.unique().item()
    B = dat.B.unique().item()
    x = dat.K 
    y = dat.P
    max_y = y.max()
    
    # Optional parameters
    try:
        qrev = dat.QREV.unique().item()
        if "LogGrowthD" not in model:
            qrev = np.nan
    except AttributeError:
        qrev = np.nan
    
    try:
        sa = dat.SA.unique().item()
    except AttributeError:
        sa = None
    
    return {
        'runName': runName,
        'model': model,
        'qsd': qsd,
        'N': N,
        'rmax': rmax,
        'B': B,
        'x': x,
        'y': y,
        'max_y': max_y,
        'qrev': qrev,
        'sa': sa
    }