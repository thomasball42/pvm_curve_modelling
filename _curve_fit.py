# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:55:15 2024

@author: Thomas Ball
"""

import numpy as np
import scipy.optimize
from scipy.stats import pearsonr
from scipy.stats import gamma
import matplotlib.pyplot as plt

import inspect

def lin(x, m, c):
    return m * x + c

def logistic(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))
    
    
def fit(curve_func, dat_x, dat_y, **kwargs):
    init_guess = kwargs.pop("init_guess", 
                            np.zeros(curve_func.__code__.co_argcount - 1))
    params, covariance = scipy.optimize.curve_fit(curve_func, dat_x, dat_y, 
                                                  init_guess, **kwargs)
    y_predicted = curve_func(dat_x, *params)
    residuals = dat_y - y_predicted
    RSS = np.sum((dat_y - np.mean(dat_y))**2)
    R2 = 1 - (np.sum(residuals**2) / RSS)
    return params, y_predicted, R2, residuals

def basic_gomp(x, a, b):
    return np.exp(-np.exp(a + b*(x)))
    
def mod_gompertz(x, a, b, c):
    return np.exp(-np.exp(a + b*(x**c)))

# def mod_gompertz_invert(y, a, b, c):
#     y = np.array(y)
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
#     return ( (np.log(-np.log(y))-a)/b) ** (-c)
    
def betterfit_gompertz(curve_func, dat_x, dat_y, **kwargs):
    try:
        dat_x, dat_y = np.array(dat_x), np.array(dat_y)
        if "ylim" in kwargs.keys():
            yl0,yl1 = kwargs["ylim"]
        else:
            yl0,yl1 = (0, 1)
        valid_indices = (dat_y > yl0) & (dat_y < yl1)
        x = dat_x[valid_indices]
        y = dat_y[valid_indices]
        if len(y) < 3:
            ret = None
        else:
            y_trans = np.log(-np.log(y))
            if "alpha_space" in kwargs.keys():
                alpha_space = kwargs["alpha_space"]
            else:
                alpha_space = np.arange(0.1, 1.1, 0.1)
            k_alpha_space = (x[:, np.newaxis] ** alpha_space.T).T
            maxr = -np.inf
            a, b, alpha = None, None, None
            for c, col in enumerate(k_alpha_space):
                r, _ = pearsonr(col, y_trans)
                r = abs(r)
                if r > maxr:
                    maxr = r
                    b, a = np.polyfit(col, y_trans, 1)
                    xf = col
                    alpha = alpha_space[c]
            if a == None:
                ret = None
            else:
                ret = fit(curve_func, dat_x, dat_y, init_guess = [a, b, alpha])
            if "plot_lins" in kwargs.keys() and kwargs["plot_lins"]:
                if alpha < 0:
                    c = "g"
                if alpha >= 0:
                    c = "r"
                plt.plot(xf, y_trans, linewidth = 2, color = c, alpha = 0.7)
                plt.xlabel("transformed K")
                plt.ylabel("transformed P")   
        return ret
    except RuntimeError:
        pass




