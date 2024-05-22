# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:55:15 2024

@author: Thomas Ball
"""

import numpy as np
import scipy.optimize
from scipy.stats import pearsonr
import sympy as sp

def lin(x, m, c):
    return m * x + c

def gompertz(x, a, b, c):
    return np.exp(-np.exp(a + b*(x**c)))

def fit(curve_func, dat_x, dat_y, **kwargs):
    if "init_guess" in kwargs.keys():
        init_guess = kwargs["init_guess"]
    else:
        init_guess = np.zeros(curve_func.__code__.co_argcount - 1)
    params, covariance = scipy.optimize.curve_fit(curve_func, dat_x, dat_y, init_guess)
    y_predicted = curve_func(dat_x, *params)
    residuals = dat_y - y_predicted
    RSS = np.sum((dat_y - np.mean(dat_y))**2)
    R2 = 1 - (np.sum(residuals**2) / RSS)
    return params, y_predicted, R2, RSS

def betterfit_gompertz(curve_func, dat_x, dat_y, **kwargs):
    dat_x, dat_y = np.array(dat_x), np.array(dat_y)
    x = dat_x[(dat_y>0)&(dat_y<1)]
    y = dat_y[(dat_y>0)&(dat_y<1)]
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
            a, b = np.polyfit(col, y_trans, 1)
            alpha = alpha_space[c]    
    if a == None:
        ret = None
    else:
        ret = fit(curve_func, dat_x, dat_y, init_guess = [b, a, alpha])
    return ret

def gomp_inflection(params, x_vals):
    """ This doesn't appear to work properly """
    _ = sp.symbols("_")
    a, b, alpha = sp.symbols("a b alpha")
    func = sp.exp(-sp.exp(a + b*(_**alpha)))
    dfunc = sp.diff(func, _)
    ddfunc = sp.diff(dfunc, _)
    pdict = {j : params[i] for i,j in enumerate(["a", "b","alpha"])}
    root = sp.nsolve(ddfunc.subs(pdict), _, x_vals[len(x_vals) / 2])
    return np.float32(root)
    
        
    