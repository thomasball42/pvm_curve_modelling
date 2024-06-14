# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:55:15 2024

@author: Thomas Ball
"""

import numpy as np
import scipy.optimize
from scipy.stats import pearsonr

def lin(x, m, c):
    return m * x + c

def gompertz(x, a, b, c):
    return np.exp(-np.exp(a + b*(x**c)))

def logistic(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

def logistic_alt(x, b0, b1):
    return np.exp(b0 + b1 * x)/(1 + np.exp(b0 + b1 * x))

def fit(curve_func, dat_x, dat_y, **kwargs):
    init_guess = kwargs.pop("init_guess", 
                            np.zeros(curve_func.__code__.co_argcount - 1))
    params, covariance = scipy.optimize.curve_fit(curve_func, dat_x, dat_y, 
                                                  init_guess, **kwargs)
    y_predicted = curve_func(dat_x, *params)
    residuals = dat_y - y_predicted
    RSS = np.sum((dat_y - np.mean(dat_y))**2)
    R2 = 1 - (np.sum(residuals**2) / RSS)
    return params, y_predicted, R2

def betterfit_gompertz(curve_func, dat_x, dat_y, 
                       **kwargs):
    dat_x, dat_y = np.array(dat_x), np.array(dat_y)
    valid_indices = (dat_y > 0) & (dat_y < 1)
    x = dat_x[valid_indices]
    y = dat_y[valid_indices]
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
            alpha = alpha_space[c]    
    if a == None:
        ret = None
    else:
        ret = fit(curve_func, dat_x, dat_y, init_guess = [a, b, alpha])
    return ret

def betterfit_logistic(curve_func, dat_x, dat_y,
                        **kwargs):
    dat_x, dat_y = np.array(dat_x), np.array(dat_y)
    valid_indices = (dat_y > 0) & (dat_y < 1)
    x = dat_x[valid_indices]
    y = dat_y[valid_indices]
    def trans(z):
        return np.log(z/(1-z))
    y_trans = trans(y)
    if len(y) < 1:
        ret = None
    else:
        b1, b0 = np.polyfit(x, y_trans, 1)
        # params = (b0, b1)
        # y_predicted = curve_func(x, *params)
        # residuals = y - y_predicted
        # RSS = np.sum((y - np.mean(y))**2)
        # R2 = 1 - (np.sum(residuals**2) / RSS)
        # ret = params, y_predicted, R2
        ret = fit(curve_func, x, y, init_guess = [b0, b1])
    return ret

def weibull(x, lamb, k):
    return 1 - np.exp(-(x/lamb)**k)

# def betterfit_weibull(curve_func, dat_x, dat_y,
#                         **kwargs):
#     dat_x, dat_y = np.array(dat_x), np.array(dat_y)
#     valid_indices = (dat_y > 0) & (dat_y < 1)
#     x = dat_x[valid_indices]
#     y = dat_y[valid_indices]
#     y_trans = np.log(1/(1-y))
    
#     if "k_space" in kwargs.keys():
#         k_space = kwargs["k_space"]
#     else:
#         k_space = np.arange(0.1, 1.1, 0.1)
#     x_k_space = (x[:, np.newaxis] ** (k_space).T).T
#     maxr = -np.inf
#     lamb, k = None, None
#     for c, col in enumerate(x_k_space):
#         r, _ = pearsonr(col, y_trans)
#         r = abs(r)
#         if r > maxr:
#             maxr = r
#             b, a = np.polyfit(col, y_trans, 1)
#             k = k_space[c]    
#     if a == None:
#         ret = None
#     else:
#         ret = fit(curve_func, dat_x, dat_y, init_guess = [lamb, k])
#     return ret