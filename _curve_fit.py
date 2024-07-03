# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:55:15 2024

@author: Thomas Ball
"""

import numpy as np
import scipy.optimize
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import inspect

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
            if "plot_lins" in kwargs.keys():
                if a == None:
                    ret = None
                else:
                    ret = fit(curve_func, dat_x, dat_y, init_guess = [a, b, alpha])
                if kwargs["plot_lins"]:
                    plt.plot(xf, y_trans, linewidth = 1, color = "r", alpha = 0.4)
                    plt.xlabel("transformed K")
                    plt.ylabel("transformed P")
                    ret = fit(curve_func, x, y, init_guess = [a, b, alpha])
                    if not np.isnan(ret[0]).any():
                        plt.plot(xf, y_trans, linewidth = 2, color = "b", alpha = 0.7)
            else:
                if a == None:
                    ret = None
                else:
                    ret = fit(curve_func, dat_x, dat_y, init_guess = [a, b, alpha])
        return ret
    except RuntimeError:
        return "MARK"
    

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
        ret = fit(curve_func, x, y, init_guess = [b0, b1])
    return ret

def weibull(x, lam, k, gam):
    return 1 - np.exp(- ((x - gam)/lam)**k)

def betterfit_weibull(curve_func, dat_x, dat_y,
                        **kwargs):
    dat_x, dat_y = np.array(dat_x), np.array(dat_y)
    valid_indices = (dat_y > 0) & (dat_y < 1)
    x = dat_x[valid_indices]
    y = dat_y[valid_indices]
    
    if "lam_space" in kwargs.keys():
        lam_space = kwargs["lam_space"]
    else:
        lam_space = np.linspace(0, 100, 20)
    if "k_space" in kwargs.keys():
        k_space = kwargs["k_space"]
    else:
        k_space = np.linspace(0.5, 5, 20)
    if "gam_space" in kwargs.keys():
        gam_space = kwargs["gam_space"]
    else:
        gam_space = np.linspace(0, 100, 20)
    maxr = -np.inf
    lam, k, gam = None, None, None
    for ilam, glam in enumerate(lam_space):
        for ik, gk in enumerate(k_space):
            for igam, ggam in enumerate(gam_space):
                x_trans = curve_func(x, glam, gk, ggam)
                try:
                    r, _ = pearsonr(x_trans, y)
                    r = abs(r)
                except ValueError:
                    pass
                if r > maxr:
                    maxr = r
                    lam, k , gam = glam, gk, ggam
                    xf = x_trans
    if "plot_lins" in kwargs.keys():
        if kwargs["plot_lins"]:
            if lam  == None:
                ret = None
            else:
                plt.plot(xf, y, linewidth = 1, color = "r", alpha = 0.4)
                plt.xlabel("transformed K")
                plt.ylabel("transformed P")
                ret = fit(curve_func, x, y, init_guess = [lam, k, gam])
                if not np.isnan(ret[0]).any():
                    plt.plot(xf, y, linewidth = 2, color = "b", alpha = 0.7)
        else:
            if lam == None:
                ret = None
            else:
                ret = fit(curve_func, dat_x, dat_y, init_guess = [lam, k, gam])
    else:
        if lam == None:
            ret = None
        else:
            ret = fit(curve_func, dat_x, dat_y, init_guess = [lam, k, gam])
    return ret


def stretched_exp(x, a, b, c):
    return (1 + a*np.exp(-b*x)) ** (-c)
    
def betterfit_stretched_exp(curve_func, dat_x, dat_y,
                        **kwargs):
    dat_x, dat_y = np.array(dat_x), np.array(dat_y)
    valid_indices = (dat_y > 0) & (dat_y < 1)
    x = dat_x[valid_indices]
    y = dat_y[valid_indices]
    if "a_space" in kwargs.keys():
        a_space = kwargs["a_space"]
    else:
        a_space = np.linspace(1, 100, 10)
    if "b_space" in kwargs.keys():
        b_space = kwargs["b_space"]
    else:
        b_space = np.linspace(0.005, 0.5, 10)
    if "c_space" in kwargs.keys():
        c_space = kwargs["c_space"]
    else:
        c_space = np.linspace(1, 100, 10)
    maxr = -np.inf
    a, b, c = None, None, None
    for ic, gc in enumerate(c_space):
        for ib, gb in enumerate(b_space):
            for ia, ga in enumerate(a_space):
                x_trans = ((1 + ga*np.exp(-gb*x)) ** (-gc).T).T
                try:
                    r, _ = pearsonr(x_trans, y)
                    r = abs(r)
                except ValueError:
                    pass
                if r > maxr:
                    maxr = r
                    a, b, c = ga, gb, gc
                    xf = x_trans
    if a == None:
        ret = None
    else:
        plt.plot(xf, y, linewidth = 1, color = "r", alpha = 0.7)
        plt.xlabel("transformed K")
        plt.ylabel("P")
        ret = fit(curve_func, x, y, init_guess = [a, b, c])
        if not np.isnan(ret[0]).any():
            plt.plot(xf, y, linewidth = 2, color = "b")
    return ret

def DR_eq1(x, a, b, c):
    return 1 - c * np.exp( -(x / a) ** b)

def DR_eq2(x, a, b, c):
    return 1 - np.exp(1) * np.exp(-np.exp(x / a) ** b )

def DR_eq3(x, a):
    return 1 - np.exp( -(x - 1)**2 /(a * 100))

def fitDR_curves12(curve_func, dat_x, dat_y,
                            **kwargs):
    dat_x, dat_y = np.array(dat_x), np.array(dat_y)
    valid_indices = (dat_y > 0) & (dat_y < 1)
    x = dat_x[valid_indices]
    y = dat_y[valid_indices]
    if "a_space" in kwargs.keys():
        a_space = kwargs["a_space"]
    else:
        a_space = np.linspace(1, 100, 20)
    if "b_space" in kwargs.keys():
        b_space = kwargs["b_space"]
    else:
        b_space = np.linspace(-20, 20, 20)
    if "c_space" in kwargs.keys():
        c_space = kwargs["c_space"]
    else:
        c_space = np.linspace(-20, 20, 20)
    maxr = -np.inf
    a, b, c = None, None, None
    for ib, gb in enumerate(b_space):
        for ia, ga in enumerate(a_space):
            for ia, gc in enumerate(c_space):
                x_trans = curve_func(x, ga, gb, gc)
                try:
                    r, _ = pearsonr(x_trans, y)
                    r = abs(r)
                except ValueError:
                    r = -np.inf
                if r > maxr:
                    maxr = r
                    a, b, c = ga, gb, gc
                    xf = x_trans
    if "plot_lins" in kwargs.keys():
        if kwargs["plot_lins"]:
            if a  == None:
                ret = None
            else:
                plt.plot(xf, y, linewidth = 1, color = "r", alpha = 0.4)
                plt.xlabel("transformed K")
                plt.ylabel("transformed P")
                ret = fit(curve_func, x, y, init_guess = [a, b])
                if not np.isnan(ret[0]).any():
                    plt.plot(xf, y, linewidth = 2, color = "b", alpha = 0.7)
        else:
            if a == None:
                ret = None
            else:
                ret = fit(curve_func, dat_x, dat_y, init_guess = [a, b, c])
    else:
        if a == None:
            ret = None
        else:
            ret = fit(curve_func, dat_x, dat_y, init_guess = [a, b, c])
    return ret

def fitDR_curves_3(curve_func, dat_x, dat_y,
                            **kwargs):
    dat_x, dat_y = np.array(dat_x), np.array(dat_y)
    valid_indices = (dat_y > 0) & (dat_y < 1)
    x = dat_x[valid_indices]
    y = dat_y[valid_indices]
    if "a_space" in kwargs.keys():
        a_space = kwargs["a_space"]
    else:
        a_space = np.linspace(1, 100, 20)
    maxr = -np.inf
    a = None
    for ia, ga in enumerate(a_space):
        x_trans = curve_func(x, ga)
        try:
            r, _ = pearsonr(x_trans, y)
            r = abs(r)
        except ValueError:
            r = -np.inf
        if r > maxr:
            maxr = r
            a = ga
            xf = x_trans
    if "plot_lins" in kwargs.keys():
        if kwargs["plot_lins"]:
            if a  == None:
                ret = None
            else:
                plt.plot(xf, y, linewidth = 1, color = "r", alpha = 0.4)
                plt.xlabel("transformed K")
                plt.ylabel("transformed P")
                ret = fit(curve_func, x, y, init_guess = [a])
                if not np.isnan(ret[0]).any():
                    plt.plot(xf, y, linewidth = 2, color = "b", alpha = 0.7)
        else:
            if a == None:
                ret = None
            else:
                ret = fit(curve_func, dat_x, dat_y, init_guess = [a])
    else:
        if a == None:
            ret = None
        else:
            ret = fit(curve_func, dat_x, dat_y, init_guess = [a])
    return ret
                    

def betterfit_general(curve_func, dat_x, dat_y, **kwargs):
    dat_x, dat_y = np.array(dat_x), np.array(dat_y)
    if "ylim" in kwargs.keys():
        yl0,yl1 = kwargs["ylim"]
    else:
        yl0,yl1 = (0, 1)
    valid_indices = (dat_y > yl0) & (dat_y < yl1)
    x = dat_x[valid_indices]
    y = dat_y[valid_indices]
    
    if "param_names" in kwargs.keys():
        param_names = kwargs["param_names"]
    else:
        param_names = tuple([f"p{n}" for n in range(len(inspect.getargspec(curve_func)))])
        
    if "param_spaces" in kwargs.keys():
        param_spaces = {param: kwargs["param_spaces"][p] for p, param in enumerate(param_names)}
    else:
        param_spaces = {param: np.linspace(-10, 10, 100) for param in param_names}
    maxr = -np.inf
    best_params = [None for p in param_names] 
    for param_combination in np.ndindex(*(len(param_spaces[param]) for param in param_names)):
        current_params = {param_names[i]: param_spaces[param_names[i]][param_combination[i]] for i in range(len(param_names))}
        x_trans = curve_func(x, *current_params)
        try:
            r, _ = pearsonr(x_trans, y)
            r = abs(r)
        except ValueError:
            r = -np.inf
        if r > maxr:
            maxr = r
            best_params = current_params
            xf = x_trans
            











