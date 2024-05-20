# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:50:01 2024
`
@author: Thomas Ball
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.colors as mcolors
import pandas as pd

import _population
import _curve_fit

#%%
# =============================================================================
# Inputs
# =============================================================================
Rmax = 0.1
S = 0.1
num_runs = 200
num_years = 100
Ks = np.geomspace(1, 20000, num = 100)
Ks = np.unique(np.round(Ks))
years = np.arange(0, num_years, 1)

# Normal generator for Q
def normal_dist(loc, S, size):
    return np.random.normal(loc=loc, scale=S, size=None)    

def poisson_dist(lam):
    return np.random.poisson(lam)
    
# =============================================================================
# Models
# =============================================================================

def Ni_log_floor(Ni, Ri, K, Q):
    newN = Ni * np.exp(Ri + Q)
    return math.floor(newN) # ROUND DOWN - check this!

def Ni_log_capped(Ni, Ri, K, Q):
    newN = Ni * np.exp(Ri + Q)
    if newN > K: newN = K
    return math.floor(newN)
    
def Ni_log_poisson(Ni, Ri, K, Q):
    newN = Ni * np.exp(Ri + Q)
    return poisson_dist(newN)

def Ni_log_realnums(Ni, Ri, K, Q):
        newN = Ni * np.exp(Ri + Q)
        return newN
        
def Ni_log_round(Ni, Ri, K, Q):
    newN = Ni * np.exp(Ri + Q)
    return round(newN)

def Ri_model_StochasticGrowth(Rmax, species, **kwargs):
    return 0, 0

def Ri_model_LogisticGrowth(Rmax, species, **kwargs):
    """ Equivalent to 'A' in Rhys' doc """
    Rm = Rmax * (1-species.Nm / species.Km)
    Rf = Rmax * (1-species.Nf / species.Kf)
    return Rf, Rm

def Ri_modelB(Rmax, species, **kwargs):
    Rm = Rmax * (1 - ((species.Nm+species.Nf)/(species.Km+species.Kf)))
    Rf = Rmax * (1 - ((species.Nm+species.Nf)/(species.Km+species.Kf)))
    return Rf, Rm

def Ri_modelC(Rmax, species, **kwargs):
    Sa, B = species.Sa,species.B
    if len(species.Nm_hist) > B:
        fecundity_factor = np.array([species.Nm_hist[-B+1]/species.Nf_hist[-B+1],
                                    species.Nf_hist[-B+1]/species.Nm_hist[-B+1]]).min()
    else:
        fecundity_factor = 1
    if "Rgen_model" in kwargs.keys():
        Rgen_model = kwargs["Rgen_model"]
    else:
        Rgen_model = Ri_model_LogisticGrowth
    Rf, Rm = Rgen_model(Rmax, species)
    def Rprime(R, fecundity):
        return np.log(Sa + ((np.exp(R) - Sa)*fecundity))
    Rf_prime = Rprime(Rf, fecundity_factor)
    Rm_prime = Rprime(Rm, fecundity_factor)
    return Rf_prime, Rm_prime

#%% set up some runs
runs = {
    # "RUNNAME": {
    #     "modelR": RMODEL,
    #     "modelN" : NIMODEL,
    #     "Sa": None,
    #     "B": None,
    #     "Rmax": Rmax,
    #     "S": S,
    #     "Q": normal_dist,
    #     "Qpars": tuple of Q paramters for Q (mean, var, size)
    #     "num_runs": num_runs,
    #     "kwargs": dict of kwargs for rmodel,
    # },
    "LogisticGrowth A poisson": {
        "modelR": Ri_model_LogisticGrowth,
        "modelN" : Ni_log_poisson,
        "Sa": 0.1,
        "B": 2,
        "Rmax": Rmax,
        "Q": normal_dist,
        "Qpars" : (0.0, 0.2, None),
        "num_runs": num_runs,
        "kwargs": {}
    },
    "LogisticGrowth A floor": {
        "modelR": Ri_model_LogisticGrowth,
        "modelN" : Ni_log_floor,
        "Sa": 0.1,
        "B": 2,
        "Rmax": Rmax,
        "Q": normal_dist,
        "Qpars" : (0.0, 0.2, None),
        "num_runs": num_runs,
        "kwargs": {}
    },
    "LogisticGrowth A realnums": {
        "modelR": Ri_model_LogisticGrowth,
        "modelN" : Ni_log_realnums,
        "Sa": 0.1,
        "B": 2,
        "Rmax": Rmax,
        "Q": normal_dist,
        "Qpars" : (0.0, 0.2, None),
        "num_runs": num_runs,
        "kwargs": {},
    },
    
    # "LogisticGrowth B poisson": {
    #     "modelR": Ri_modelB,
    #     "modelN" : Ni_log_poisson,
    #     "Sa": 0.1,
    #     "B": 2,
    #     "Rmax": Rmax,
    #     "Q": normal_dist,
    #     "Qpars" : (0.0, 0.2, None),
    #     "num_runs": num_runs,
    #     "kwargs": {}
    # },
    # "LogisticGrowth B floor": {
    #     "modelR": Ri_modelB,
    #     "modelN" : Ni_log_floor,
    #     "Sa": 0.1,
    #     "B": 2,
    #     "Rmax": Rmax,
    #     "Q": normal_dist,
    #     "Qpars" : (0.0, 0.2, None),
    #     "num_runs": num_runs,
    #     "kwargs": {}
    # },
    # "LogisticGrowth B realnums": {
    #     "modelR": Ri_modelB,
    #     "modelN" : Ni_log_realnums,
    #     "Sa": 0.1,
    #     "B": 2,
    #     "Rmax": Rmax,
    #     "Q": normal_dist,
    #     "Qpars" : (0.0, 0.2, None),
    #     "num_runs": num_runs,
    #     "kwargs": {},
    # },
    # "LogisticGrowth C poisson": {
    #     "modelR": Ri_modelC,
    #     "modelN" : Ni_log_poisson,
    #     "Sa": 0.1,
    #     "B": 2,
    #     "Rmax": Rmax,
    #     "Q": normal_dist,
    #     "Qpars" : (0.0, 0.2, None),
    #     "num_runs": num_runs,
    #     "kwargs": {}
    # },
    # "LogisticGrowth C floor": {
    #     "modelR": Ri_modelC,
    #     "modelN" : Ni_log_floor,
    #     "Sa": 0.1,
    #     "B": 2,
    #     "Rmax": Rmax,
    #     "Q": normal_dist,
    #     "Qpars" : (0.0, 0.2, None),
    #     "num_runs": num_runs,
    #     "kwargs": {}
    # },
    # "LogisticGrowth C realnums": {
    #     "modelR": Ri_modelC,
    #     "modelN" : Ni_log_realnums,
    #     "Sa": 0.1,
    #     "B": 2,
    #     "Rmax": Rmax,
    #     "Q": normal_dist,
    #     "Qpars" : (0.0, 0.2, None),
    #     "num_runs": num_runs,
    #     "kwargs": {},
    # },
}
    

# %%
# =============================================================================
# Main
# =============================================================================

# dataframe to put the result into
odf = pd.DataFrame()

# Loop through run parameters
for run_name, run_params in runs.items(): 
    
    modelR = run_params["modelR"]
    modelN = run_params["modelN"]
    Sa = run_params["Sa"]
    B = run_params["B"]
    Rmax = run_params["Rmax"]
    q_gen = run_params["Q"]
    q_pars = run_params["Qpars"]
    num_runs = run_params["num_runs"]
    kwargs = run_params["kwargs"]
    
    for k, K in enumerate(Ks):
        extinctions = 0
        print(f"{run_name}: {k}/{len(Ks)}")
        for r in range(num_runs):
            
            sp = _population.Population(K, Sa, B, Rmax)

            for y in years:
                sp.iterate(modelR, modelN, q_gen(*q_pars), **kwargs)
                
                # record extinction and stop run
                if sp.Nf < 1 or sp.Nm < 1:
                    extinctions += 1
                    break
                
        P = 1 - extinctions / num_runs
        
        # Store results
        odf.loc[len(odf), ["runName", "K", "Sa", "B", "Rmax", "N", "P"]] = [
            run_name, K, Sa, B, Rmax, num_runs, P]
    
#%% PLOT
fig, ax = plt.subplots()

base_cmap = plt.get_cmap('viridis')
colors = base_cmap(np.linspace(0, 1, len(runs)))

func = _curve_fit.gompertz

for i, runName in enumerate(odf.runName.unique()):
    
    if "asdasd" in runName:
        continue
    dat = odf[odf.runName == runName]

    x = dat.K
    y = dat.P
    
    color = colors[i]
    ax.scatter(x, y, alpha = 0.7, s = 15, color = color)
    scatter_color = ax.collections[-1].get_facecolor()
    
    try:
        params, y_predicted, R2, RSS = _curve_fit.betterfit_gompertz(func, x, y)
        R2 = round(R2, 5)
        label = f"{runName} :: Gompertz R2:{R2}, \nParams:{params}]"
        ax.plot(np.geomspace(x.min(), x.max(), 200), 
                func(np.geomspace(x.min(), x.max(), 200), *params), 
                color = scatter_color, alpha = 0.7,
                label = label)
    except RuntimeError: 
        print(f"Warning: No fit for {runName}")
        label = f"{runName} [fit gompertz NO FIT]"
        ax.plot(x, [0 for k in x], color = color, alpha = 0.7,
                label = label)
    
ax.set_xscale("log", base = 2)
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.set_xlabel("K")
ax.set_ylabel(f"P (N={num_runs})")
ax.legend()