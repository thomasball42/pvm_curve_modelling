# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:50:01 2024

@author: Thomas Ball
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Inputs
# =============================================================================
Rmax = 0.1
S = 0.1
num_runs = 1000
num_years = 100
Ks = np.geomspace(1, 120, num = 100)

years = np.arange(0, num_years, 1)


# Normal generator for Q
Qnormal = lambda : np.random.normal(loc=0.0, scale=S, size=None)

# =============================================================================
# Models
# =============================================================================
def Ri_modelA(Rmax, species, **kwargs):
    Rm = Rmax * (1-species.Nm / species.Km)
    Rf = Rmax * (1-species.Nf / species.Kf)
    return Rf, Rm

def Ri_modelB(Rmax, species, **kwargs):
    Rm = Rmax * (1 - ((species.Nm+species.Nf)/(species.Km+species.Kf)))
    Rf = Rmax * (1 - ((species.Nm+species.Nf)/(species.Km+species.Kf)))
    return Rf, Rm

def Ri_modelC(Rmax, species, **kwargs):
    
    Sa, B = species.Sa,species.B
    
    if len(species.Nm_hist) >= B:
        fecundity_factor = np.array([species.Nm_hist[-B+1]/species.Nf_hist[-B+1],
                                    species.Nf_hist[-B+1]/species.Nm_hist[-B+1]]).min()
    else:
        fecundity_factor = 1
        
    if "Rgen_model" in kwargs.keys():
        Rgen_model = kwargs["Rgen_model"]
    else:
        Rgen_model = Ri_modelA
        
    Rf, Rm = Rgen_model(Rmax, species)
    
    def Rprime(R, fecundity):
        return np.log(Sa + ((np.exp(R) - Sa)*fecundity))
    
    Rf_prime = Rprime(Rf, fecundity_factor)
    Rm_prime = Rprime(Rm, fecundity_factor)
    
    return Rf_prime, Rm_prime

# =============================================================================
# Population
# =============================================================================
class Population:
    def __init__(self, K, Sa, B):
        self.K, self.Sa, self.B = K,Sa,B
        self.Km, self.Kf = K/2,K/2
        self.Nm = self.Km
        self.Nf = self.Kf
        self.Nm_hist = []
        self.Nf_hist = []
        self.Rm_hist = []
        self.Rf_hist = []
        
    def updateN(self, Ni, Ri, K, Q):
        newN = Ni * np.exp(Ri + Q)
        if newN > K:
            newN = K
        return math.floor(newN) 
    
    def iterate(self, model, Q, **kwargs):
        Nm, Nf, Km, Kf = self.Nm,self.Nf,self.Km,self.Kf
        Rf, Rm = model(Rmax, sp, **kwargs)
        self.Nf_hist.append(Nf)
        self.Nm_hist.append(Nm)
        self.Rf_hist.append(Rf)
        self.Rm_hist.append(Rm)
        self.Nf = self.updateN(Nf, Rf, Kf, Q)
        self.Nm = self.updateN(Nm, Rm, Km, Q)

# =============================================================================
# Main
# =============================================================================

# dataframe to put the result into
odf = pd.DataFrame()

# set up some runs
runs = {
    "ModelA": {
        "model": Ri_modelA,
        "Sa": None,
        "B": None,
        "Rmax": Rmax,
        "S": S,
        "Q": Qnormal,
        "num_runs": num_runs,
        "kwargs": {}
    },
    "ModelB": {
        "model": Ri_modelB,
        "Sa": None,
        "B": None,
        "Rmax": Rmax,
        "S": S,
        "Q": Qnormal,
        "num_runs": num_runs,
        "kwargs": {}
    },
    "ModelCb": {
        "model": Ri_modelC,
        "Sa": 0.5,
        "B": 10,
        "Rmax": Rmax,
        "S": S,
        "Q": Qnormal,
        "num_runs": num_runs,
        "kwargs": {"Rgen_model" : Ri_modelB}
    }
}

# Loop through run parameters
for run_name, run_params in runs.items(): 
    
    model = run_params["model"]
    Sa = run_params["Sa"]
    B = run_params["B"]
    Rmax = run_params["Rmax"]
    S = run_params["S"]
    q_gen = run_params["Q"]
    num_runs = run_params["num_runs"]
    kwargs = run_params["kwargs"]
    
    for k, K in enumerate(Ks):
        extinctions = 0
        
        for r in range(num_runs):
            print(f"{run_name}: Progress - {k}/{len(Ks)}, {r}/{num_runs}   ")
            sp = Population(K, Sa, B)

            for y in years:
                sp.iterate(model, q_gen(), **kwargs)
                
                # record extinction and stop run
                if sp.Nf < 1 or sp.Nm < 1:
                    extinctions += 1
                    break
                
        P = 1 - extinctions / num_runs
        
        # Store results
        odf.loc[len(odf), ["runName", "K", "Sa", "B", "Rmax", "S", "N", "P"]] = [
            run_name, K, Sa, B, Rmax, S, num_runs, P]
    
#%%

fig, ax = plt.subplots()  
for runName in odf.runName.unique():
    dat = odf[odf.runName == runName]
    
    Ks = dat.K
    P = dat.P
    ax.scatter(np.log2(Ks), P, label = runName)
    ax.set_xticks(ax.get_xticks(), labels = 2 ** ax.get_xticks())
    ax.legend()