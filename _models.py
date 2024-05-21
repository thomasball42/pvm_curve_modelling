# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:22:31 2024

@author: tom
"""

import math
import numpy as np

def normal_dist(loc, S):
    return np.random.normal(loc=loc, scale=S, size=None)    

def poisson_dist(lam):
    return np.random.poisson(lam)

def Ni_log_floor(Ni, Ri, K, Q):
    newN = Ni * np.exp(Ri + Q)
    return math.floor(newN) # ROUND DOWN - check this!

def Ni_log_capped(Ni, Ri, K, Q):
    newN = Ni * np.exp(Ri + Q)
    if newN > K: newN = K
    return math.floor(newN)
    
def Ni_log_poisson(Ni, Ri, K, Q):
    newN = Ni * np.exp(Ri + Q)
    try:
        newNp = poisson_dist(newN)
    except ValueError as e:
        print(f"ValueError in np.random.poisson: {e} - likely caused because 'Sa' is too big. Run discounted.")
        newNp = None
    return newNp 

def Ni_log_realnums(Ni, Ri, K, Q):
        newN = Ni * np.exp(Ri + Q)
        return newN
        
def Ni_log_round(Ni, Ri, K, Q):
    newN = Ni * np.exp(Ri + Q)
    return round(newN)

def Ri_model_EnvStochasticOnlyGrowth(Rmax, species, **kwargs):
    return 0, 0

def Ri_model_LogisticGrowthA(Rmax, species, **kwargs):
    """ Equivalent to 'A' in Rhys' doc """
    Rm = Rmax * (1-species.Nm / species.Km)
    Rf = Rmax * (1-species.Nf / species.Kf)
    return Rf, Rm

def Ri_model_LogisticGrowthB(Rmax, species, **kwargs):
    Rm = Rmax * (1 - ((species.Nm+species.Nf)/(species.Km+species.Kf)))
    Rf = Rmax * (1 - ((species.Nm+species.Nf)/(species.Km+species.Kf)))
    return Rf, Rm

def Ri_model_LogisticGrowthC(Rmax, species, **kwargs):
    Sa, B = species.Sa,species.B
    if len(species.Nm_hist) > B:
        fecundity_factor = np.array([species.Nm_hist[-B+1]/species.Nf_hist[-B+1],
                                    species.Nf_hist[-B+1]/species.Nm_hist[-B+1]]).min()
    else:
        fecundity_factor = 1
    if "Rgen_model" in kwargs.keys():
        Rgen_model = kwargs["Rgen_model"]
    else:
        Rgen_model = Ri_model_LogisticGrowthA
    Rf, Rm = Rgen_model(Rmax, species)
    def Rprime(R, fecundity):
        return np.log(Sa + ((np.exp(R) - Sa)*fecundity))
    Rf_prime = Rprime(Rf, fecundity_factor)
    Rm_prime = Rprime(Rm, fecundity_factor)
    return Rf_prime, Rm_prime

def getB(Rmax, Sa):
    return (1/Rmax) - (Sa / (np.exp(Rmax) - Sa))