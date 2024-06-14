# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:22:31 2024

@author: tom
"""

import numpy as np

def normal_dist(loc, S):
    return np.random.normal(loc=loc, scale=S, size=None)    

def poisson_dist(lam): 
    if lam < 0:
        ret = -1 # causes extinction
    else:
        try:
            ret = np.random.poisson(lam)
        except ValueError as e:
            print(f"ValueError in np.random.poisson: {e} - likely caused because 'Q' is too big. Run discounted.")
            ret = None   
    return ret

# =============================================================================
# Population models
# =============================================================================
def Ni_exp_poisson(Ni, Ri, K, Q):
    newN = Ni * np.exp(Ri + Q)
    newNp = poisson_dist(newN)
    return newNp 

def Ni_mult_poisson(Ni, Ri, K, Q):
    newN = Ni + Ni * (Ri + Q)
    newNp = poisson_dist(newN)
    return newNp 

def Ni_log(Ni, Ri, K, Q):
    newN = Ni + (Ni * Ri) + (Ni * Q)
    newNp = poisson_dist(newN)
    return newNp

# =============================================================================
# growth rates
# =============================================================================
def Ri_model_GompertzGrowthA(Rmax, species, **kwargs):
    Rm = Rmax * (1-species.Nm / species.Km)
    Rf = Rmax * (1-species.Nf / species.Kf)
    return Rf, Rm

def Ri_model_GompertzGrowthB(Rmax, species, **kwargs):
    Rm = Rmax * (1 - ((species.Nm+species.Nf)/(species.Km+species.Kf)))
    Rf = Rmax * (1 - ((species.Nm+species.Nf)/(species.Km+species.Kf)))
    return Rf, Rm

def Ri_model_GompertzGrowthC(Rmax, species, **kwargs):
    Sa, B = species.Sa,species.B
    if len(species.Nm_hist) > B:
        fecundity_factor = np.array([species.Nm_hist[-B+1]/species.Nf_hist[-B+1],
                                    species.Nf_hist[-B+1]/species.Nm_hist[-B+1]]).min()
    else:
        fecundity_factor = 1
    if "Rgen_model" in kwargs.keys():
        Rgen_model = kwargs["Rgen_model"]
    else:
        Rgen_model = Ri_model_GompertzGrowthA
    Rf, Rm = Rgen_model(Rmax, species)
    def Rprime(R, fecundity):
        return np.log(Sa + ((np.exp(R) - Sa)*fecundity))
    Rf_prime = Rprime(Rf, fecundity_factor)
    Rm_prime = Rprime(Rm, fecundity_factor)
    return Rf_prime, Rm_prime

# =============================================================================
# Extras
# =============================================================================
def getB(Rmax, Sa):
    B = (1/Rmax) - (Sa / (np.exp(Rmax) - Sa))
    if np.isinf(B):
        B = None
    else: 
        B = round(B)
    return B