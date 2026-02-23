# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:22:31 2024

@author: tom
"""

import copy
import numpy as np
import _population

def poisson_dist(lam): 
    if lam < 0 or np.isnan(lam):
        ret = -1 # causes extinction
    else:
        try:
            ret = np.random.poisson(lam)
        except ValueError:
            ret = round(lam) 
    return ret

def Q_normal_dist(loc, S, RF, Q_hist):
    return np.random.normal(loc=loc, scale=S, size=None)    

def Q_ornstein_uhlenbeck(loc, S, RF, Q_hist):
    """effectively a random walk with a central tendency, loc and S are the 
    position and standard deviation of the normal respectively, RF is the
    reversion factor"""
    if len(Q_hist) > 0:
        curr = Q_hist[-1]
    else:
        curr = loc
    reversion = RF * (loc - curr)
    return curr + reversion + np.random.normal(loc, S)
    
# =============================================================================
# Population models
# =============================================================================
def Ni_exp_poisson(Ni, Ri, K, Q):
    """unused"""
    newN = Ni * np.exp(Ri + Q)
    newNp = poisson_dist(newN)
    return newNp 

def Ni_mult_poisson(Ni, Ri, K, Q):
    """unused"""
    newN = Ni + Ni * (Ri + Q)
    newNp = poisson_dist(newN)
    return newNp 

def Ni_log(Ni, Ri, K, Q):
    newN = Ni + (Ni * Ri) + (Ni * Q)
    newNp = poisson_dist(newN)
    return newNp

def Ni_realnum_intuit(Ni, Ri, K,  Q):
    newN = Ni + (Ni * Ri) + (Ni * Q)
    return newN
    
# =============================================================================
# growth rates
# =============================================================================
def Ri_model_I(Rmax, species, **kwargs):
    """For non-integer runs for intuitive plots"""
    Rm = Rmax * (1-(species.Nm / species.Km))
    Rf = Rmax * (1-(species.Nf / species.Kf))
    if Rm <0:
        Rm = 0
    if Rf<0:
        Rf =0
    return Rf, Rm
 
def Ri_model_A(Rmax, species, **kwargs):
    Rm = Rmax * (1-(species.Nm / species.Km))
    Rf = Rmax * (1-(species.Nf / species.Kf))
    return Rf, Rm

def Ri_model_alleeA(Rmax, species, **kwargs):
    if "allee_params_theta_upsil" in kwargs.keys():
        theta, upsil = kwargs["allee_params_theta_upsil"]
    else: theta, upsil = 100, 0.9
    def pc_growth(R, N, K, theta, upsil):
        return R - ((R * N) / K) - ((theta * upsil) / (theta + N))
    Rm = pc_growth(Rmax, species.Nm, species.Km, theta, upsil)
    Rf = pc_growth(Rmax, species.Nf, species.Kf, theta, upsil)
    return Rf, Rm

def Ri_model_B(Rmax, species, **kwargs):
    Rm = Rmax * (1 - ((species.Nm+species.Nf)/(species.Km+species.Kf)))
    Rf = Rmax * (1 - ((species.Nm+species.Nf)/(species.Km+species.Kf)))
    return Rf, Rm

def Ri_model_C(Rmax, species, **kwargs):
    if "Rgen_model" in kwargs.keys():
        Rgen_model = kwargs["Rgen_model"]
    else:
        Rgen_model = Ri_model_B
        
    if species.Sa is None or species.B is None:
        raise Exception("Parameters 'SA' and 'B' undefined in ModelC run.")
    else:
        Sa, B = species.Sa,species.B
    tsp = copy.copy(species)
    if len(species.Nm_hist) > B:
        fecundity_factor = min([species.Nm_hist[-B+1]/species.Nf_hist[-B+1],
                                    species.Nf_hist[-B+1]/species.Nm_hist[-B+1]])
        tsp.Nm = species.Nm_hist[-B+1]
        tsp.Nf = species.Nf_hist[-B+1]
        t1m = species.Nm_hist[-B+1]/species.Nm
        t1f = species.Nf_hist[-B+1]/species.Nf
    else:
        if len(species.Nm_hist) == 0:
            t1m, t1f = 1, 1
            tsp.Nm = species.Nm
            tsp.Nf = species.Nf
        else:
            t1m = species.Nm_hist[-1]/species.Nm 
            t1f = species.Nf_hist[-1]/species.Nf
            tsp.Nm = species.Nm_hist[-1]
            tsp.Nf = species.Nf_hist[-1]
        fecundity_factor = 1

    Rm, Rf = Rgen_model(species.Rmax, tsp)
    t2m = Rm + 1 - Sa
    Rm_prime = Sa - 1 + (t1m * t2m * fecundity_factor)
    t2f = Rf + 1 - Sa
    Rf_prime = Sa - 1 + (t1f * t2f * fecundity_factor)
    return Rf_prime, Rm_prime

# =============================================================================
# Extras
# =============================================================================
def getB(Rmax, Sa): # small birds
    B = (1/Rmax) - (Sa / (np.exp(Rmax) - Sa))
    if np.isinf(B):
        B = None
    else: 
        B = round(B)
    return B

def get_new_K(current_K,
              varK_schedule=None,
              varK_strength=None,
              varK_schedule_fn=None):
    
    if varK_schedule is not None:
        if varK_schedule == "increase":
            new_K = current_K * (1 + varK_strength)
        elif varK_schedule == "decrease":
            new_K = current_K * max(0, 1 - varK_strength)
        elif varK_schedule == "random_walk":
            new_K = max(0, current_K + np.random.normal(0, varK_strength * current_K))
        elif varK_schedule == "custom":
            new_K = varK_schedule_fn(current_K, varK_strength)
        else:
            raise ValueError(f"Unknown varK_schedule: {varK_schedule}")
    return new_K