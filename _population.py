# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:52:42 2024

@author: tom
"""

import numpy as np
import math

# =============================================================================
# Population
# =============================================================================
class Population:
    def __init__(self, K, Sa, B, Rmax):
        self.K, self.Sa, self.B, self.Rmax = K,Sa,B,Rmax
        self.Km, self.Kf = K/2,K/2
        self.Nm = self.Km
        self.Nf = self.Kf
        self.Nm_hist = []
        self.Nf_hist = []
        self.Rm_hist = []
        self.Rf_hist = []
    
    def iterate(self, modelR, modelN, Q, **kwargs):
        """
        Takes the desired growth model and a Q value.
        """
        Nm, Nf, Km, Kf = self.Nm,self.Nf,self.Km,self.Kf
        Rf, Rm = modelR(self.Rmax, self, **kwargs)
        self.Nf_hist.append(Nf)
        self.Nm_hist.append(Nm)
        self.Rf_hist.append(Rf)
        self.Rm_hist.append(Rm)
        self.Nf = modelN(Nf, Rf, Kf, Q)
        self.Nm = modelN(Nm, Rm, Km, Q)