# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:52:42 2024

@author: tom
"""

# import numpy as np
# # import math

# =============================================================================
# Population
# =============================================================================
class Population:
    def __init__(self, K, B, Rmax, Sa):
        self.K, self.B, self.Rmax, self.Sa = K,B,Rmax,Sa
        self.Km, self.Kf = K/2,K/2
        self.Nm = self.Km
        self.Nf = self.Kf
        self.Nm_hist = []
        self.Nf_hist = []
        self.Rm_hist = []
        self.Rf_hist = []
        self.EXTANT = True
        self.RUNABORT = False
    
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
        newNf, newNm = modelN(Nf, Rf, Kf, Q), modelN(Nm, Rm, Km, Q)
        
        if isinstance(newNf, type(None)) or isinstance(newNm, type(None)):
            if newNf == None or newNm == None:
                self.RUNABORT = True
                # self.Nm, self.Nf = 0, 0
        else:
            if newNf < 1 or newNm < 1:
                self.EXTANT = False
            else:
                self.Nf, self.Nm = newNf, newNm
            
        