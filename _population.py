# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:52:42 2024

@author: tom
"""

# =============================================================================
# Population
# =============================================================================
class Population:
    def __init__(self, K, B=None, Rmax=None, Sa=None, N0=0):
        self.K, self.B, self.Rmax, self.Sa = K, B, Rmax, Sa
        self.Km, self.Kf = K / 2, K / 2
        self.Nm = N0 / 2
        self.Nf = N0 / 2
        self.Nm_hist = []
        self.Nf_hist = []
        self.Rm_hist = []
        self.Rf_hist = []
        self.Q_hist = []
        self.EXTANT = True
        self.RUNABORT = False
    
    def iterate(self, modelR, modelN, modelQ, **kwargs):
        """
        Takes the desired growth model and a Q value.
        """
        Nm, Nf, Km, Kf = self.Nm,self.Nf,self.Km,self.Kf
        Rf, Rm = modelR(self.Rmax, self, **kwargs)
        self.Nf_hist.append(Nf)
        self.Nm_hist.append(Nm)
        self.Rf_hist.append(Rf)
        self.Rm_hist.append(Rm)
        Q = modelQ(*kwargs["q_params"], self.Q_hist)
        self.Q_hist.append(Q)
        newNf, newNm = modelN(Nf, Rf, Kf, Q), modelN(Nm, Rm, Km, Q)
        if isinstance(newNf, type(None)) or isinstance(newNm, type(None)):
            if newNf == None or newNm == None:
                self.RUNABORT = True
        else:
            if newNf < 1 or newNm < 1:
                self.EXTANT = False
                self.Nf, self.Nm = newNf, newNm
            else:
                self.Nf, self.Nm = newNf, newNm
                
            
        