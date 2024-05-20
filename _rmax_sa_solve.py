# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:22:16 2024

@author: Thomas Ball
"""
import numpy as np

def funcR(R, S, B):
    return 1 / (B + (S / (np.exp(R) - S)))

def funcS(R, S, B):
    return ((1/R) - B) * (np.exp(R) - S)

def funcB(R, S, B):
    return (1/R) - (S / (np.exp(R) - S))

def solve(sub, R0, S0, B0, tolerance = 1E-6, max_iter = 100):
    if sub == "R":
        func = funcR
        R, S, B = R0,S0,B0
        for _ in range(max_iter):
            R1 = func(R, S, B)
            if abs(R1 - R) < tolerance:
                return R, S, B
            else:
                R = R1
            print(R)
    if sub == "S":
        func = funcS
        R, S, B = R0,S0,B0
        for _ in range(max_iter):
            S1 = func(R, S, B)
            if abs(S1 - S) < tolerance:
                return R, S, B
            else:
                S = S1
            print(S)
    if sub == "B":
        func = funcB
        R, S, B = R0,S0,B0
        for _ in range(max_iter):
            B1 = func(R, S, B)
            if abs(B1 - B) < tolerance:
                return R, S, B
            else:
                B = B1
            print(B)
        

    
        
    