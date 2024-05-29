# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:22:16 2024

@author: Thomas Ball
"""
import numpy as np

def funcR(R, Sa, B):
    return 1 / (B + (Sa / (np.exp(R) - Sa)))

def funcSa(R, Sa, B):
    return ((1/R) - B) * (np.exp(R) - Sa)

def funcB(R, Sa, B):
    return (1/R) - (Sa / (np.exp(R) - Sa))

def solve(sub, R0, Sa0, B0, tolerance = 1E-6, max_iter = 100):
    if sub == "R":
        func = funcR
        R, Sa, B = R0,Sa0,B0
        for _ in range(max_iter):
            R1 = func(R, Sa, B)
            if abs(R1 - R) < tolerance:
                return R, Sa, B
            else:
                R = R1
            print(R)
    if sub == "Sa":
        func = funcSa
        R, Sa, B = R0,Sa0,B0
        for _ in range(max_iter):
            Sa1 = func(R, Sa, B)
            if abs(Sa1 - Sa) < tolerance:
                return R, Sa, B
            else:
                Sa = Sa1
            print(Sa)
    if sub == "B":
        func = funcB
        R, Sa, B = R0,Sa0,B0
        for _ in range(max_iter):
            B1 = func(R, Sa, B)
            if abs(B1 - B) < tolerance:
                return R, Sa, B
            else:
                B = B1
            print(B)
        

    
        
    