# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:55:22 2024

@author: Thomas Ball
"""

import os
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



od_path = "C:\\Users\\Thomas Ball\\OneDrive - University of Cambridge"

results_path = os.path.join(od_path, "Work\\P_curve_shape\\results")

f = []
for path, subdirs, files in os.walk(results_path):
    for name in files:
        f.append(os.path.join(path, name))

f = [file for file in f if ".csv" in file]