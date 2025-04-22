# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:31:42 2025

@author: Thomas Ball
"""

import pandas as pd
import numpy as np
import os 
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker

sys.path.append("..")
import _curve_fit

# my onedrive path, computer dependent..
od_path = "C:\\Users\\Thomas Ball\\OneDrive - University of Cambridge"
# od_path = "E:\\OneDrive\\OneDrive - University of Cambridge"

# dir that the simulation outputs are in
dat_path = os.path.join(od_path, "Work\\P_curve_shape\\version2\\dat\\simulation_results\\results_curve_comparison")

f = []
for path, subdirs, files in os.walk(dat_path):
    for name in files:
        f.append(os.path.join(path, name))




dat = pd.read_csv(f[0])
runName = dat.runName.unique().item()

ddf = pd.DataFrame()

model = runName.split("_")[0]
qsd = dat.QSD.unique().item()
N = dat.N.unique().item()
x = dat.K 
y = dat.P




# initialise fitting
fit = False
model_name = np.nan
R2 = np.nan
resids = np.nan

# TRY GOMPERTZ
func = _curve_fit.mod_gompertz
param_names = ("param_a", "param_b", "param_alpha")
params = tuple([np.nan for _ in param_names])
ret = _curve_fit.betterfit_gompertz(func, x, y, 
                        alpha_space = np.arange(0, 5, 0.001), 
                        ylim=(0.05, 0.95), 
                        plot_lins=False,)
if ret == None:
    ret = _curve_fit.betterfit_gompertz(func, x, y, 
                            alpha_space = np.arange(-5, 0, 0.001),
                            ylim=(0.05, 0.95), 
                            plot_lins=False)
if not fit and not ret == None:
    fit = True
    params, y_predicted, R2, resids = ret
    model_name = func.__name__


#%%
fig, ax = plt.subplots()

# find the first non 0 y. 
dat_offset = 0
do2 = 0
c = matplotlib.cm.get_cmap('viridis')

kwargs = {"linewidth" : 2, 
          "alpha" : 0.99,
          "linestyle" : "-"}

ygt0 = (y > 0).argmax()
ylt1 = (y < 1).argmin()
yy = y[ygt0 - dat_offset - do2 if dat_offset + do2 < ygt0 else 0 : ylt1 + dat_offset if ylt1 + dat_offset < len(y) else len(y)]
xx = x[yy.index].to_numpy()

# Plot points
ax.scatter(xx, 1 - yy, color="k", alpha = 0.4, marker = "o", s = 50)

xff = np.geomspace(xx.min(), xx.max(), num = 100000)
# scatter_color = ax.collections[-1].get_facecolor()

# CURVE GOMP
def nnnn(r2):
    nnnn = 0 #batman
    while round(R2, nnnn) == 1:
        nnnn += 1
    return nnnn


label = f"Modified gompertz  (R2:{round(R2, nnnn(R2)+1)})"
ax.plot(xff, 1 - func(xff, *params), color = "g", label = label , **kwargs)


func2 = _curve_fit.logistic
params2, y_predicted2, R2_2, residuals_2 = _curve_fit.fit(func2, x, y, init_guess=[1, 0.5, 200])
label = f"Logistic  (R2:{round(R2_2, nnnn(R2_2)+1)})"
ax.plot(xff, 1 - func2(xff, *params2), color = c(0.5), label = label, **kwargs)


def power(x, z, x0, y0): 
    return y0 + ((x - x0) ** z)
def powerX(x, x0, y0): 
    return y0 + ((x - x0) ** 0.25)

func3 = powerX
params3, y_predicted3, R2_3, residuals_3 = _curve_fit.fit(func3, xx, yy, init_guess=[5, 1])
label = f"Power law [0.25]  (R2:{round(R2_3, nnnn(R2_3)+1)})"
ax.plot(xff, 1 - func3(xff, *params3), color = c(0.9), label = label, **kwargs)
# ax.plot(xff, 1 - func3(xff, 0.25, 0, 0), color = "g", label = label)


func4 = _curve_fit.basic_gomp
params4, y_predicted4, R2_4, residuals_4 = _curve_fit.fit(func4, xx, yy)
label = f"Simple gompertz  (R2:{round(R2_4, nnnn(R2_4)+1)})"
# ax.plot(xff, 1 - func4(xff, *params4), color = "m", label = label, **kwargs, linestyle = "--")
kwargs.pop("linestyle", None)
ax.plot(xff, 1 - func4(xff, *params4), color = "m", label = label, linestyle = "--", **kwargs, )

ax.set_xscale("log", base = 2)
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=2.0, numticks=10))
# def custom_formatter(xx, pos):
#     return f'$10^{{{int(np.log10(xx))}}}$'
# ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(custom_formatter))
ax.set_ylabel(f"Probability of extinction (N={round(N)})")
ax.set_xlabel("Carrying capacity K")

ax.set_ylim(-0.05, 1.05)
ax.legend()

fig.set_size_inches(8, 5.5)
fig.tight_layout()
