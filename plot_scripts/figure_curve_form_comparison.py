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

# dir that the simulation outputs are in
dat_path = "..\\results\\simulation_results\\results_curve_comparison"

f = []
for path, subdirs, files in os.walk(dat_path):
    for name in files:
        f.append(os.path.join(path, name))




dat = pd.read_csv(f[0])
runName = dat.runName.unique().item()

ddf = pd.DataFrame()

model = runName.split("_")[0]
qsd = dat.QSD.unique().item()
rmax = dat.RMAX.unique().item()
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

def powerX2(x, x0, z0):
    return (z0 *( (x - x0) ** 0.25))

func3 = powerX2
params3, y_predicted3, R2_3, residuals_3 = _curve_fit.fit(func3, xx[7:40], yy[7:40], init_guess=[20, 0.26])
label = f"Power law [0.25]  (R2:{round(R2_3, nnnn(R2_3)+1)})"
ax.plot(xff, 1.035 - func3(xff, *params3), color = c(0.9), label = label, **kwargs)
ax.plot((xff[0], 20), (1,1,),color = c(0.9))


def lande(x, a,  b0, c0):
    """
    P(EXT) = ( a  / K)^( Abs[ (rmax - s/2)  / s ] - (d - s/2)/ s )
    """
    poww = ( np.abs((b0-(0.5*c0))/c0)  + ((b0-(0.5*c0))/c0) )
    print(poww) 
    y = 1- (a / x) ** poww 
    y[y<0] = 0
    
    return y
    
func5 = lande
params5, y_predicted5, R2_5, residuals_5 = _curve_fit.fit(func5, xx[7:40], yy[7:40], init_guess=[10, rmax, qsd])
label = f"Lande  (R2:{round(R2_5, nnnn(R2_5)+1)})"
ax.plot(xff, 1-func5(xff, *params5), color = c(0.1), label = label, **kwargs)


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
