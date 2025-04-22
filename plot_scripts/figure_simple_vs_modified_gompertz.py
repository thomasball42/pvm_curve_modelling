# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:16:08 2025

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

clip_x = False

# my onedrive path, computer dependent..
od_path = "C:\\Users\\Thomas Ball\\OneDrive - University of Cambridge"
# od_path = "E:\\OneDrive\\OneDrive - University of Cambridge"

# dir that the simulation outputs are in
dat_path = os.path.join(od_path, "Work\\P_curve_shape\\version2\\dat\\simulation_results\\results_curve_comparison")

f = []
for path, subdirs, files in os.walk(dat_path):
    for name in files:
        f.append(os.path.join(path, name))


fig, axs = plt.subplots(1, len(f), sharey=True)


for fi, file in enumerate(f):
    
    ax = axs[fi]
    dat = pd.read_csv(file)
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
    
    
    # find the first non 0 y. 
    dat_offset = 5
    do2 = 5
    c = matplotlib.cm.get_cmap('viridis')
    
    kwargs = {"linewidth" : 3, 
              "alpha" : 0.99,
              "linestyle" : "-"}
    
    if clip_x:
        ygt0 = (y > 0).argmax()
        ylt1 = (y < 1).argmin()
        yy = y[ygt0 - dat_offset - do2 if dat_offset + do2 < ygt0 else 0 : ylt1 + dat_offset if ylt1 + dat_offset < len(y) else len(y)]
        xx = x[yy.index].to_numpy()
    else:
        xx = x
        yy = y
        
    # Plot points
    ax.scatter(xx, 1 - yy, color="k", alpha = 0.6, marker = "o", s = 50)
    
    xff = np.geomspace(xx.min(), xx.max(), num = 100000)
    # scatter_color = ax.collections[-1].get_facecolor()
    
    # CURVE GOMP
    def nnnn(r2):
        nnnn = 0 #batman
        while round(R2, nnnn) == 1:
            nnnn += 1
        return nnnn
    
    
    label = f"Modified gompertz  (R2:{round(R2, nnnn(R2)+1)})"
    ax.plot(xff, 1 - func(xff, *params), color = "c", label = label , **kwargs)
    
    
    func4 = _curve_fit.basic_gomp
    params4, y_predicted4, R2_4, residuals_4 = _curve_fit.fit(func4, xx, yy)
    label = f"Simple gompertz  (R2:{round(R2_4, nnnn(R2_4)+1)})"
    ax.plot(xff, 1 - func4(xff, *params4), color = "m", label = label, **kwargs)
    

    ax.set_xscale("log", base = 2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=2.0, numticks=4))
    def custom_formatter(xx, pos):
        return f'$10^{{{int(np.log10(xx))}}}$'
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(custom_formatter))
    
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(10, 4096)
    ax.legend()


fig.text(0.5, 0, "Carrying capacity K", ha= "center",va = "bottom")
axs[0].set_ylabel(f"Probability of extinction (N={round(N)})")    
fig.set_size_inches(10, 5)
fig.tight_layout()
