# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:55:22 2024

@author: Thomas Ball
"""

import os
import pandas as pd
import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.stats import gaussian_kde as kde
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

import _curve_fit

scale_1_0 = False
plot_pspace = False
plot_curves = True
overwrite = True
all_out = True

r2_threshold = 0.999

od_path = "C:\\Users\\Thomas Ball\\OneDrive - University of Cambridge"
# od_path = "E:\\OneDrive\\OneDrive - University of Cambridge"

results_path = os.path.join(od_path, "Work\\P_curve_shape\\dat\\results_log_ABC")
data_fits_path = os.path.join(od_path, "Work\\P_curve_shape\\dat\\data_fits.csv")

# =============================================================================
# Load data
# =============================================================================
f = []
for path, subdirs, files in os.walk(results_path):
    for name in files:
        f.append(os.path.join(path, name))
f = [file for file in f if ".csv" in file]
# f = [k for k in f if "LogGrowthC" in k]
f = [k for k in f if "Rmax0.2" in k]
f = [k for k in f if "nan" in k or "SA0.55" in k]
Qlim = 0.2

#%%
if os.path.isfile(data_fits_path) and not overwrite:
    data_fits = pd.read_csv(data_fits_path, index_col=0)
else:
    data_fits = pd.DataFrame()
    

ddf = pd.DataFrame()

fig, ax = plt.subplots()

fx = []
fy = []

for i, file in enumerate(f[:]):
    
    dat = pd.read_csv(file, index_col = 0)
    runName = dat.runName.unique().item()
    
    if not data_fits.empty and not overwrite and runName in data_fits.runName.to_list():
        R2 = data_fits[data_fits.runName == runName].R2.squeeze()
        if R2 > r2_threshold:
            print(f"Skipping {runName}")
            continue
    else:
        ## SETUP
        model = runName.split("_")[0]
        Q = dat.Q.unique().item()
        if Q > Qlim:
            continue
        rmax = dat.Rmax.unique().item()
        B = dat.B.unique().item()
        x = dat.K 
        y = dat.P
        k50 = np.percentile(dat.P, 50)
        
        reversed_arr = y[::-1]
        first_non_one_index = np.argmax(reversed_arr != 1)
        tail_start_index = len(y) - first_non_one_index
        
        if scale_1_0:
            y = y[:tail_start_index]
            if len(x) == tail_start_index:
                x = x/ x.max()
            else:
                x = x[:tail_start_index] / x[tail_start_index]
        
        try:
            Sa = dat.Sa.unique().item()
        except AttributeError:
            Sa = None
            
        # initialise
        fit = False
        model_name = np.nan
        R2 = np.nan
            
        # TRY GOMPERTZ
        func = _curve_fit.mod_gompertz
        param_names = ("param_a", "param_b", "param_alpha")
        params = tuple([np.nan for _ in param_names])
        ret = _curve_fit.betterfit_gompertz(func, x, y, 
                                alpha_space = np.arange(0, 5, 0.005), 
                                ylim=(0.05, 0.95), 
                                plot_lins=False,)
        if ret == None:
            ret = _curve_fit.betterfit_gompertz(func, x, y, 
                                    alpha_space = np.arange(-5, 0, 0.005),
                                    # ylim=(0.2, 0.8), 
                                    plot_lins=False)
        if not fit and not ret == None:
            fit = True
            params, y_predicted, R2 = ret
            model_name = func.__name__


        # calc k50
        xff = np.linspace(dat.K.min(), dat.K.max(), num = 10000)
        yff = func(xff, *params)
        gt50 = xff[yff >= 0.5]
        if len(gt50) > 0:
            k50 = gt50[0]
        else:
            k50 = np.nan
        
        # APPEND DATA TO OUTPUT
        ddf.loc[len(ddf), ["model", "runName", "rmax", "Q", "B", "Sa", 
                            "model_name", *param_names, "R2", "k50"]] = [
                            model, runName, rmax, Q, B, Sa, 
                            model_name, *params, R2, k50]
        
        # # PLOT CURVES AND FITS
        if plot_curves:
            label = f"Model {model.strip('LogGrowth')}"
            label = f"{runName}"
            fx.append(x)
            fy.append(y)
            ax.scatter(x, y)
            xff = np.linspace(x.min(), x.max(), num = 10000)
            scatter_color = ax.collections[-1].get_facecolor()
            ax.plot(xff, func(xff, *params), color = scatter_color, label = label)
            if not scale_1_0:
                ax.set_xscale("log", base = 2)
                ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.legend()
            ax.set_ylabel("Probability of persistence (N=10000)")
            ax.set_xlabel("Carrying capacity K")
            
        if plot_pspace:
            if np.isnan(R2):
                c = "r"
            else:
                c = matplotlib.cm.get_cmap('viridis')((R2 - 0.990)/(1-0.990))
            ax.scatter(Q, rmax, color = c, s = 70)
            ax.set_ylabel("R$_{max}$")
            ax.set_xlabel("Stochasicity SD")
if plot_pspace:
    norm = plt.Normalize(0.990, 1)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('R2')
    
fig.set_size_inches(8, 6)
fig.tight_layout()

if all_out:
    out_df = ddf
else:
    out_df = ddf[ddf.R2 > r2_threshold]
data_fits = pd.concat([data_fits, out_df])
data_fits.to_csv(data_fits_path)

quit()

#%% =============================================================================
# K50
# ===============================================================================
# fig, ax = plt.subplots()

# for q, Q in enumerate(ddf.Q.unique()):
#     for rm, rmax in enumerate(ddf.rmax.unique()):
#         d = ddf[(ddf.Q==Q)&(ddf.rmax==rmax)]
        
#         ax.scatter(Q, d.k50)
        







#%% =============================================================================
# # do some analysing
# # =============================================================================
for model in ddf.model.unique():
    
    df = ddf[ddf.model == model]
    
    input_params = df[['rmax', 'Q']]
    model_params = df[['param_a', 'param_b', 'param_alpha']]
    
    fig, axs = plt.subplots(len(input_params.T), len(model_params.T))
    for i, ip in enumerate(input_params):
        for m, mp in enumerate(model_params):
            ax = axs[i, m]

            x, y, z = df[[ip, mp, input_params.columns.difference([ip]).item()]].values.T
            
            ax.scatter(x, y, c=z, cmap='viridis', alpha=0.7)
            slope, inter, r, p_value, std_err = scipy.stats.linregress(x, y)
            ax.plot(x, _curve_fit.lin(x, slope, inter), label = f"r2:{round(r**2, 2)}")
            ax.set_xlabel(ip)
            ax.set_ylabel(mp)
            ax.legend()
            
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    fig.suptitle(f"Model '{model}'", fontsize=14)
    fig.tight_layout()
    
    
# # %% =============================================================================
# def gaussianKernelDensity(x, y, z, density):
#     x = x[~np.isnan(z)]
#     y = y[~np.isnan(z)]
#     z = z[~np.isnan(z)]
    
#     pdf = kde(np.vstack([x, y]), weights = z)
#     xg, yg = np.meshgrid(np.linspace(x.min(),x.max(),density),
#                           np.linspace(y.min(),y.max(),density))
#     kz = pdf(np.vstack([xg.flatten(), yg.flatten()]))
#     zg = kz.reshape(xg.shape)
    
    
#     return xg, yg, zg, pdf

# def predict_z(pdf, x_val, y_val):
#     # Evaluate the pdf at the given (x, y) point
#     z_pred = pdf(np.vstack([x_val, y_val]))
#     return z_pred

# for model in ddf.model.unique()[:1]:
    
#     df = ddf[ddf.model == model]
    
#     input_params = df[['rmax', 'Q']]
#     model_params = df[['param_a', 'param_b', 'param_alpha']]
    
#     fig = plt.figure() 
    
#     for m, mp in enumerate(model_params):
        
#         ax = fig.add_subplot(1, len(model_params.columns), m+1, projection='3d')

#         x, y = input_params.values.T 
#         z = np.array(df[mp])
        
#         xg, yg, zg, pdf = gaussianKernelDensity(x, y, z, 17)
        
#         ax.scatter(x, y, z, alpha = 0.7, s = 40)
#         ax.set_title(mp)
#         ax.plot_surface(xg, yg, zg)
#         ax.set_xlabel(input_params.columns[0])
#         ax.set_ylabel(input_params.columns[1])
        
#     fig.set_size_inches(8, 6)
#     fig.suptitle(f"Model '{model}'", fontsize=14)
#     fig.tight_layout()


#%%

density = 30

def interpolateZ(x, y, z, density):
    x = x[~np.isnan(z)]
    y = y[~np.isnan(z)]
    z = z[~np.isnan(z)]
    xg, yg = np.meshgrid(np.linspace(x.min(), x.max(), density), 
                          np.linspace(y.min(), y.max(), density))
    zg = griddata((x, y), z, (xg, yg), method='cubic')
    return xg, yg, zg

def predict_z(x, y, z, x_val, y_val, method='cubic'):
    x = x[~np.isnan(z)]
    y = y[~np.isnan(z)]
    z = z[~np.isnan(z)]
    z_pred = griddata((x, y), z, (x_val, y_val), method=method)
    return z_pred

for model in ddf.model.unique():
    
    df = ddf[ddf.model == model]
    
    input_params = df[['rmax', 'Q']]
    ipl = {"rmax": "R$_{max}$",
            "Q"  : "S-SD"}
    model_params = df[['param_a', 'param_b', 'param_alpha']]
    
    fig = plt.figure()
    
    for m, mp in enumerate(model_params):
        
        ax = fig.add_subplot(1, len(model_params.columns), m+1, projection='3d')

        x, y = input_params.values.T
        z = np.array(df[mp])
        
        xg, yg, zg = interpolateZ(x, y, z, density)
        
        # ax.scatter(x, y, z, alpha=0.9, s=30, c = matplotlib.cm.get_cmap('viridis')(0.3) )
        ax.set_title(mp)
        ax.plot_surface(xg, yg, zg, alpha=0.5, cmap = matplotlib.cm.get_cmap('viridis'))
        ax.set_xlabel(ipl[input_params.columns[0]])
        ax.set_ylabel(ipl[input_params.columns[1]])
        ax.set_zlabel(mp)
        
    fig.set_size_inches(12, 6)
    fig.suptitle(f"Model '{model}'", fontsize=14)
    fig.tight_layout(pad = 2.2)
    plt.show()


    
    
    

    












