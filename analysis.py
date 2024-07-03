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

plot_curves = False

r_threshold = 0.9999

od_path = "C:\\Users\\Thomas Ball\\OneDrive - University of Cambridge"
# od_path = "E:\\OneDrive\\OneDrive - University of Cambridge"

results_path = os.path.join(od_path, "Work\\P_curve_shape\\dat\\results_log_ABC")

# =============================================================================
# Load data
# =============================================================================
f = []
for path, subdirs, files in os.walk(results_path):
    for name in files:
        f.append(os.path.join(path, name))
f = [file for file in f if ".csv" in file]

#%%
ddf = pd.DataFrame()

fig, ax = plt.subplots()

fx = []
fy = []

for file in f[10:15]:
    dat = pd.read_csv(file, index_col = 0)
    runName = dat.runName.unique().item()
    model = runName.split("_")[0]
    Q = dat.Q.unique().item()
    rmax = dat.Rmax.unique().item()
    B = dat.B.unique().item()
    x = dat.K
    y = dat.P
    try:
        Sa = dat.Sa.unique().item()
    except AttributeError:
        Sa = None
        
        
    func = _curve_fit.gompertz
    param_names = ("param_a", "param_b", "param_alpha")
    ret = _curve_fit.betterfit_gompertz(func, x, y, 
                            alpha_space = np.arange(0, 5, 0.05), 
                            plot_lins=False,
                            ylim=(0,1))
    
    if ret == None:
        ret = _curve_fit.betterfit_gompertz(func, x, y, 
                                alpha_space = np.arange(0, 10, 0.01),
                                ylim=(0.2, 0.8),
                                plot_lins=True)
        
        
    if ret == "MARK":
        print(runName)
    
    
    if ret != None and not ret == "MARK":    
        params, y_predicted, R2 = ret
    else:
        params, R2 = tuple([np.nan for _ in param_names]), np.nan
            
    ddf.loc[len(ddf), ["model", "rmax", "Q", "B", "Sa", 
                        *param_names, "R2"]] = [
                        model, rmax, Q, B, Sa, 
                        *params, R2]
                            
    # func = _curve_fit.weibull
    # param_names = ("param_lam", "param_k", "param_gam")
    # lam_space = np.linspace(0, 100, 20)
    # k_space = np.linspace(0.5, 5, 20)
    # gam_space = np.linspace(0, 100, 20)
    # try:
    #     params, y_predicted, R2 = _curve_fit.betterfit_weibull(func, x, y,
    #                                                             lam_space=lam_space,
    #                                                             k_space=k_space,
    #                                                             gam_space=gam_space,
    #                                                             plot_lins = False)
    # except RuntimeError:
    #     params, R2 = (np.nan, np.nan, np.nan), np.nan
    # ddf.loc[len(ddf), ["model", "rmax", "Q", "B", "Sa", 
    #                     *param_names, "R2"]] = [
    #                     model, rmax, Q, B, Sa, 
    #                     *params, R2]
    
                            
    # func = _curve_fit.stretched_exp
    # param_names = ("param_a", "param_b", "param_c")
    # a_space = np.linspace(0, 100, 20)
    # b_space = np.linspace(-5, 0.5, 20)
    # c_space = np.linspace(0, 100, 20)
    # try:
    #     params, y_predicted, R2 = _curve_fit.betterfit_stretched_exp(func, x, y,
    #                                                                   a_space=a_space,
    #                                                                   b_space=b_space,
    #                                                                   c_space=c_space)
    # except RuntimeError:
    #     params, R2 = (np.nan, np.nan, np.nan), np.nan   
    # ddf.loc[len(ddf), ["model", "rmax", "Q", "B", "Sa", 
    #                     *param_names, "R2"]] = [
    #                     model, rmax, Q, B, Sa, 
    #                     *params, R2]
    
    # func = _curve_fit.DR_eq2
    # param_names = ("param_a", "param_b", "param_c")
    # a_space = np.linspace(-5000, 40000, 50)
    # b_space = np.linspace(-50, 50, 100)
    # c_space = np.linspace(0, 4, 10)
    # try:
    #     params, y_predicted, R2 = _curve_fit.fitDR_curves(func, x, y,
    #                                                                   a_space=a_space,
    #                                                                   b_space=b_space,
    #                                                                   c_space=c_space,
    #                                                                   plot_lins=False)
    # except RuntimeError:
    #     params, R2 = (np.nan for p in param_names), np.nan   
    # ddf.loc[len(ddf), ["model", "rmax", "Q", "B", "Sa", 
    #                     *param_names, "R2"]] = [
    #                     model, rmax, Q, B, Sa, 
    #                     *params, R2]
    
    # func = _curve_fit.DR_eq3
    # param_names = ("param_a",)
    # a_space = np.linspace(-10, 100, 500)
    # try:
    #     params, y_predicted, R2 = _curve_fit.fitDR_curves_3(func, x, y,
    #                                                                   a_space=a_space,
    #                                                                   plot_lins=False)
    # except RuntimeError:
    #     params, R2 = np.nan, np.nan 
        
    # ddf.loc[len(ddf), ["model", "rmax", "Q", "B", "Sa", 
    #                     *param_names, "R2"]] = [
    #                     model, rmax, Q, B, Sa, 
    #                     *params, R2]
    
                            
    if plot_curves and np.isnan(R2):
        fx.append(x)
        fy.append(y)
        ax.scatter(x, y)
        xff = np.linspace(dat.K.min(), dat.K.max(), num = 10000)
        scatter_color = ax.collections[-1].get_facecolor()
        ax.plot(xff, func(xff, *params), color = scatter_color)
        ax.set_xscale("log", base = 2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.legend()
    
    # print(len(ddf[~np.isnan(ddf.R2)]), len(ddf))
        
#%% ===========================================================================
# # do some analysing
# # =============================================================================
# for model in ddf.model.unique():
    
#     df = ddf[ddf.model == model]
    
#     input_params = df[['rmax', 'Q']]
#     model_params = df[['param_a', 'param_b', 'param_alpha']]
    
#     fig, axs = plt.subplots(len(input_params.T), len(model_params.T))
#     for i, ip in enumerate(input_params):
#         for m, mp in enumerate(model_params):
#             ax = axs[i, m]

#             x, y, z = df[[ip, mp, input_params.columns.difference([ip]).item()]].values.T
            
#             ax.scatter(x, y, c=z, cmap='viridis', alpha=0.7)
#             slope, inter, r, p_value, std_err = scipy.stats.linregress(x, y)
#             ax.plot(x, _curve_fit.lin(x, slope, inter), label = f"r2:{round(r**2, 2)}")
#             ax.set_xlabel(ip)
#             ax.set_ylabel(mp)
#             ax.legend()
            
#     fig = plt.gcf()
#     fig.set_size_inches(8, 6)
#     fig.suptitle(f"Model '{model}'", fontsize=14)
#     fig.tight_layout()
    
    
# # %% =============================================================================
# # def gaussianKernelDensity(x, y, z, density):
# #     x = x[~np.isnan(z)]
# #     y = y[~np.isnan(z)]
# #     z = z[~np.isnan(z)]
    
# #     pdf = kde(np.vstack([x, y]), weights = z)
# #     xg, yg = np.meshgrid(np.linspace(x.min(),x.max(),density),
# #                           np.linspace(y.min(),y.max(),density))
# #     kz = pdf(np.vstack([xg.flatten(), yg.flatten()]))
# #     zg = kz.reshape(xg.shape)
    
    
# #     return xg, yg, zg, pdf

# # def predict_z(pdf, x_val, y_val):
# #     # Evaluate the pdf at the given (x, y) point
# #     z_pred = pdf(np.vstack([x_val, y_val]))
# #     return z_pred

# # for model in ddf.model.unique()[:1]:
    
# #     df = ddf[ddf.model == model]
    
# #     input_params = df[['rmax', 'Q']]
# #     model_params = df[['param_a', 'param_b', 'param_alpha']]
    
# #     fig = plt.figure() 
    
# #     for m, mp in enumerate(model_params):
        
# #         ax = fig.add_subplot(1, len(model_params.columns), m+1, projection='3d')

# #         x, y = input_params.values.T 
# #         z = np.array(df[mp])
        
# #         xg, yg, zg, pdf = gaussianKernelDensity(x, y, z, 17)
        
# #         ax.scatter(x, y, z, alpha = 0.7, s = 40)
# #         ax.set_title(mp)
# #         ax.plot_surface(xg, yg, zg)
# #         ax.set_xlabel(input_params.columns[0])
# #         ax.set_ylabel(input_params.columns[1])
        
# #     fig.set_size_inches(8, 6)
# #     fig.suptitle(f"Model '{model}'", fontsize=14)
# #     fig.tight_layout()


#%%

# density = 30

# def interpolateZ(x, y, z, density):
#     x = x[~np.isnan(z)]
#     y = y[~np.isnan(z)]
#     z = z[~np.isnan(z)]
#     xg, yg = np.meshgrid(np.linspace(x.min(), x.max(), density), 
#                           np.linspace(y.min(), y.max(), density))
#     zg = griddata((x, y), z, (xg, yg), method='cubic')
#     return xg, yg, zg

# def predict_z(x, y, z, x_val, y_val, method='cubic'):
#     x = x[~np.isnan(z)]
#     y = y[~np.isnan(z)]
#     z = z[~np.isnan(z)]
#     z_pred = griddata((x, y), z, (x_val, y_val), method=method)
#     return z_pred

# for model in ddf.model.unique():
    
#     df = ddf[ddf.model == model]
    
#     input_params = df[['rmax', 'Q']]
#     model_params = df[['param_a', 'param_b', 'param_alpha']]
    
#     fig = plt.figure()
    
#     for m, mp in enumerate(model_params):
        
#         ax = fig.add_subplot(1, len(model_params.columns), m+1, projection='3d')

#         x, y = input_params.values.T
#         z = np.array(df[mp])
        
#         xg, yg, zg = interpolateZ(x, y, z, density)
        
#         ax.scatter(x, y, z, alpha=0.4, s=30)
#         ax.set_title(mp)
#         # ax.plot_surface(xg, yg, zg, cmap='viridis', alpha=0.5)
#         ax.set_xlabel(input_params.columns[0])
#         ax.set_ylabel(input_params.columns[1])
#         ax.set_zlabel(mp)
        
#     fig.set_size_inches(8, 6)
#     fig.suptitle(f"Model '{model}'", fontsize=14)
#     # fig.tight_layout()
#     plt.show()

# # #%%

# # # import numpy.linalg as lg 

# # # def fit_surface(x, y, z):
    
# # #     def design_mat(x, y, order):
# # #         A = np.zeros((x.size, (order + 1) * (order + 2) // 2))
# # #         index = 0
# # #         for i in range(order + 1):
# # #             for j in range(order + 1 - i):
# # #                 A[:, index] = (x ** i) * (y ** j)
# # #                 index += 1
# # #         return A
    
# # #     A = design_mat(x, y, 3)

    
    
    

    












