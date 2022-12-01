
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:04:04 2021

@author: macbjmu
"""
import numpy as np
from scipy.stats import nbinom
from scipy.optimize import fsolve
from scipy import special
import scipy.integrate as integrate
from scipy.stats import gamma
import pymc3 as pm
import arviz as az
import csv
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


def gamma2discrete(mean_GT,sd_GT,MaxInfctPrd):
    
    '''
    Parameters
    ----------
    mean_GT : float
        The mean of generation time.
    sd_GT : float
        the sd of generation time.
    MaxInfctPrd : TYPE
        DESCRIPTION.

    Returns
    -------
    np.array
        the ratio for each .

    '''
    # shape_para = sd_GT/mean_GT**2;
    # rate_para  = shape_para/mean_GT
    shape_para = (mean_GT/sd_GT)**2;
    rate_para  = shape_para/mean_GT
    
    wRatio = np.zeros(MaxInfctPrd+1)
    for tmps in range(1,len(wRatio)):
        wRatio[tmps] = gamma.cdf(tmps+0.5,shape_para,scale = 1/rate_para) - gamma.cdf(tmps-0.5,shape_para,scale = 1/rate_para)
    
    return wRatio/np.sum(wRatio);

def tvInfer(IncData_impt,FOI_impt):
    
    # new model
    basic_model = pm.Model()
    # nb_r,nb_p = np.zeros(dta_Num),np.zeros(dta_Num)
    with basic_model:
        k_para =  pm.Uniform("k_disp",0,100)
        Rt_para = pm.Uniform("Rt", 0.1, 10);
        nb_r = FOI_impt*k_para;
        nb_p = k_para/(Rt_para+k_para);    
        Y_obs = pm.NegativeBinomial("Y_obs", n=nb_r, p=nb_p, observed=IncData_impt)
        trace = pm.sample(10000, return_inferencedata=False,cores = 1);
    
    map_estimate = pm.find_MAP(model=basic_model)
    k_hpd = az.hdi(trace["k_disp"],hdi_prob = 0.95)
    R_hpd = az.hdi(trace["Rt"],hdi_prob = 0.95)
    k_median = np.median(trace["k_disp"])
    
    # pop Model
    basic_model_pop = pm.Model()
    with basic_model_pop:
        k_para =  pm.Uniform("k_disp_pop",0,100)
        Rt_para = pm.Uniform("Rt_pop", 0.1, 10);
        nb_r = k_para;
        nb_p = k_para/(Rt_para*FOI_impt+k_para)       
        Y_obs = pm.NegativeBinomial("Y_obs_pop", n=nb_r, p=nb_p, observed=IncData_impt)
        trace = pm.sample(10000, return_inferencedata=False,cores = 1);
    map_estimate_pop = pm.find_MAP(model=basic_model_pop)
    k_pop_hpd = az.hdi(trace["k_disp_pop"],hdi_prob = 0.95)
    R_pop_hpd = az.hdi(trace["Rt_pop"],hdi_prob = 0.95)
    
    
    return [map_estimate['k_disp'],k_hpd,map_estimate['Rt'],R_hpd,\
            map_estimate_pop['k_disp_pop'],k_pop_hpd,\
            map_estimate_pop['Rt_pop'],R_pop_hpd];


'''
1. given the input information
'''

mean_GT = 5.2;
sd_GT   = 1.72;
MaxInfctPrd = int(mean_GT+3*sd_GT)+1; # maximum length of infectious period in day


'''
2. perform inference for each area
'''
Wratios = gamma2discrete(mean_GT,sd_GT,MaxInfctPrd)
wndw_len = 7;
arNum = 5
Re_5areas = []
Areas = ['Cobb','DeKalb','Fulton','Gwinnett','Dougherty']

dataPath = '/Users/macbjmu/Documents/research/NewIdeas/dynamic_Rt/dynaRt_code/data/'
dataFile = 'GA_data.csv'

for ar_i in range(arNum):
    tmpArea = Areas[ar_i]+'_IncData';

    IncData = pd.read_csv(dataPath+dataFile)[tmpArea].values
    tempFOI = signal.convolve(np.concatenate((np.zeros(len(Wratios)-1),IncData)),Wratios, mode='valid')    


    dta_Num = len(IncData)
    stDate = np.nonzero(IncData)[0][1]
    
    IncData_impt = IncData[stDate:dta_Num+1]
    FOI_impt     = tempFOI[stDate:dta_Num+1]
    
    tvRslt = []

    for i in range(len(IncData_impt)+1-wndw_len):
        Inc_wndw = IncData_impt[i:(wndw_len+i)];
        FOI_wndw = FOI_impt[i:(wndw_len+i)];
        tmpRe    = tvInfer(Inc_wndw,FOI_wndw)
        tvRslt += [tmpRe]
    Re_5areas += [tvRslt]
    
'''
3) plot the results
'''
plt.style.use('ggplot') # pretty matplotlib plots
lw = 1.5
f, axes = plt.subplots(3, arNum, sharex='col',sharey = 'row', dpi = 500)
for ar_i in range(arNum):
    tmpArea = Areas[ar_i]+'_IncData';
    IncData = pd.read_csv(dataPath+dataFile)[tmpArea].values
    # generate three pandas for plot
    pop_est = np.zeros((len(IncData),7))
    pop_est[:,0] = np.arange(len(IncData))+1
    pop_est[:,1:] = np.nan
    
    new_est  = pop_est.copy()
    

    
    tvRslt = Re_5areas[ar_i];
    # setting values for these pandas
    for i in range(1,1+len(tvRslt)):
        tmpRe = tvRslt[-i]
        new_est[-i][4],new_est[-i][5:7] = tmpRe[0:2]; # for k 
        new_est[-i][1],new_est[-i][2:4] = tmpRe[2:4]; # for R
        
        pop_est[-i][4],pop_est[-i][5:7] = tmpRe[4:6]; # for k 
        pop_est[-i][1],pop_est[-i][2:4] = tmpRe[6:8]; # for R

    
    axes[0][ar_i].plot(new_est[:,0],IncData,color = 'k',linewidth = lw) # for incidence data  
    if ar_i >0:
        axes[0][ar_i].tick_params(axis='both', which='both', length=0)
       # for R variation
    axes[1][ar_i].plot(new_est[:,0],new_est[:,1],linewidth = lw)
    axes[1][ar_i].fill_between(new_est[:,0],new_est[:,2],new_est[:,3],alpha = 0.3)
    axes[1][ar_i].plot(pop_est[:,0],pop_est[:,1],linewidth = lw)
    axes[1][ar_i].fill_between(pop_est[:,0],pop_est[:,2],pop_est[:,3],alpha = 0.3)
    axes[1][ar_i].set_yscale('log')
    axes[1][ar_i].set_yticks([0.1,1]) 
    axes[1][ar_i].minorticks_off()
    if ar_i >0:
        axes[1][ar_i].tick_params(axis='both', which='both', length=0)
    # for k variation
    axes[2][ar_i].plot(new_est[:,0],new_est[:,4],linewidth = lw)
    axes[2][ar_i].fill_between(new_est[:,0],new_est[:,5],new_est[:,6],alpha = 0.3)
    axes[2][ar_i].plot(pop_est[:,0],pop_est[:,4],linewidth = lw)
    axes[2][ar_i].fill_between(pop_est[:,0],pop_est[:,5],pop_est[:,6],alpha = 0.3) 
    # axes[2].set_yticks([1])
    axes[2][ar_i].set_yscale('log')
    axes[2][ar_i].set_yticks([0.1,1])    
    axes[2][ar_i].set_xticks([0,33,47])
    if ar_i >0:
        axes[2][ar_i].tick_params(axis='y', which='both', length=0)
f.subplots_adjust(hspace = 0.01,wspace = 0.01)

    
