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
import joypy
from matplotlib import cm
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
    MaxInfctPrd : int
        the maximum length of infectious period.

    Returns
    w_s, the infectivity profile

    '''
    shape_para = sd_GT/mean_GT**2;
    rate_para  = shape_para/mean_GT
    
    wRatio = np.zeros(MaxInfctPrd+1)
    for tmps in range(1,len(wRatio)):
        wRatio[tmps] = gamma.cdf(tmps+0.5,shape_para,scale = 1/rate_para) - gamma.cdf(tmps-0.5,shape_para,scale = 1/rate_para)
    
    return wRatio/np.sum(wRatio);



def Inci2Epi_Para(outbreakData):
    outbreakName, IncDataPath, mean_GT,sd_GT,MaxInfctPrd = outbreakData
    IncData = pd.read_csv(IncDataPath)['IncData'].values    
    Wratios = gamma2discrete(mean_GT,sd_GT,MaxInfctPrd)
    
    
    '''
    2. Simulate the incidence data IncData
    '''    
    SimDays = len(IncData)
    tempFOI = signal.convolve(np.concatenate((np.zeros(len(Wratios)-1),IncData)),Wratios, mode='valid')     
    posterior_samples = pd.DataFrame(columns = ["outbreak","R_pop","k_pop","R_new","k_new"])    
    dta_Num = len(IncData)
    stDate = int(np.ceil(mean_GT));
    
    IncData_impt = IncData[stDate:]
    FOI_impt     = tempFOI[stDate:]
    
    
    # '''
    # 4. MCMC estimation
    # '''
    basic_model = pm.Model()
    with basic_model:
        k_para =  pm.Uniform("k_disp",1e-6,100)
        Rt_para = pm.Uniform("Rt", 0.1, 10);
        nb_r = FOI_impt*k_para;
        nb_p = k_para/(Rt_para+k_para);    
        Y_obs = pm.NegativeBinomial("Y_obs", n=nb_r, p=nb_p, observed=IncData_impt)
        trace = pm.sample(10000, return_inferencedata=False,cores=1,chains = 1);
    
    posterior_samples["k_new"] = pd.DataFrame(trace['k_disp'].squeeze().T)
    posterior_samples["R_new"] = pd.DataFrame(trace['Rt'].squeeze().T)
    
    k_hpd = az.hdi(trace["k_disp"],hdi_prob = 0.95)
    R_hpd = az.hdi(trace["Rt"],hdi_prob = 0.95)
    k_median = np.median(trace["k_disp"])
    R_median = np.median(trace["Rt"])
    
    basic_model_pop = pm.Model()
    with basic_model_pop:
        k_para =  pm.Uniform("k_disp_pop",1e-6,100)
        Rt_para = pm.Uniform("Rt_pop", 0.1, 10);    

        nb_r = k_para;
        nb_p = k_para/(Rt_para*FOI_impt+k_para)
        
        Y_obs = pm.NegativeBinomial("Y_obs_pop", n=nb_r, p=nb_p, observed=IncData_impt) 
        trace_pop = pm.sample(10000, return_inferencedata=False,cores=1,chains = 1);    
    
    posterior_samples['k_pop'] = pd.DataFrame(trace_pop['k_disp_pop'].squeeze().T)
    posterior_samples['R_pop'] = pd.DataFrame(trace_pop['Rt_pop'].squeeze().T)
    
    posterior_samples['outbreak'] = outbreakName
    
    k_hpd_pop = az.hdi(trace_pop["k_disp_pop"],hdi_prob = 0.95)
    R_hpd_pop = az.hdi(trace_pop["Rt_pop"],hdi_prob = 0.95)
    k_median_pop = np.median(trace_pop["k_disp_pop"])
    R_median_pop = np.median(trace_pop["Rt_pop"])
    return [posterior_samples,k_median_pop,k_hpd_pop,R_median_pop,R_hpd_pop,k_median,k_hpd,R_median,R_hpd];

'''

'''
IncPath = '/Users/macbjmu/Documents/research/NewIdeas/dynamic_Rt/dynaRt_code/data/'

Incfile = 'mers-sk.csv'
outbreakName = 'Mers-South Korea'
mean_GT = 12.6;
sd_GT   = 2.8;
MaxInfctPrd = int(mean_GT+3*sd_GT)+1
obkData = [outbreakName,IncPath+Incfile,mean_GT,sd_GT,MaxInfctPrd]
posterior_obk =  Inci2Epi_Para(obkData)
# the estimated k and R of Mers from literature
K_CI_Mers = [0.03,0.09]
R_CI_Mers = [0.36,1.44]

Incfile = 'CoV-HK.csv'
outbreakName = 'COVID-HongKong'
mean_GT = 5.2;
sd_GT   = 1.72;
MaxInfctPrd = int(mean_GT+3*sd_GT)+1
obkData = [outbreakName,IncPath+Incfile,mean_GT,sd_GT,MaxInfctPrd]
posterior_obk_HK =  Inci2Epi_Para(obkData)
# the estimated k and R of COVID-19, Hongkong from literature
K_CI_CoV_HK = [0.29,0.67]
R_CI_CoV_HK = [0.58,0.97]


Incfile = 'CoV-TJ.csv'
outbreakName = 'COVID-Tianjin'
# MaxInfctPrd = 14; 
mean_GT = 5.2;
sd_GT   = 1.72;
MaxInfctPrd = int(mean_GT+3*sd_GT)+1
obkData = [outbreakName,IncPath+Incfile,mean_GT,sd_GT,MaxInfctPrd]
posterior_obk_TJ =  Inci2Epi_Para(obkData)
# the estimated k and R of COVID-19, Tianjing from literature
K_CI_CoV_TJ = [0.13,0.88]
R_CI_CoV_TJ = [0.44,1.03]




# k plot
posterior_obk_all = pd.concat((posterior_obk[0],posterior_obk_HK[0],posterior_obk_TJ[0]))
plt.style.use('ggplot') 
fig,ax = plt.subplots(figsize=(16,9),dpi = 500)

fig, axes = joypy.joyplot(posterior_obk_all, ax = ax,by="outbreak", 
                          overlap = 0,column=["k_pop",'k_new'],alpha = 0.5,ylim = 'own',
                          x_range = [2.5*1e-2,6],grid = True,ylabels = False,linewidth = 1.5)

K_CIs = [K_CI_CoV_HK,K_CI_CoV_TJ,K_CI_Mers]
ar_pos = [-0.4,-0.1,-0.2,-0.2]
for i in range(len(axes)-1):
    tempCI = K_CIs[i]
    axes[i].hlines(ar_pos[i] , tempCI[0],tempCI[1],color = 'k',lw = 5)  
    axes[i].plot((tempCI[0]+tempCI[1])/2,ar_pos[i],'r+',markersize=16)
axes[-1].set_xscale('log')
axes[-2].set_xscale('log')
axes[-3].set_xscale('log')
axes[-1].set_xlabel("k")


# R plot
posterior_obk_all = pd.concat((posterior_obk[0],posterior_obk_HK[0],posterior_obk_TJ[0]))
plt.style.use('ggplot') # pretty matplotlib plots
# plt.rc('font', size=20)
fig,ax = plt.subplots(figsize=(16,9),dpi = 500)
fig, axes = joypy.joyplot(posterior_obk_all, ax = ax,by="outbreak", overlap = 0,column=["R_pop",'R_new'],alpha = 0.5,ylim = 'own',x_range = [0.3,2],grid = True,ylabels = False)
R_CIs = [R_CI_CoV_HK,R_CI_CoV_TJ,R_CI_Mers]
ar_pos = [-0.3,-0.1,-0.1]
for i in range(len(axes)-1):
    tempCI = R_CIs[i]
    axes[i].hlines(ar_pos[i] , tempCI[0],tempCI[1],color = 'k',lw = 5)   
    axes[i].plot((tempCI[0]+tempCI[1])/2,ar_pos[i],'r+',markersize=16)
axes[-1].set_xlabel("R")


