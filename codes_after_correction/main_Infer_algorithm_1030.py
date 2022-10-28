#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:22:30 2022

@author: macbjmu
"""
import numpy as np
from scipy.stats import nbinom
from scipy.stats import gamma
import pymc3 as pm
import arviz as az
from scipy import signal

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
    shape_para = (mean_GT/sd_GT)**2;
    rate_para  = shape_para/mean_GT
    
    wRatio = np.zeros(MaxInfctPrd+1)
    for tmps in range(1,len(wRatio)):
        wRatio[tmps] = gamma.cdf(tmps+0.5,shape_para,scale = 1/rate_para) - gamma.cdf(tmps-0.5,shape_para,scale = 1/rate_para)
    return wRatio/np.sum(wRatio);

def Inci2Epi_Para(outbreakData):
    '''
    

    Parameters
    ----------
    outbreakData : list,
        containt.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    IncData, mean_GT,sd_GT,MaxInfctPrd = outbreakData
    Wratios = gamma2discrete(mean_GT,sd_GT,MaxInfctPrd)   

    SimDays = len(IncData)
    tempFOI = signal.convolve(np.concatenate((np.zeros(len(Wratios)-1),IncData)),Wratios, mode='valid')     
    stDate = 1;    
    IncData_impt = IncData[stDate:]
    FOI_impt     = tempFOI[stDate:]
    
    basic_model = pm.Model()
    with basic_model:
        k_para =  pm.Uniform("k_disp",1e-6,100)
        Rt_para = pm.Uniform("Rt", 0.1, 10);
        nb_r = FOI_impt*k_para;
        nb_p = k_para/(Rt_para+k_para);    
        Y_obs = pm.NegativeBinomial("Y_obs", n=nb_r, p=nb_p, observed=IncData_impt)
        trace = pm.sample(10000, return_inferencedata=False,cores=1,chains = 1);
    
    
    k_hpd = az.hdi(trace["k_disp"],hdi_prob = 0.95)
    R_hpd = az.hdi(trace["Rt"],hdi_prob = 0.95)
    map_estimate = pm.find_MAP(model=basic_model)
    
    return [map_estimate["k_disp"],k_hpd,map_estimate["Rt"],R_hpd];

mean_GT = 12.6;
sd_GT   = 2.8;
MaxInfctPrd = int(mean_GT+3*sd_GT)+1
IncData = np.arange(1,100)
obkData = [IncData,mean_GT,sd_GT,MaxInfctPrd]
re = Inci2Epi_Para(obkData)
