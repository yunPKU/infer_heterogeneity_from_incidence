#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:04:04 2021

@author: macbjmu
"""
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from scipy import special
import scipy.integrate as integrate
from scipy.stats import gamma
import pymc3 as pm
import arviz as az
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
    # shape_para = mean_GT/sd_GT**2;   The wrong code 
    shape_para = (mean_GT/sd_GT)**2;
    rate_para  = shape_para/mean_GT
    
    wRatio = np.zeros(MaxInfctPrd+1)
    for tmps in range(1,len(wRatio)):
        wRatio[tmps] = gamma.cdf(tmps+0.5,shape_para,scale = 1/rate_para) - gamma.cdf(tmps-0.5,shape_para,scale = 1/rate_para)
    
    return wRatio/np.sum(wRatio);

def ttlInfFun(IncData,Wratios):
    # return the total infectiousness at time t
    # truncate or expand the IncData 
    wm = len(Wratios)
    Im = len(IncData)
    if Im <= wm-1:
        FOI = signal.convolve(IncData,Wratios[1:Im+1], mode='valid')
    elif Im > wm-1:
        FOI = signal.convolve(IncData[Im-wm+1:],Wratios[1:], mode='valid')
    return FOI[0]   


# generate the incidence data according to the proposed model
def IncSimu_dailyNB1216(SimDays,IndexNum,mean_GT,sd_GT,MaxInfctPrd,tmpRt,tmpKt): 
    Wratios = gamma2discrete(mean_GT,sd_GT,MaxInfctPrd)
    IncData = np.zeros(SimDays);
    IncData[0] = IndexNum;
    for i in range(1,SimDays):
        # get the ttlInf, kt and Rgt
        ttl_Inf = ttlInfFun(IncData[:i],Wratios);
        if ttl_Inf>0:
            IncData[i] = nbinom.rvs(tmpKt*ttl_Inf,tmpKt/(tmpRt+tmpKt)); # generate the incidence     
    return IncData


# perform single run of simulation and analysis
def sngleRun0707(mean_GT,sd_GT,MaxInfctPrd,initialCase,tmpRt,tmpKt,tmpDays):    
    # SimDays --the length of simulation
    # initialCase -- the number of cases on time 0;
    # 1) simulation the incidence data
    zero_cnt = 1
    while zero_cnt >0:      
        # generate the incidence data
        simLen = 24
        EpiSimuData  = IncSimu_dailyNB1216(simLen,initialCase,mean_GT,sd_GT,MaxInfctPrd,tmpRt,tmpKt)
        
        # imput for inference
        Wratios = gamma2discrete(mean_GT,sd_GT,MaxInfctPrd)
        tempFOI = signal.convolve(np.concatenate((np.zeros(len(Wratios)-1),EpiSimuData)),Wratios, mode='valid')    
        IncData_impt = EpiSimuData[-tmpDays:]
        FOI_impt     = tempFOI[-tmpDays:]
        zero_cnt = np.where(abs(FOI_impt) < 0.00001)[0].size
    
    # performe inference
    basic_model = pm.Model()
    chain_len = 10000
    brn_st = int(chain_len*0.1)
    with basic_model:
        k_para =  pm.Uniform("k_disp",0,100)
        Rt_para = pm.Uniform("Rt", 0.1, 10);
        nb_r = FOI_impt*k_para;
        nb_p = k_para/(Rt_para+k_para);    
        Y_obs = pm.NegativeBinomial("Y_obs", n=nb_r, p=nb_p, observed=IncData_impt)
        trace = pm.sample(chain_len, return_inferencedata=False,chains=1,cores=1);
    
    map_estimate = pm.find_MAP(model=basic_model)
    k_hpd = az.hdi(trace["k_disp"][brn_st:],hdi_prob = 0.95)
    R_hpd = az.hdi(trace["Rt"][brn_st:],hdi_prob = 0.95) 
    
    kt_ess = az.ess(trace["k_disp"][brn_st:])
    Rt_ess = az.ess(trace["Rt"][brn_st:])   
    del trace
    # pm.memoize.clear_cache()
    return [map_estimate['Rt'],map_estimate['k_disp'],R_hpd[0],R_hpd[1],k_hpd[0],k_hpd[1],kt_ess,Rt_ess];


# perform mulple simulation runs
def mltRun(mean_GT,sd_GT,MaxInfctPrd,initialCase,tmpRt,tmpKt,tmpDays,simN):
    # simN-- number of simulation
    # 1) perform simulation and inference
    # return [Rt_map,Rt_mean,R_hpd[0],R_hpd[1],kt_map,k_mean,k_median,k_hpd[0],k_hpd[1]];
    ase_R = [];
    ase_k = [];
    cvrg_re = 0;
    cprb_re = 0
    Rt_mean = 0;
    kt_mean = 0;
    kt_ess = 0;
    Rt_ess = 0;
    for i in range(simN):
        tmpInfer = sngleRun0707(mean_GT,sd_GT,MaxInfctPrd,initialCase,tmpRt,tmpKt,tmpDays)
        temp_sse_R,temp_sse_k,temp_cvrg,tmp_cprb = Esti_Evl(tmpInfer,tmpRt,tmpKt);
        ase_R += [temp_sse_R];
        ase_k += [temp_sse_k];
        cvrg_re += temp_cvrg
        cprb_re += tmp_cprb
        kt_ess += tmpInfer[6];
        Rt_ess += tmpInfer[7];
    MAD_R = np.median(np.array(ase_R))**0.5;
    MAD_k = np.median(np.array(ase_k))**0.5
    cvrg_re = cvrg_re/simN
    cprb_re = cprb_re/simN
    kt_ess = kt_ess/simN
    Rt_ess = Rt_ess/simN
    # del ase_R;
    # del ase_k;
    return [MAD_R,MAD_k,cvrg_re[0],cvrg_re[1],cprb_re[0],cprb_re[1],kt_ess,Rt_ess]

# evaluation of the inference results
def Esti_Evl(tmpInfer,tmpRt,tmpKt):
    # tvInference starts from time = 1
    # calculate the coverage and RMSE based on the MAP and the related HPD 
    evl_sse = np.zeros(2) # Rt_map_sse,Rt_mean_sse,,kt_map_sse,k_mean_sse,k_median_sse, 
    evl_cvrg = np.zeros(2)# R_hpd_cvrg,k_hpd_cvrg
    evl_cprb = np.zeros(2)# R_hpd_cvrg,k_hpd_cvrg
    
    evl_sse_R = (tmpInfer[0] - tmpRt)**2;
    evl_sse_k = (tmpInfer[1] - tmpKt)**2;
    evl_cprb[0] = (tmpInfer[2]<=tmpRt)*(tmpInfer[3]>=tmpRt)
    evl_cprb[1] = (tmpInfer[1]<1)*(tmpKt<1) + (tmpInfer[1]>1)*(tmpKt>1) # kt and K at the same side of 1
    
    evl_cvrg[0] = (tmpInfer[2]<=tmpRt)*(tmpInfer[3]>=tmpRt)
    evl_cvrg[1] = (tmpInfer[4]<=tmpKt)*(tmpInfer[5]>=tmpKt)
    # evl_cvrg[1] = (tmpInfer[4]<=tmpKt)*(tmpInfer[5]>=tmpKt)
    return [evl_sse_R,evl_sse_k,evl_cvrg,evl_cprb]



# single run
mean_GT = 5.2;
sd_GT   = 1.72;
MaxInfctPrd = int(mean_GT+3*sd_GT)+1; # maximum length of infectious period in day
Rt_seq = [1.1,1.3,1.5];
kt_seq = [0.2,0.5,2,5];
simDays_seq = [7,14,21];
tmpLEN = simDays_seq[2]
simuRun = 2;
initialCase = 10;
opt_cnt = 0
# run simulation
for tmpR in Rt_seq:
    for tmpK in kt_seq:
        re = [mltRun(mean_GT,sd_GT,MaxInfctPrd,initialCase,tmpR,tmpK,tmpLEN,simuRun)] 
        outcomes = pd.DataFrame(re) if opt_cnt == 0 else pd.concat([outcomes,pd.DataFrame(re)])
        opt_cnt += 1;





# '''
#  transform the result into four np
# '''
RMAD_R = np.zeros((len(Rt_seq),len(kt_seq)))
Cvrg_R = np.zeros((len(Rt_seq),len(kt_seq)))
RMAD_k = np.zeros((len(Rt_seq),len(kt_seq)))
Cvrg_k = np.zeros((len(Rt_seq),len(kt_seq)))
Cprb_k = np.zeros((len(Rt_seq),len(kt_seq)))
cnt_id = 0
for R_id in range(len(Rt_seq)):
    for K_id in range(len(kt_seq)):
        tmpR = Rt_seq[R_id]
        tmpK = kt_seq[K_id]
        RMAD_R[R_id,K_id] = outcomes.iat[cnt_id,1]/tmpR;
        RMAD_k[R_id,K_id] = outcomes.iat[cnt_id,2]/tmpK;
        Cvrg_R[R_id,K_id] = outcomes.iat[cnt_id,3];
        Cvrg_k[R_id,K_id] = outcomes.iat[cnt_id,4];
        Cprb_k[R_id,K_id] = outcomes.iat[cnt_id,6];  


'''
regular plot
# '''
lw = 1.5

plt.style.use('ggplot') # pretty matplotlib plots
f, axes = plt.subplots(5, 1, sharex='col',dpi = 500)
axes[0].plot(kt_seq,RMAD_k[0,:],kt_seq,RMAD_k[1,:],kt_seq,RMAD_k[2,:],linewidth = lw)
# axes[0].set_yscale('log',subs = [0.5,1,10])
# axes[0].set_yscale('log',subs = [0.5,1,5,20])
axes[0].set_ylim([-2,20.5])
axes[0].set_yticks([0.5,10,20])

axes[1].plot(kt_seq,Cvrg_k[0,:],kt_seq,Cvrg_k[1,:],kt_seq,Cvrg_k[2,:])
axes[1].set_ylim([0.4,1.05])
axes[1].set_yticks([0.5,0.75,1])


axes[2].plot(kt_seq,Cprb_k[0,:],kt_seq,Cprb_k[1,:],kt_seq,Cprb_k[2,:])
axes[2].set_ylim([0.5,1.05])
axes[2].set_yticks([0.5,0.75,1])


axes[3].plot(kt_seq,RMAD_R[0,:],kt_seq,RMAD_R[1,:],kt_seq,RMAD_R[2,:])
axes[3].set_ylim([0.02,0.4])
axes[3].set_yticks([0.05,0.2,0.35])

axes[4].plot(kt_seq,Cvrg_R[0,:],kt_seq,Cvrg_R[1,:],kt_seq,Cvrg_R[2,:])
axes[4].set_ylim(0.75,0.95)
axes[4].set_yticks([0.8,0.9,1])

axes[4].set_xscale('log',subs = [0.2,0.5,1,2,5])

        



