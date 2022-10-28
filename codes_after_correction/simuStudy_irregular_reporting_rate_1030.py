#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This file tests the performance of the IIH model under the situation with irregular reporting rate.
The main function is the sngleRun_IRR_0712 which generates the incidence data with irregular reporting rate.
'''

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from scipy import special
import scipy.integrate as integrate
from scipy.stats import gamma
import pymc3 as pm
import arviz as az
from scipy import signal
import multiprocessing as multip
from scipy import stats


# calculate the discrete serial interval distribution
def gamma2discrete(mean_GT,sd_GT,MaxInfctPrd):
    
    shape_para = (mean_GT/sd_GT)**2;
    rate_para  = shape_para/mean_GT

    
    wRatio = np.zeros(MaxInfctPrd+1)
    for tmps in range(1,len(wRatio)):
        wRatio[tmps] = gamma.cdf(tmps+0.5,shape_para,scale = 1/rate_para) - gamma.cdf(tmps-0.5,shape_para,scale = 1/rate_para)
    
    return wRatio/np.sum(wRatio);

# calculate the total infectiousness based on the incidence data (IncData) and the distribution of serial interval (Wratios)
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

# generate the incidence data till the SimDays
def IncSimu_dailyNB1216(SimDays,IndexNum,mean_GT,sd_GT,MaxInfctPrd,tmpRt,tmpKt): 
    Wratios = gamma2discrete(mean_GT,sd_GT,MaxInfctPrd)
    IncData = np.zeros(SimDays);
    IncData[0] = IndexNum;
    for i in range(1,SimDays):
        # get the ttlInf, kt and Rgt
        ttl_Inf = ttlInfFun(IncData[:i],Wratios);
        ttl_Inf2=signal.convolve(np.concatenate((np.zeros(len(Wratios) - 1), IncData)), Wratios, mode='valid')[i]
        if ttl_Inf>0:
            IncData[i] = nbinom.rvs(tmpKt*ttl_Inf,tmpKt/(tmpRt+tmpKt)); # generate the incidence     
    return IncData

# calculated the MAP estimtion for R and k, and the ess for R and k respectively.
def dynmcInfer(IncData_impt,FOI_impt):    
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

# # to perform one run of simulation study and to make inference based on the last period (tmpDays) of incidence data
def sngleRun_IRR_0712(mean_GT,sd_GT,MaxInfctPrd,initialCase,tmpRt,tmpKt,tmpDays,rpt_freq_srs):    
    # SimDays --the length of simulation
    # initialCase -- the number of cases on time 0;    
    # 1) simulation the incidence data
    zero_iRR_cnt = 1
    while zero_iRR_cnt >0:
        zero_cnt = 1
        simLen = 30
        while zero_cnt >0:      
            # generate the incidence data        
            EpiSimuData  = IncSimu_dailyNB1216(simLen,initialCase,mean_GT,sd_GT,MaxInfctPrd,tmpRt,tmpKt)        
            # check if there is zero element in FOI_impt
            Wratios = gamma2discrete(mean_GT,sd_GT,MaxInfctPrd)
            tempFOI = signal.convolve(np.concatenate((np.zeros(len(Wratios)-1),EpiSimuData)),Wratios, mode='valid')    
            IncData_impt = EpiSimuData[-tmpDays:]
            FOI_impt     = tempFOI[-tmpDays:]
            zero_cnt = np.where(abs(FOI_impt) < 0.00001)[0].size
       
    
        # 2) perfom irregular reporting and generate the synthetic data iRR_Inc
        st_Inf = simLen - tmpDays; # begin to transform the data 
        # rpt_freq_srs = [1,3]
        iRR_Inc = EpiSimuData.copy()
        rpt_frq_ID = 0;
        dly_cnt = 0;
        acc_Inc = 0;
        rpt_frq = rpt_freq_srs[rpt_frq_ID]
        for i in range(st_Inf,len(EpiSimuData)):
            dly_cnt += 1;
            acc_Inc += EpiSimuData[i]
            if dly_cnt == rpt_frq: # it's the time to report data
                iRR_Inc[i - rpt_frq+1 :i +1] = np.floor(acc_Inc/rpt_frq)
                iRR_Inc[i] += acc_Inc- np.floor(acc_Inc/rpt_frq)*rpt_frq
                # update the situation
                acc_Inc = 0; # empty the accumulation
                dly_cnt = 0;
                rpt_frq_ID += 1;  # shift to the next delay
                rpt_frq = rpt_freq_srs[rpt_frq_ID%len(rpt_freq_srs)];
    
        # calculated the total infectiousness based on the synthetic data iRR_Inc
        iRR_FOI = signal.convolve(np.concatenate((np.zeros(len(Wratios) - 1), iRR_Inc)), Wratios, mode='valid');
        
        # 3) select the synthetic data/FOI at the reporting date for infection
        dly_cnt = 0;
        iRR_Inc_Input = []
        iRR_FOI_Input = []
        rpt_frq_ID = 0;
        rpt_frq = rpt_freq_srs[rpt_frq_ID]
        for i in range(st_Inf,len(EpiSimuData)):
            dly_cnt += 1;
            if dly_cnt == rpt_frq:
                iRR_Inc_Input   += [iRR_Inc[i]];
                iRR_FOI_Input   += [iRR_FOI[i]];
                # update
                dly_cnt = 0;
                rpt_frq_ID += 1;  # shift to the next delay
                rpt_frq = rpt_freq_srs[rpt_frq_ID % len(rpt_freq_srs)];    
        zero_iRR_cnt = np.where(abs(np.array(iRR_FOI_Input)) < 0.00001)[0].size
    
    # 4) perform inference
    return dynmcInfer(iRR_Inc_Input,iRR_FOI_Input)

def mltRun(mean_GT,sd_GT,MaxInfctPrd,initialCase,tmpRt,tmpKt,tmpDays,simN,rpt_freq_srs):
    # simN-- number of simulation
    # 1) perform simulation and inference
    # return [Rt_map,Rt_mean,R_hpd[0],R_hpd[1],kt_map,k_mean,k_median,k_hpd[0],k_hpd[1]];
    bias_R = [];
    bias_k = [];
    cvrg_re = 0;
    cprb_re = 0
    Rt_mean = 0;
    kt_mean = 0;
    kt_ess = 0;
    Rt_ess = 0;
    for i in range(simN):
        tmpInfer = sngleRun_IRR_0712(mean_GT,sd_GT,MaxInfctPrd,initialCase,tmpRt,tmpKt,tmpDays,rpt_freq_srs)
        temp_bias_R,temp_bias_k,temp_cvrg,tmp_cprb = Esti_Evl(tmpInfer,tmpRt,tmpKt);
        bias_R += [temp_bias_R];
        bias_k += [temp_bias_k];
        cvrg_re += temp_cvrg
        cprb_re += tmp_cprb
        kt_ess += tmpInfer[6];
        Rt_ess += tmpInfer[7];
    MAD_R = np.median(np.abs(np.array(bias_R)));
    MAD_k = np.median(np.abs(np.array(bias_k)));
    MBS_R = np.median(np.array(bias_R));
    MBS_k = np.median(np.array(bias_k)); 
    cvrg_re = cvrg_re/simN
    cprb_re = cprb_re/simN
    kt_ess = kt_ess/simN
    Rt_ess = Rt_ess/simN
    return [MAD_R,MAD_k,cvrg_re[0],cvrg_re[1],cprb_re[0],cprb_re[1],kt_ess,Rt_ess,MBS_R,MBS_k]


def Esti_Evl(tmpInfer,tmpRt,tmpKt):
    # tvInference starts from time = 1
    # calculate the coverage and RMSE based on the MAP and the related HPD 
    evl_cvrg = np.zeros(2)# R_hpd_cvrg,k_hpd_cvrg
    evl_cprb = np.zeros(2)# R_hpd_cvrg,k_hpd_cvrg
    
    evl_bias_R = (tmpInfer[0] - tmpRt);
    evl_bias_k = (tmpInfer[1] - tmpKt);
    evl_cprb[0] = (tmpInfer[2]<=tmpRt)*(tmpInfer[3]>=tmpRt)
    evl_cprb[1] = (tmpInfer[1]<1)*(tmpKt<1) + (tmpInfer[1]>1)*(tmpKt>1) # kt and K at the same side of 1
    
    evl_cvrg[0] = (tmpInfer[2]<=tmpRt)*(tmpInfer[3]>=tmpRt)
    evl_cvrg[1] = (tmpInfer[4]<=tmpKt)*(tmpInfer[5]>=tmpKt)
    # evl_cvrg[1] = (tmpInfer[4]<=tmpKt)*(tmpInfer[5]>=tmpKt)
    return [evl_bias_R,evl_bias_k,evl_cvrg,evl_cprb]


# perform simulation study under different scenario: R, k, and the length of data used in the inference. 
# for each scenario, repeating the simulation for n (simuRun) times
    # single run
mean_GT = 5.2;
sd_GT   = 1.72;
MaxInfctPrd = int(mean_GT+3*sd_GT)+1; # maximum length of infectious period in day
Rt_seq = [1.1,1.3,1.5];
kt_seq = [0.2,0.5,2,5];
initialCase = 10;
rpt_freq_srs = [1,3]

simuRun = 100;
simDays_seq = [7,14,24];
tmpLen = simDays_seq[2]


outPth = '/Users/macbjmu/Documents/research/NewIdeas/dynamic_Rt/dynaRt_code/simulation_code/simu_result_0916/'
outFile = outPth+'IRR_MAD_simuRun='+str(simuRun)+'_tmpLen='+str(tmpLen)+'_0930.csv';

# # run simulation
re = []
for tmpR in Rt_seq:
    for tmpK in kt_seq:
        re = [mltRun(mean_GT,sd_GT,MaxInfctPrd,initialCase,tmpR,tmpK,tmpLen,simuRun,rpt_freq_srs)] 
        outcomes = pd.DataFrame(re)
        outcomes.to_csv(outFile,mode="a", header=False)

