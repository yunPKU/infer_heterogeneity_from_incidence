# Introduction
This repository contains an implementation of the instant-individual heterogeneity model to estimate real-time transmission heterogeneity from incidence data. 
The input is a incidence time series, the parameters (i.e., mean and standard deviation) of serial interval (assuming a gamma distribution) . The incidence series should be generated regularly, such as in days or in weeks. The main output includes the estimation of dispersion number ($k$) and reproduction number ($R$).  

# Usage
The instant-individual heterogeneity model is implemented with the file “main_Infer_algorithm.py”. There are also several examples (such as Figure_2_overall_Estimation.py) used to generate the figures in the manuscript. 

The meaning of the used parameters in “main_Infer_algorithm.py” are as follows:
parameter	|  meaning	
--------- |---------
mean_GT	| the mean of generation time 	
sd_GT	| the standard deviation of generation time
MaxInfctPrd | the maximum length of generation time 
IncData | a numpy array contains the incidence time series
Output | return the MAP and the HPD of the dispersion parameter (k) and the reproduction number (R)
