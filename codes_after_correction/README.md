These files are the codes before the correction of the bug.

# Explanation about the bug in the python code

## Where is the bug
The bug lies in the first line in the function of “gamma2discrete” which calculates the probability distribution of serial interval. 
The input parameters include the mean and standard deviation of the serial interval (denoted as mean_GT and sd_GT respectively) and the maximum length of infectious period for a particular case (denoted as MaxInfctPrd). We modelled the serial interval as a gamma shaped distribution, so we firstly need to transform the mean and standard deviation of the distribution into scale and rate parameters. 

## What is the bug
The wrong code was: “shape_para = mean_GT/sd_GT**2”

The corrected code was: “shape_para = (mean_GT/sd_GT)**2”

(Please see https://en.wikipedia.org/wiki/Gamma_distribution for more details of the transformation)
