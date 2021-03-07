# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:37:33 2020

@author: kanishk 
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:26:28 2020

@author: kanishk 
"""
""" Post processing CLD Data from UofS"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.optimize import curve_fit

# Each count corresponds to an occurence of a certain length lying with the interval.
#Each interval is given by logspace distribution of bins, given below
counts_exp =np.array([ 5.9188  ,  6.3421  ,  8.6642  , 12.713   , 13.623   , 14.597   ,
       16.343   , 20.495   , 21.961   , 23.532   , 29.041   , 31.259   ,
       32.442   , 32.513   , 34.608   , 34.805   , 36.362   , 33.957   ,
       33.599   , 32.569   , 32.64    , 30.98    , 28.098   , 24.374   ,
       19.099   , 16.84    , 13.82    , 13.278   , 13.621   , 12.877   ,
        9.9978  ,  6.6081  ,  7.6032  ,  8.0517  ,  4.5022  ,  5.0245  ,
        4.3898  ,  5.4387  ,  4.8596  ,  4.0522  ,  3.8057  ,  4.5899  ,
        4.8472  ,  3.6182  ,  2.525   ,  4.129   ,  3.3418  ,  3.7332  ,
        2.3115  ,  3.1803  ,  2.5602  ,  2.6225  ,  2.0946  ,  2.5827  ,
        1.9317  ,  2.4979  ,  1.4364  ,  1.2115  ,  1.5103  ,  0.49907 ,
        0.35787 ,  0.35787 ,  0.71574 ,  0.17894 ,  0.35787 ,  0.      ,
        0.17894 ,  0.      ,  0.087635,  0.091301,  0.      ,  0.      ,
        0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,
        0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,
        0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,
        0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,
        0.      ,  0.      ,  0.      ,  0.      ])

bins = np.logspace(0,3, 101) # each bin represent the count given above repectively.
## FOr example there were about 6 observations for an entity having length between 
## 1 and 1.07152 (first bin in bins)


### Plotting histogram of the droplets.
# restoring data to obtain histogram, frequency data is converted to sample data 
restored = [[d]*int(counts_exp[n]) for n,d in enumerate((bins[1:]+bins[:-1])/2)]
###Midpoint of bins
d = np.array([0.5*(bins[i+1] + bins[i]) for i in range(len(bins)-1)])
restored = [item for sublist in restored for item in sublist]
counts, bins, ignored = plt.hist(restored,bins = bins, density=True,color ='bisque', align='mid')
plt.xscale('log')

## Ealuating mean from restored data
# Distrbution 1
act_mean = np.mean(np.log(restored))
act_std = np.std(np.log(restored))

##evaluating from the regular experimental frequency distribution with no normalisation
#Distirbution 2
mu = np.sum(np.log(d)*counts_exp)/np.sum(counts_exp)
sigma = np.sqrt(np.sum(counts_exp*np.power(np.log(d) - mu,2))/np.sum(counts_exp))

# evaluating mean and standard deviation for the frequency distribution, i.e normalised
## Histogram
# Distrbution 3
mu_from_hist = np.sum(np.log(d)*counts)/np.sum(counts)
sigma_from_hist = np.sqrt(np.sum(counts*np.power(np.log(d) - mu_from_hist,2)
                                 )/np.sum(counts))



### Evaluating a lognormal distribution based on the mean and sigma.
x = np.logspace(0,3,1000)
def LogNormalpdf(x,mean,sigma):
    func = (1/(sigma*np.sqrt(2*np.pi)*x))*np.exp(-np.power((np.log(x)-mean),2)/(2*sigma*sigma))
    return func

#The below two must be same slight variation acceptable given log values have been introduced
plt.semilogx(x, LogNormalpdf(x,act_mean,act_std), linewidth=2, color='k', 
             label = "Distribution 1")

plt.semilogx(x, LogNormalpdf(x,mu,sigma), linewidth=2, color='g', label = "Distribution 2")

plt.semilogx(x, LogNormalpdf(x,mu_from_hist,sigma_from_hist), linewidth=2, color='r',
             label = "Distribution 3")



### Now using lognormal fit to fit the restotred data. 
shape,loc,scale = lognorm.fit(restored)

x = np.logspace(0, 5, 200)
pdf = lognorm.pdf(x, shape, loc, scale)

## I recieve a lognormal distribution with different mean and standard deviation 
plt.plot(x, pdf, 'c', label = "Scipy Lognormal Fit")
plt.xscale('log')

mu_from_the_scipy_pdf = np.sum(np.log(x)*pdf)/np.sum(pdf)
sigma_from_scipy_pdf = np.sqrt(np.sum(pdf*np.power(np.log(x) - mu_from_the_scipy_pdf,2)
                                 )/np.sum(pdf)) 
plt.plot(x,LogNormalpdf(x, mu_from_the_scipy_pdf, sigma_from_scipy_pdf), 'b', label = 'dist from scipy mean and std')
plt.legend(frameon = False, fontsize=20)

################ Scipy pdf way 
def LogNormalScipy(x,loc,shape,scale):
    func = (1/(shape*np.sqrt(2*np.pi)*(x-loc)))*np.exp(-np.power(np.log((x-loc)/scale),2)/(2*shape*shape))
    return func

plt.plot(x,LogNormalScipy(x,loc,shape,scale), '--k', label = 'my version of scipy')
plt.legend(frameon = False, fontsize=20)

def LogNormalpdf(x,a,mean,sigma):
    func = (a/(sigma*np.sqrt(2*np.pi)*x))*np.exp(-np.power((np.log(x)-mean),2)/(2*sigma*sigma))
    return func

# y1 = counts_exp/counts_exp.max()
# np.trapz(y1,d)
popt, pcov = curve_fit(LogNormalpdf, d, counts_exp)
plt.plot(d,counts_exp,':k')
plt.semilogx(d, LogNormalpdf(d,*popt))

mean_randon_data = np.mean(restored)
std_random_data=np.std(restored)
mean = np.sum(LogNormalpdf(d,*popt)*d)/np.sum(LogNormalpdf(d,*popt))
### Conversion of standard normal mean and std to lognormal mean and std.
mu_formula = (2*np.log(mean_randon_data) - 0.5*np.log(std_random_data*std_random_data 
                                                     +mean_randon_data*mean_randon_data))
std_formula = np.sqrt(-2*np.log(mean_randon_data) + np.log(std_random_data*std_random_data 
                                                     +mean_randon_data*mean_randon_data))

mean = np.sum(d*counts_exp)/np.sum(counts_exp)



