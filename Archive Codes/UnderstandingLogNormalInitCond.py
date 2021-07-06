# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 10:39:01 2021

@author: kanishk
"""
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.optimize import curve_fit
import pandas as pd
import scipy

plt.figure(figsize = (10,10))
for mu in [50]:
    # mu = 60
    sigma = 15
    normal_std = np.sqrt(np.log(1 + (sigma/mu)**2))
    normal_mean = np.log(mu) - normal_std**2 / 2
    ## random will give me approximate droplet sizes
    droplet_counts = np.random.lognormal(normal_mean,normal_std,3000)*1e-6 # as my droplet size is in microns
    # volume_counts = (4/3)*np.pi*np.power(droplet_counts,3)
    # coverting to volume of particles
    
    shape,loc,scale = lognorm.fit(droplet_counts)
    d = np.logspace(np.log10(droplet_counts.min()),np.log10(droplet_counts.max()),64)
    c = lognorm(shape, loc, scale)
    # plt.plot(d, pdf)
    
    x = np.logspace(-8,8,64)
    plt.semilogx(x, c.pdf(x),'-*', label = 'mean = {}'.format(mu))
    
plt.legend()


mu,sigma = 15,15
# mean, sigma = 50/mean, 15/mean

normal_std = np.sqrt(np.log(1 + (sigma/mu)**2))
normal_mean = np.log(mu) - normal_std**2 / 2

dsd  = np.random.lognormal(normal_mean,normal_std, 30000)*1e-6
print(np.mean(dsd*1e6), np.std(dsd*1e6))
shape,loc,scale = lognorm.fit(dsd)
# d = np.logspace(np.log10(droplet_counts.min()),np.log10(droplet_counts.max()),64)
curve = lognorm(shape, loc, scale)
# plt.plot(d, pdf)
x = np.logspace(-8,1,64)
plt.semilogx(x, curve.pdf(x),'-*', label = 'mean = {}'.format(mu))
plt.hist(dsd, bins = 100, density = True)
plt.xscale('log')
# plt.xlim(1e-8,1)


plt.figure()
x = np.random.random_sample(3000)
mu,sigma = 50,15
normdist = mu + x*sigma
plt.hist(normdist, bins = 100)
lndist = np.exp(mu) + np.exp(x)*sigma
plt.plot(x, lndist)

######################

mu = 50*1e-6 # Mean of sample !!! Make sure your data is positive for the lognormal example 
sigma = 15*1e-6 # Standard deviation of sample
N = 2000 # Number of samples

norm_dist = scipy.stats.norm(loc=mu, scale=sigma) # Create Random Process
x = norm_dist.rvs(size=N)
plt.hist(x,bins = 100)
plt.xscale('log')
x_exp = np.exp(x)
mu_exp = np.exp(mu)
sigma_exp = np.exp(sigma)

fitting_params_lognormal = scipy.stats.lognorm.fit(x_exp, floc=0, scale=mu_exp)
lognorm_dist_fitted = scipy.stats.lognorm(*fitting_params_lognormal)
t = np.linspace(np.min(x_exp), np.max(x_exp), 100)

# Here is the magic I was looking for a long long time
lognorm_dist = scipy.stats.lognorm(s=sigma, loc=0, scale=np.exp(mu))
# Plot lognormals
f, ax = plt.subplots(1, sharex='col', figsize=(10, 5))
ax.plot(t, lognorm_dist_fitted.pdf(t), lw=2, color='r',
        label='Fitted Model X~LogNorm(mu={}, sigma={})'.format(lognorm_dist_fitted.mean(), lognorm_dist_fitted.std()))
ax.semilogx(t, lognorm_dist.pdf(t), lw=2, color='g', ls=':',
        label='Original Model X~LogNorm(mu={}, sigma={})'.format(lognorm_dist.mean(), lognorm_dist.std()))
ax.legend()
plt.show()


mu = 50*1e-6 # Mean of sample !!! Make sure your data is positive for the lognormal example 
sigma = 15*1e-6 # Standard deviation of sample
N = 2000 # Number of samples

norm_dist = scipy.stats.norm(loc=mu, scale=sigma) # Create Random Process
x = norm_dist.rvs(size=N) # Generate samples

# Fit normal
fitting_params = scipy.stats.norm.fit(x)
norm_dist_fitted = scipy.stats.norm(*fitting_params)
t = np.linspace(np.min(x), np.max(x), 100)
x = np.logspace(-8, 1, 100)
# Plot normals
f, ax = plt.subplots(1, sharex='col', figsize=(10, 5))
plt.hist(x,bins = 100, density = True)
ax.plot(t, norm_dist_fitted.pdf(t), lw=2, color='r',
        label='Fitted Model X~N(mu={}, sigma={})'.format(norm_dist_fitted.mean(), norm_dist_fitted.std()))
ax.semilogx(x, norm_dist.pdf(x), lw=2, color='g', ls=':',
        label='Original Model X~N(mu={}, sigma={})'.format(norm_dist.mean(), norm_dist.std()))
ax.legend()
plt.show()


