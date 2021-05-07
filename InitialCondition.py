# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:39:12 2020

@author: kanishk 
The script defines initial condition classes, it specifies the initial condition
while making sure that the mass of the system is unity
"""
import numpy as np
from scipy.stats import lognorm
from scipy.optimize import curve_fit
import pandas as pd
from scipy.stats import lognorm
import scipy

class InitialConditionNumberDensity:
    # x vector or the initial x coordinate must be specified for the 
    ## realisaion of initial condition 
    ### Currently the initial conditions specified are for verification and validation
     # total mass of the particles irrespective of the initial condition
    global total_massICND
    total_massICND = 1
    def __init__(self,x, delta_x, no_of_nodes = 16):
        self.x = x
        self.delta_x = delta_x
        self.no_of_nodes = no_of_nodes
        
    def OneLargeParticle(self):
        n = np.zeros_like(self.x)
        # total mass of the system is given as 
        # Sum(ni*xi*delta_xi) = 1
        n[-1] = total_massICND/(self.delta_x[-1]*self.x[-1])
        return self.x*n
        
    def ExponentialDistribution(self):
        n = np.exp(-self.x) # defining exponential number density
        # evaluating mass of the system based on initial number density
        virtual_mass = np.sum(n*self.x*self.delta_x)
        # To make sure that the total mass of the system remain to 
        # unity we inflate the values the number density at each node by a fix 
        # constant. This however, does not change the nature of the initial distribution.
        constant = total_massICND/virtual_mass
        # finally inflating the values of number density by the constant found
        n = constant*n
        return n
        

    def FilbetAndLaurencotExponentialDistribution(self):
        n = np.exp(-self.x) # defining exponential number density
        # evaluating mass of the system based on initial number density
        M0 = np.sum(n*self.delta_x) # number of particles 
        n = np.power(M0,2)*np.exp(-M0*self.x)
        return self.x*n
    
    def FilbetAndLaurencotExponentialDistributionForProductKernel(self):
        n = np.exp(-self.x) # defining exponential number density
        # evaluating mass of the system based on initial number density
        n = np.exp(-self.x) / self.x
        return self.x*n
    
    def Solsvik_LogNormalDistribution(self):
        # This initial condition has been obtianed from a paper (Solsvik.et.al.2015)
        # just for the purpose of Solver vaidation,
        ## Loading dataset.
        n_solvik = pd.read_csv('Validation/Solsvik_Init_Breakage_Dominated_Cases.csv'
                               , names = ['x','g'])

        def lognormal(x,A,s,m):
            return (A/(s*np.sqrt(2*np.pi)*x))*np.exp(-np.square(np.log(x)-m)/(2*s*s))
        
        popt,pcov = curve_fit(lognormal, n_solvik['x'], n_solvik['g'])
        kx,ky = 1e12,1e7
        # the following function return g and not x take care
        return lognormal(self.x, popt[0],popt[1]*4,popt[2]*0.1)*ky        
        
    def InitialConditionBasedMeanandStd(self,mu,sigma):
        # mu, sigma = 50, 14.6 #in micrometers
        lnsigma = np.sqrt(np.log( np.square(sigma) / np.square(mu) + 1) );
        lnmu = np.log(mu/np.exp(0.5*np.square(lnsigma)));
        ## random will give me approximate droplet sizes
        droplet_counts = np.random.lognormal(lnmu,lnsigma,3000)*1e-6 # as my droplet size is in microns
        # volume_counts = (4/3)*np.pi*np.power(droplet_counts,3)
        # coverting to volume of particles
        
        shape,loc,scale = lognorm.fit(droplet_counts)
        d = np.logspace(np.log10(droplet_counts.min()),np.log10(droplet_counts.max()),self.no_of_nodes)
        pdf = lognorm.pdf(d, shape, loc, scale)
        return pdf
    
    def NormDistributionBasedMeanandStd(self,mu,sigma):
        mu = mu*1e-6 # Mean of sample !!! Make sure your data is positive for the lognormal example 
        sigma = sigma*1e-6 # Standard deviation of sample
        N = 6000 # Number of samples
        
        norm_dist = scipy.stats.norm(loc=mu, scale=sigma) # Create Random Process
        x = norm_dist.rvs(size=N) # Generate samples
        
        # Fit normal
        fitting_params = scipy.stats.norm.fit(x)
        norm_dist_fitted = scipy.stats.norm(*fitting_params)
        return norm_dist.pdf(self.x)

    
    
        