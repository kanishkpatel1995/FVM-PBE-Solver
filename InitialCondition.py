# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:39:12 2020

@author: kanishk 
The script defines initial condition classes, it specifies the initial condition
while making sure that the mass of the system is unity
"""
import numpy as np
from scipy.stats import lognorm

class InitialConditionNumberDensity:
    # x vector or the initial x coordinate must be specified for the 
    ## realisaion of initial condition 
    ### Currently the initial conditions specified are for verification and validation
     # total mass of the particles irrespective of the initial condition
    global total_massICND
    total_massICND = 1
    def __init__(self,x, delta_x):
        self.x = x
        self.delta_x = delta_x
        
    def OneLargeParticle(self):
        n = np.zeros_like(self.x)
        # total mass of the system is given as 
        # Sum(ni*xi*delta_xi) = 1
        n[-1] = total_massICND/(self.delta_x[-1]*self.x[-1])
        return n
        
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
        return n
    
    def FilbetAndLaurencotExponentialDistributionForProductKernel(self):
        n = np.exp(-self.x) # defining exponential number density
        # evaluating mass of the system based on initial number density
        n = np.exp(-self.x) / self.x
        return n
    
    def LogNormalDistribution(self):
        mean = 1.5*np.log10(self.x.min())
        # mean = np.log10(0.1)
        sigma  = 1
        x = self.x
        func = (1/(sigma*x*np.sqrt(2*np.pi)))*np.exp(-np.power((np.log(x)-mean),2)/(2*sigma*sigma))*1e6
        return func
        
        
        
        