# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 19:51:42 2020

@author: kanishk 

The script evaluates known analytical solution for PBE.


"""
import numpy as np
from Grid_X import GridX
from TemporalGrid import TemporalGrid
from scipy.special import iv



class AnalyticalSolution:
    
    def __init__(self, x, final_time, delta_x = None):
        self.x = x
        self.time = final_time
        self.delta_x = delta_x
        
    def BinaryBreakageLinearSelection(self):
        num_density = np.zeros(len(self.x))
        num_density[:-2] = np.exp(-self.time*self.x[:-2])*(2*self.time + self.time*self.time*(self.x[-1]-self.x[:-2]))
        num_density[-1] = np.exp(-self.time*self.x[-1])
        return num_density
        
            
    def BinaryBreakageSquareSelection(self):
        t = self.time
        x = self.x
        num_density = np.zeros(len(x))
        num_density[:-2] = np.exp(-t*x[:-2]*x[:-2])*(2*t*x[-1])
        num_density[-1] = np.exp(-t*x[-1]*x[-1])
        return num_density
    
    def BinaryBreakageLinearSelectionExpInitialCondition(self):
        t = self.time
        x = self.x
        num_density = np.exp(-x*(1+t))*(1+t)*(1+t)
        return num_density
    
    def BinaryBreakageSquareSelectionExpInitialCondition(self):
        t = self.time
        x = self.x
        num_density = np.exp(-(t*x*x) - x)*(1+ (2*t*(1+x)))
        return num_density
        
    def ConstantUnityCoagulationNormExpInitialCondition(self):
        n = np.exp(-self.x)
        N0 = np.sum(n*self.delta_x)
        N0_t = (2*N0)/(2 + N0*self.time)
        num_den_ana = np.power(N0_t,2) * np.exp(-N0_t*self.x)
        return num_den_ana
    
    
    def ProductKernelCoagulationNormExpInitialCondition(self):
        if self.time <= 1:
            T = 1+self.time
        else :
            T = 2*np.sqrt(self.time)
        if self.time == 0:
            num_den_ana = np.exp(-self.x)/self.x
        else:
            num_den_ana = (np.exp(-T*self.x) * 
                       ((iv(1,2*self.x*np.power(self.time,0.5)))/
                        (np.power(self.x,2)*np.power(self.time,0.5))))
        return num_den_ana
    
