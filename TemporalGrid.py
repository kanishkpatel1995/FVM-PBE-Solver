# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:20:39 2020

@author: kanishk 

The module generates a temporal grid
"""
import numpy as np 

class TemporalGrid:
    
    def __init__(self,min_time,max_time, delta_t):
        self.min_time = min_time
        self.max_time = max_time
        self.delta_t = delta_t
        
    def Uniform(self):
        divisons = int((self.max_time - self.min_time)/self.delta_t)
        return np.linspace(self.min_time, self.max_time, divisons)
        
        
# time = TemporalGrid(1,10,1).Uniform()
    