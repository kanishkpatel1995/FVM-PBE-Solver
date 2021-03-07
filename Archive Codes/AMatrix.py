# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:40:14 2020

@author: kanishk
The file generates he matrix representing system of equations. 
The system of equations represent solution at each node x_{i} at a given time
node t_{i}. Note that for the purpose of verification the  breakge kernals are 
time independent and hence the matrix has to be evaluated only once at initial 
time t = 0.
"""
import numpy as np
from BreakageFunction import BreakageFunction
from SelectionFunction import SelectionFunction


class APureBreakage:
    
    def __init__(self, x, x_node_boundaries, delta_x, type_s):
        self.x = x
        self.x_node_boundaries = x_node_boundaries
        self.delta_x = delta_x
        self.type_s = type_s
    
    def derec(self,k,i):
        d_k_i = 0
        inner_integral = 0
        if i>=0:
            for j in range(i+1): 
                inner_integral = inner_integral + self.x[j]*BreakageFunction.BinaryBreakge(self.x[k], self.x[j])*self.delta_x[j]
        else :
            inner_integral = 0
        d_k_i = SelectionFunction(self.type_s,self.x[k])*(1/self.x[k])*self.delta_x[k]*inner_integral
        return d_k_i

    def A(self):
        A = np.zeros((len(self.x),len(self.x)))
        for i in range(len(self.x)):
            print(i) 
            A[i,i] = -(1/self.delta_x[i])*self.derec(i,i-1) # diagonal elements
            for k in range(i+1,len(self.x)):
                A[i,k] = -(1/self.delta_x[i]) * (self.derec( k, i-1) - 
                                          self.derec( k, i))
        return A
        
        
        
        
        
        
        