# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:00:28 2020

@author: kanishk 

The grid for evaluating the physical quantity x.
"""
import numpy as np 

class GridX:
    
    def __init__(self, minimum_x, maximum_x, no_of_nodes):
        self.minimum_x = minimum_x
        self.maximum_x = maximum_x
        self.no_of_nodes = no_of_nodes
    
    def UniformLogarithmic(self):
        x = np.logspace(np.log10(self.minimum_x),np.log10(self.maximum_x),self.no_of_nodes)
        x_node_boundaries = np.zeros(len(x)+1)
        x_node_boundaries[0] = self.minimum_x
        x_node_boundaries[-1] = self.maximum_x
        delta_x = np.zeros(len(x_node_boundaries)-1)
        
        for i in range(1,len(x_node_boundaries)-1):
            x_node_boundaries[i] = 0.5*(x[i-1] + x[i])
        for i in range(len(delta_x)):
            delta_x[i] = x_node_boundaries[i+1] - x_node_boundaries[i]
        return x,x_node_boundaries,delta_x
    
    
    
# implementation shown below
# x, x_node_boundaries,delta_x = GridX(1e-8,1,5).UniformLogarithmic()

    def Uniform(self):
        x = np.linspace(self.minimum_x,self.maximum_x,self.no_of_nodes)
        x_node_boundaries = np.zeros(len(x)+1)
        x_node_boundaries[0] = self.minimum_x
        x_node_boundaries[-1] = self.maximum_x
        delta_x = np.zeros(len(x_node_boundaries)-1)
        for i in range(1,len(x_node_boundaries)-1):
            x_node_boundaries[i] = 0.5*(x[i-1] + x[i])
        for i in range(len(delta_x)):
            delta_x[i] = x_node_boundaries[i+1] - x_node_boundaries[i]
        return x,x_node_boundaries,delta_x
        

# implementation shown below
# x, x_node_boundaries,delta_x = GridX(0,1,5).Uniform()






