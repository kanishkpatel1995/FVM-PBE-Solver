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
        
    def geomspace(self):
        x = np.geomspace(self.minimum_x,self.maximum_x,self.no_of_nodes)
        x_node_boundaries = np.zeros(len(x)+1)
        x_node_boundaries[0] = self.minimum_x
        x_node_boundaries[-1] = self.maximum_x
        delta_x = np.zeros(len(x_node_boundaries)-1)
        for i in range(1,len(x_node_boundaries)-1):
            x_node_boundaries[i] = 0.5*(x[i-1] + x[i])
        for i in range(len(delta_x)):
            delta_x[i] = x_node_boundaries[i+1] - x_node_boundaries[i]
        return x,x_node_boundaries,delta_x

    def owngrid(self, r1,r2):
        k = int(self.no_of_nodes/3)
        x = np.zeros(self.no_of_nodes)
        x[0] = self.minimum_x
        for node in range(1,k):
            x[node] = np.power(x[node -1],r1)
        for node in range(k, int(2*k)+1):
            x[node] = np.power(x[node -1],r2)
        x[2*k:] = np.linspace(x[2*k],self.maximum_x,self.no_of_nodes - 2*k)
        x_node_boundaries = np.zeros(len(x)+1)
        x_node_boundaries[0] = self.minimum_x
        x_node_boundaries[-1] = self.maximum_x
        delta_x = np.zeros(len(x_node_boundaries)-1)
        for i in range(1,len(x_node_boundaries)-1):
            x_node_boundaries[i] = 0.5*(x[i-1] + x[i])
        for i in range(len(delta_x)):
            delta_x[i] = x_node_boundaries[i+1] - x_node_boundaries[i]
        return x,x_node_boundaries,delta_x
      
    
    def owngrid2(self, r1,r2):
        k = int(self.no_of_nodes/3)
        x = np.zeros(self.no_of_nodes)
        x[0] = self.minimum_x
        for node in range(1,k):
            x[node] = (x[node -1]*r1)
        for node in range(k, int(2*k)+1):
            x[node] = np.power(x[node -1],r2)
        x[2*k:] = np.linspace(x[2*k],self.maximum_x,self.no_of_nodes - 2*k)
        x_node_boundaries = np.zeros(len(x)+1)
        x_node_boundaries[0] = self.minimum_x
        x_node_boundaries[-1] = self.maximum_x
        delta_x = np.zeros(len(x_node_boundaries)-1)
        for i in range(1,len(x_node_boundaries)-1):
            x_node_boundaries[i] = 0.5*(x[i-1] + x[i])
        for i in range(len(delta_x)):
            delta_x[i] = x_node_boundaries[i+1] - x_node_boundaries[i]
        return x,x_node_boundaries,delta_x
    
    def bigeometric(self, node_value,r):
        #r1 = number of nodes in first half
        x = np.zeros(self.no_of_nodes)
        x[0] = self.minimum_x
        for node in range(1,node_value):
            if r>= 0:
                x[node] = x[node-1] + np.power(x[node-1],r)
            elif r> 0:
                x[node] = x[node-1] - np.power(x[node-1],r)
        if node_value != self.no_of_nodes:
            x[node_value:] = np.logspace(np.log10(x[node_value-1]), np.log10(self.maximum_x), self.no_of_nodes-node_value)
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
# min_x,max_x,nodes =1e-8,1,20
# x_u, x_node_boundaries,delta_x = GridX(min_x,max_x,nodes).Uniform()
# x_l, x_node_boundaries,delta_x = GridX(min_x,max_x,nodes).UniformLogarithmic()
# x_g, x_node_boundaries,delta_x = GridX(min_x,max_x,nodes).geomspace()
# x_o, x_node_boundaries,delta_x = GridX(min_x,max_x,nodes).owngrid(0.75,0.75)
# x_gp, x_node_boundaries,delta_x = GridX(min_x,max_x,nodes).bigeometric(10,0.9)
# y = np.zeros(len(x_l))
# import matplotlib.pyplot as plt
# plt.figure()
# plt.semilogx(x_u,y,'r*-', label = 'Uniform')
# plt.plot(x_l,y+1,'b^-', label = 'log')
# plt.plot(x_g,y+2,'g<-', label = 'geom')
# plt.plot(x_o,y+3,'k<-', label = 'own')
# plt.plot(x_gp,y+4,'cv-', label = 'GP')
# plt.legend()




