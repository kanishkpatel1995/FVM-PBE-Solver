# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 20:04:00 2020

@author: kanishk 

## Euler Explicit Solver
"""
import numpy as np

def EulerExplicitSolver(g, t, delta_t,x, A):
    g_euler_eplicit = np.zeros([len(t),len(g)])
    g_euler_eplicit[0,:] = g
    for time in range(len(t)-1):
        g_euler_eplicit[time+1,:] = g_euler_eplicit[time,:] + (t[time+1] - t[time])*np.matmul(A, g_euler_eplicit[time,:])
    
    
    num_density_euler_explicit = np.zeros_like(g_euler_eplicit)
    for i in range(len(g_euler_eplicit[:,1])):
        num_density_euler_explicit[i,:] =  g_euler_eplicit[i,:]/(x)
    
    return num_density_euler_explicit, g_euler_eplicit
    