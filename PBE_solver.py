# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:58:39 2020

@author: kanishk

"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import matplotlib.pyplot as plt

""" Creating Spatial grid """

x_min = 1e-8 #minimum particle volume
x_max = 1   # maximum particle volume  
nodes = 31
x = np.logspace(np.log10(x_min),np.log10(x_max),nodes)
# x = np.linspace(x_min,x_max,nodes)
x_node_boundaries = np.zeros(len(x)+1)
x_node_boundaries[0] = x_min
x_node_boundaries[-1] = x_max
for i in range(1,len(x_node_boundaries)-1):
    x_node_boundaries[i] = 0.5*(x[i-1] + x[i])

delta_x = np.zeros(len(x_node_boundaries)-1)
for i in range(len(delta_x)):
    delta_x[i] = x_node_boundaries[i+1] - x_node_boundaries[i]
"""Temporal Grid"""

t_min = 0
t_max = 1000
delta_t = 1

""" Defining Equation parameters """

n = np.zeros_like(x) #number density in each cell
""" Define Initial Conditions """
mass = 1
n[-1] = mass/(delta_x[-1]*x[-1]) # This is done to make sure that the mass of the 
                                 # overall system is not ffected by change in grid size

g = x*n # particle mass density 

""" Defining selection and breakage functions """

def selection(e):
    return e*e

def breakage(epsilon,u):
    #recheck this function
    return 2/epsilon

"""Defining d for flux matrix A """

#i starts from 0, The first node is represented by 0 and the 
#value at his node is x_{0}. When connecting to the literature 
# the zeroth node represents i = 1

def inner_integral_function(x,a,b):
    return x*breakage(a,x)*b

def derec(x, x_node_boundaries,k,i):
    #     x = x_node_boundaries
    d_k_i = 0
    inner_integral = 0
    if i>=0:
        for j in range(i+1): 
            inner_integral = inner_integral + x[j]*breakage(x[k], x[j])*delta_x[j]
    else :
        inner_integral = 0
            
    d_k_i = selection(x[k])*(1/x[k])*delta_x[k]*inner_integral
    return d_k_i


# k,i = 4,1
# d_k_i = 0
# inner_integral = 0
# for i_node in range(i):
#     delta_x_i = x_node_boundaries[i+1] - x_node_boundaries[i] 
#     inner_integral = inner_integral + x[i_node]*breakage(x[k], x[i_node])*delta_x_i
#     print(inner_integral)
# delta_k = x_node_boundaries[k+1] - x_node_boundaries[k]
# d_k_i = selection(x[k])*(1/x[k])*delta_k*inner_integral     
        
# now creating A matrix 

A = np.zeros((len(x),len(x)))
for i in range(len(x)):
    print(i) 
    A[i,i] = -(1/delta_x[i])*derec(x, x_node_boundaries, i, i-1) # diagonal elements
    for k in range(i+1,len(x)):
        print(i,k,derec(x, x_node_boundaries, k, i-1),derec(x, x_node_boundaries, k, i))
        A[i,k] = -(1/delta_x[i]) * (derec(x, x_node_boundaries, k, i-1) - 
                                  derec(x, x_node_boundaries, k, i))


""" Defining Differential function """


def pbe_ode(t,y):
    return np.matmul(A,y)


t_solve = np.linspace(t_min, t_max, 10)
g_solve = solve_ivp(pbe_ode, [t_min,t_max],g,t_eval = t_solve, vectorized=True)
Mass_at_each_time = np.matmul(g_solve.y.transpose(),delta_x)


num_density = np.zeros_like(g_solve.y)
for i in range(len(g_solve.y[1,:])):
    num_density[:,i] =  1*g_solve.y[:,i]/(x)

plt.figure()
plt.semilogx(x,num_density[:,-1], '*k-', label = "numerical")
# plt.legend()
# plt.grid(which = 'both')





"""Analytical Solution """
# for linear selection function and bunary breakage function 

### Initial condition is monodisperse 

# def num_density_analytical(x):
#     t = 1000
#     num_density = np.zeros(len(x))
#     num_density[:-2] = np.exp(-t*x[:-2])*(2*t + t*t*(x[-1]-x[:-2]))
#     num_density[-1] = np.exp(-t*x[-1])
#     return num_density
    

# num_density_ana = num_density_analytical(x)
# plt.semilogx(x, num_density_ana,'r-', label = 'Analytical')


# Defining error

# """ Impementing Euler explicit time solver """

# delta_t = 0.01
# t = np.linspace(t_min,t_max,int((t_max - t_min)/delta_t) + 1)
# g_euler_eplicit = np.zeros([len(t),len(g)])
# g_euler_eplicit[0,:] = g
# for time in range(len(t)-1):
#     g_euler_eplicit[time+1,:] = g_euler_eplicit[time,:] + (t[time+1] - t[time])*np.matmul(A, g_euler_eplicit[time,:])


# num_density_euler_explicit = np.zeros_like(g_euler_eplicit)
# for i in range(len(g_euler_eplicit[:,1])):
#     num_density_euler_explicit[i,:] =  g_euler_eplicit[i,:]/(x)

# plt.semilogx(x,num_density_euler_explicit[-1,:], 'vg-', label = "numerical (Euler's')")

# plt.legend()
# plt.grid(which = 'both')


### Analytical solution for S(x) = X^2.

def num_density_analytical_sel_x_sqaure(x):
    t = 1000
    num_density = np.zeros(len(x))
    num_density[:-2] = np.exp(-t*x[:-2]*x[:-2])*(2*t*x[-1])
    num_density[-1] = np.exp(-t*x[-1]*x[-1])
    return num_density

x_ana = np.logspace(np.log10(x[0]),np.log10(x[-1]),1000)
num_density_ana_x_squared = num_density_analytical_sel_x_sqaure(x_ana)
plt.semilogx(x_ana, num_density_ana_x_squared,'r-', label = 'Analytical ($S(x) = x^{2}$)')
plt.legend()
plt.grid(which = 'both')






