# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 00:38:23 2020

@author: kanishk
Debug file for pure aggregation
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import matplotlib.pyplot as plt
from Grid_X import GridX
from TemporalGrid import TemporalGrid
from CoagulationFunction import CoagulationFunction
from AnalyticalSolution import AnalyticalSolution
from InitialCondition import InitialConditionNumberDensity

########Initial Conditions and boundary conditions ########
# Minimum particle size 
minimimum_particle_size = 1e-5
# maximum particle size 
maximum_particle_size = 1e5
#initial time
t_min = 0 
# Final Time
t_max = 0.5
# Save solution after Delta_t time 
delta_t = 0.1

no_of_nodes = 128
    
x, x_node_boundaries,delta_x = GridX(minimimum_particle_size,
                                      maximum_particle_size,
                                      no_of_nodes).UniformLogarithmic()

# x, x_node_boundaries,delta_x = GridX(minimimum_particle_size,
#                                       maximum_particle_size,
#                                       no_of_nodes).Uniform()
#solution parameters 
#number density

# n = InitialConditionNumberDensity(x,delta_x).OneLargeParticle()

n = InitialConditionNumberDensity(x,delta_x).FilbetAndLaurencotExponentialDistributionForSumKernel()

# n = np.ones(len(x))
# particle mass density
g0 = x*n

### Getting spatial matrix A pure aggregation ###

### Getting spatial matrix A pure aggregation ###
class APureAggregation:
    ### Getting spatial matrix A pure aggregation ###
    # we need to add g array from previous timestep as the solution is now dependent on g 
    def __init__(self, x, x_node_boundaries, delta_x,g):
        self.x = x
        self.x_node_boundaries = x_node_boundaries
        self.delta_x = delta_x
        self.g = g # it is the mass density matrix
    
    def derec(self,i,k):
        ### Evaluating alpha
        x = self.x
        x_node_boundaries = self.x_node_boundaries
        delta_x = self.delta_x
        g = self.g
        value = x_node_boundaries[i+1] - x[k]
        array = np.asarray(x_node_boundaries)
        idx = (np.abs(array - value)).argmin()
        if value > array[idx]:
            alpha = idx + 1
        elif value == 0:
            alpha = idx + 1
            ## on the last cell the node boundary value is same as the node value
            #henceforth value = x_node_boundaries[i+1] - x[k] came out to be zero.
            ## For an integral with 1/x a zero value will result in sigularity. 
            ## to avoid this the value of 0 is made equivalent to the size of the smallest particle.
            value = x_node_boundaries[0] ## making value equal to smallest particle
            # value = 1e-12
        else:
            alpha = idx
        # print(x[alpha:len(x)],alpha)
        rhs_term1 = np.sum(g[alpha:]*(1/x[alpha:])*
                           CoagulationFunction(x[alpha:], x[k], 'Product')
                           *delta_x[alpha:])
        
        def integrand(x,a):
            return (1/x)*CoagulationFunction(x,a,'Product')
        # using sci py quad function for integration 
        a = x[k]
        I = quad(integrand, value, x_node_boundaries[alpha],args=(a))
        rhs_term2 = g[alpha - 1] * I[0]
        
        # # using mid point rule instead of integrand
        # x_midpoint = (value + x_node_boundaries[alpha])/2
        # rhs_term2 = integrand(x_midpoint, x[k])*(x_node_boundaries[alpha] - value)
        # print(I,alpha,value,x_node_boundaries[alpha])
        return rhs_term1 +rhs_term2
         

    def A(self):
        A = np.zeros((len(self.x),len(self.x)))
        J = np.zeros((len(self.x),len(self.x)))
        for i in range(len(self.x)):
            for j in range(i+1):
                J[i,j] = self.delta_x[j] * self.derec(i,j)
        for i in range(len(A[:,0])):
            if i == 0:
                A[i,:] = (-1/self.delta_x[i]) * J[i,:]
            else:
                A[i,:] = (-1/self.delta_x[i]) * (J[i,:] - J[i-1,:])
        return A

# g_at_previous_time = g
A = APureAggregation(x, x_node_boundaries,delta_x,g0).A()


"""Linear Algebric solver """
## Implementing EulerExplicitSolver as A mtrix depends on time 
## The solver evaluates the particle mass density.
""" Euler Explicit Solver start """
# creating time array 
no_of_timesteps = int((t_max - t_min)/delta_t)
t = np.linspace(t_min,t_max,no_of_timesteps+1)

g_euler_eplicit = np.zeros([len(t),len(g0)])
g_euler_eplicit[0,:] = g0
for time in range(len(t)-1):
    print(t[time])
    A = APureAggregation(x, x_node_boundaries,delta_x,g_euler_eplicit[time]).A()
    g_euler_eplicit[time+1,:] = g_euler_eplicit[time,:] + (t[time+1] - t[time])*np.matmul(A, g_euler_eplicit[time,:])

""" Euler Explicit Solver Ends """

""" Using python inbuilt solvers """
# A = APureAggregation(x, x_node_boundaries,delta_x,g0).A()
def pbe_ode(t,y):
    dy_dt = np.matmul(APureAggregation(x, x_node_boundaries,delta_x,y).A(),y)
    return dy_dt

t_solve = TemporalGrid(t_min,t_max,delta_t).Uniform()
    
g_solve = solve_ivp(pbe_ode, [t_min,t_max],g0,t_eval = t_solve, method = 'RK45')

num_density_solve_ivp = np.zeros_like(g_solve.y)
for i in range(len(g_solve.y[1,:])):
    num_density_solve_ivp[:,i] =  1*g_solve.y[:,i]/(x)

num_density_solve_ivp = num_density_solve_ivp.transpose()
""" Using python inbuilt solvers Ends"""

g_euler_eplicit = g_solve.y.transpose()
num_density_euler_explicit = np.zeros_like(g_solve.y.transpose())
total_mass = np.zeros(len(num_density_euler_explicit[:,0]))
total_number_of_particles = np.zeros(len(num_density_euler_explicit[:,0]))
for i in range(len(g_euler_eplicit[:,1])):
    num_density_euler_explicit[i,:] =  g_euler_eplicit[i,:]/(x)
    total_mass[i] =  np.sum(num_density_euler_explicit[i,:]*x*delta_x)
    total_number_of_particles[i] = np.sum(num_density_euler_explicit[i,:]*delta_x)



"""  Linear Algebric Solver ends """

fig, ax1 = plt.subplots(1,1)
fig.set_figheight(10)
fig.set_figwidth(10)
ax1.semilogx(x, num_density_euler_explicit[1,:]*x,'--r*', markevery = 5, label = 'Numerical')
ax1.semilogx(x, num_density_solve_ivp[1,:]*x,'--b*', markevery = 5, label = 'Numerical Solve_ivp')
num_den_ana = AnalyticalSolution(x, t_max, delta_x = delta_x).ProductKernelCoagulationNormExpInitialCondition()


ax1.semilogx(x, num_den_ana*x,'-b*', markevery = 5, label = 'Analytical')
# ax1.semilogx(x,num_density_euler_explicit[0,:]*x,':k', label = 'Initial Condition')
ax1.legend(fontsize = 20, frameon = False)
ax1.tick_params(axis ='both', labelsize = 20,grid_alpha=0.5)
ax1.set_xlabel(r'$x$',fontsize=32)
ax1.set_ylabel(r'$g$',fontsize=32, rotation = 'horizontal')
ax1.xaxis.set_label_coords(0.55,-0.06)
ax1.yaxis.set_label_coords(-0.125,0.5)
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
ax1.yaxis.offsetText.set_fontsize(20)
ax1.set_xlim(minimimum_particle_size,maximum_particle_size)
ax1.set_ylim(0,)
ax1.grid()

# evaluating n
fig, ax1 = plt.subplots(1,1)
fig.set_figheight(10)
fig.set_figwidth(10)
ax1.semilogx(x, num_density_euler_explicit[-1,:],'--r*', markevery = 5, label = 'Numerical')
num_den_ana = AnalyticalSolution(x, t_max, delta_x = delta_x).ProductKernelCoagulationNormExpInitialCondition()

ax1.semilogx(x, num_den_ana,'-b*', markevery = 5, label = 'Analytical')
# ax1.semilogx(x,num_density_euler_explicit[0,:],':k', label = 'Initial Condition')
ax1.legend(fontsize = 20, frameon = False)
ax1.tick_params(axis ='both', labelsize = 20,grid_alpha=0.5)
ax1.set_xlabel(r'$x$',fontsize=32)
ax1.set_ylabel(r'$n$',fontsize=32, rotation = 'horizontal')
ax1.xaxis.set_label_coords(0.55,-0.06)
ax1.yaxis.set_label_coords(-0.125,0.5)
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
ax1.yaxis.offsetText.set_fontsize(20)
ax1.set_xlim(minimimum_particle_size,maximum_particle_size)
ax1.set_ylim(0,)
ax1.grid()

### Evaluating number of particles with respect to x

fig, ax1 = plt.subplots(1,1)
fig.set_figheight(10)
fig.set_figwidth(10)
ax1.semilogx(x, num_density_euler_explicit[-1,:]*delta_x,'--r*', markevery = 5, label = 'Numerical')
num_den_ana = AnalyticalSolution(x, t_max, delta_x = delta_x).ProductKernelCoagulationNormExpInitialCondition()

ax1.semilogx(x, num_den_ana*delta_x,'-b*', markevery = 5, label = 'Analytical')
# ax1.semilogx(x,num_density_euler_explicit[0,:],':k', label = 'Initial Condition')
ax1.legend(fontsize = 20, frameon = False)
ax1.tick_params(axis ='both', labelsize = 20,grid_alpha=0.5)
ax1.set_xlabel(r'$x$',fontsize=32)
ax1.set_ylabel(r'$N$',fontsize=32, rotation = 'horizontal')
ax1.xaxis.set_label_coords(0.55,-0.06)
ax1.yaxis.set_label_coords(-0.125,0.5)
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
ax1.yaxis.offsetText.set_fontsize(20)
ax1.set_xlim(minimimum_particle_size,maximum_particle_size)
ax1.set_ylim(0,)
ax1.grid()




total_mass_ana = np.zeros(len(num_density_euler_explicit[:,0]))
total_num_particles_ana = np.zeros(len(num_density_euler_explicit[:,0]))
for time in range(len(t)-1):
    num_den_ana = AnalyticalSolution(x, t[time], 
                                     delta_x = delta_x).ProductKernelCoagulationNormExpInitialCondition()
    total_mass_ana[time] = np.sum(num_den_ana*x*delta_x)
    total_num_particles_ana[time] = np.sum(num_den_ana*delta_x)
fig, ax1 = plt.subplots(1,1)
fig.set_figheight(10)
fig.set_figwidth(10)
ax1.plot(t,total_mass,':k*',label = 'Numerical')
ax1.plot(t,total_mass_ana,':r',label = 'Ananlytical')
ax1.legend(fontsize = 20, frameon = False)
ax1.tick_params(axis ='both', labelsize = 20,grid_alpha=0.5)
ax1.set_xlabel(r'$t$',fontsize=32)
ax1.set_ylabel(r'$M_{1}$',fontsize=32, rotation = 'horizontal')
ax1.xaxis.set_label_coords(0.55,-0.06)
ax1.yaxis.set_label_coords(-0.125,0.5)
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
ax1.yaxis.offsetText.set_fontsize(20)
ax1.set_xlim(t_min,t_max)
ax1.set_ylim(total_mass_ana[0]-0.5,total_mass_ana[0]+0.5)
ax1.grid()



fig, ax1 = plt.subplots(1,1)
fig.set_figheight(10)
fig.set_figwidth(10)
ax1.plot(t,total_number_of_particles,':k*',label = 'Numerical')
ax1.plot(t,total_num_particles_ana,':r',label = 'Ananlytical')
ax1.legend(fontsize = 20, frameon = False)
ax1.tick_params(axis ='both', labelsize = 20,grid_alpha=0.5)
ax1.set_xlabel(r'$t$',fontsize=32)
ax1.set_ylabel(r'$M_{0}$',fontsize=32, rotation = 'horizontal')
ax1.xaxis.set_label_coords(0.55,-0.06)
ax1.yaxis.set_label_coords(-0.125,0.5)
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
ax1.yaxis.offsetText.set_fontsize(20)
ax1.set_xlim(t_min,t_max)
ax1.set_ylim(total_mass_ana[0]-0.5,total_mass_ana[0]+0.5)
ax1.grid()


# # Minimum particle size 
# minimimum_particle_size = 0
# # maximum particle size 
# maximum_particle_size = 14
# #initial time

# no_of_nodes = 1000
    
# x, x_node_boundaries,delta_x = GridX(minimimum_particle_size,
#                                       maximum_particle_size,
#                                       no_of_nodes).Uniform()
# n = np.exp(-x)
# N0 = np.sum(n*delta_x)
# M1 = np.sum(x*n*delta_x)
# x0 = M1/N0 
# n_norm = (N0/x0)*np.exp(-x/x0)

# plt.plot(x,n,'-r', label = 'n')
# plt.plot(x,n_norm,'-k', label = 'norm_n')

# T_a = N0*1*t
# num_den_ana = ((4*N0)/(x0*np.power(T_a+2,2))) * np.exp(-2*x/(x0*(T_a + 2)))
    
























