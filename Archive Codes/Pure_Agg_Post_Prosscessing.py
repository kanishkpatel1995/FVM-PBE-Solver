# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 18:37:21 2020

@author: kanishk

Post Proceessing file for Aggregation 
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os as os
from AnalyticalSolution import AnalyticalSolution
from MainSolver import PureAggregtionPBESolver
from Grid_X import GridX
from Moments import moments
### managing fonts for plots 

plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'
##############################

########Initial Conditions and boundary conditions ########
# Minimum particle size 
minimum_particle_size = 1e-4
# maximum particle size 
maximum_particle_size = 100
#initial time
t_min = 0 
# Final Time
t_max = 0.3
# Save solution after Delta_t time 
delta_t = 0.01

""" Comparision with analytical solution """
    
fig, (ax1,ax2) = plt.subplots(1,2)
fig.subplots_adjust(wspace = 0.2)
fig.set_figheight(6)
fig.set_figwidth(14)
linestyles = ['--ko', '-r*','-.gv','-b^',':c>']
mk = [1,5,6,12,24]
i = 0
for nodes in [32]:
    x,delta_x,t,num_density,total_mass,total_number_of_particles = PureAggregtionPBESolver(
                   minimum_particle_size, 
                   maximum_particle_size,
                   nodes,
                   t_min, 
                   t_max,
                   delta_t,
                   type_coagulation_function = 'CandT', 
                   type_of_initial_condition = 'LND', 
                   temporal_solver = 'RK45')
    ax1.semilogx(x,num_density[-1,:],linestyles[i],markevery=mk[i],
                 markersize = 7, 
                 label = r'Numerical ($N_x = {}$)'.format(nodes))
    i = i+1
# x,x_node_boundaries,delta_x = GridX(minimum_particle_size,maximum_particle_size,1000).UniformLogarithmic()
# num_den_ana = AnalyticalSolution(x, t_max, delta_x = delta_x).ConstantUnityCoagulationNormExpInitialCondition()
# ax1.semilogx(x,num_den_ana, ':k', label = 'Analytical')
# plt.legend(fontsize = 20)
ax1.semilogx(x,num_density[0,:],linestyles[i],markevery=mk[i],
                 markersize = 7, 
                 label = r'Initial ($N_x = {}$)'.format(nodes))
ax1.tick_params(axis ='both', labelsize = 20,grid_alpha=0.5)
ax1.set_xlabel(r'$x$',fontsize=32)
ax1.set_ylabel(r'$n$',fontsize=32, rotation = 'horizontal')
ax1.xaxis.set_label_coords(0.55,-0.06)
ax1.yaxis.set_label_coords(-0.125,0.5)
ax1.yaxis.offsetText.set_fontsize(20)
ax1.set_title(r'Case 3', fontsize = 32)
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
ax1.set_xlim(minimum_particle_size,maximum_particle_size)
ax1.set_ylim(0,)
ax1.grid()

i = 0
for nodes in [16,32]:
    x,delta_x,t,num_density,total_mass,total_number_of_particles = PureAggregtionPBESolver(
                   minimum_particle_size, 
                   maximum_particle_size,
                   nodes,
                   t_min, 
                   t_max,
                   delta_t,
                   type_coagulation_function = 'Product', 
                   type_of_initial_condition = 'LND', 
                   temporal_solver = 'RK45')
    ax2.semilogx(x,num_density[-1,:],linestyles[i],markevery=mk[i],
                  markersize = 7, 
                  label = r'Numerical ($N_x = {}$)'.format(nodes))
    i = i+1
x,x_node_boundaries,delta_x = GridX(minimum_particle_size,maximum_particle_size,1000).UniformLogarithmic()
num_den_ana = AnalyticalSolution(x, t_max, delta_x = delta_x).ProductKernelCoagulationNormExpInitialCondition()
ax2.semilogx(x,num_den_ana, ':k', label = 'Analytical')
plt.legend(fontsize = 20, frameon = False)
ax2.tick_params(axis ='both', labelsize = 20,grid_alpha=0.5)
ax2.set_xlabel(r'$x$',fontsize=32)
ax2.set_ylabel(r'$n$',fontsize=32, rotation = 'horizontal')
ax2.xaxis.set_label_coords(0.55,-0.06)
ax2.yaxis.set_label_coords(-0.125,0.45)
ax2.yaxis.offsetText.set_fontsize(20)
ax2.set_title(r'Case 4', fontsize = 32)
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
ax2.set_xlim(minimum_particle_size,maximum_particle_size)
ax2.set_ylim(0,)
ax2.grid()
fig.savefig('Comparision_num_ana_pure_aggregation', dpi=300,
        transparent=True, bbox_inches='tight' )

### Comparing particle mass density

fig, (ax1,ax2) = plt.subplots(1,2)
fig.subplots_adjust(wspace = 0.2)
fig.set_figheight(6)
fig.set_figwidth(14)
linestyles = ['--ko', '-r*','-.gv','-b^',':c>']
mk = [1,5,6,12,24]
i = 0
for nodes in [8,16,32,64]:
    x,delta_x,t,num_density,total_mass,total_number_of_particles = PureAggregtionPBESolver(
                   minimum_particle_size, 
                   maximum_particle_size,
                   nodes,
                   t_min, 
                   t_max,
                   delta_t,
                   type_coagulation_function = 'ConstantUnity', 
                   type_of_initial_condition = 'ExponentialDistribution', 
                   temporal_solver = 'RK45')
    ax1.semilogx(x,x*num_density[-1,:],linestyles[i],markevery=mk[i],
                 markersize = 7, 
                 label = r'Numerical ($N_x = {}$)'.format(nodes))
    i = i+1
x,x_node_boundaries,delta_x = GridX(minimum_particle_size,maximum_particle_size,1000).UniformLogarithmic()
num_den_ana = AnalyticalSolution(x, t_max, delta_x = delta_x).ConstantUnityCoagulationNormExpInitialCondition()
ax1.semilogx(x,x*num_den_ana, ':k', label = 'Analytical')
# plt.legend(fontsize = 20)
ax1.tick_params(axis ='both', labelsize = 20,grid_alpha=0.5)
ax1.set_xlabel(r'$x$',fontsize=32)
ax1.set_ylabel(r'$g$',fontsize=32, rotation = 'horizontal')
ax1.xaxis.set_label_coords(0.55,-0.06)
ax1.yaxis.set_label_coords(-0.125,0.5)
ax1.yaxis.offsetText.set_fontsize(20)
ax1.set_title(r'Case 3', fontsize = 32)
ax1.set_xlim(minimum_particle_size,maximum_particle_size)
ax1.set_ylim(0,)
ax1.grid()
# ax1.ticklabel_format(axis = 'y', style = 'sci')

i = 0
for nodes in [8,16,32,64]:
    x,delta_x,t,num_density,total_mass,total_number_of_particles = PureAggregtionPBESolver(
                   minimum_particle_size, 
                   maximum_particle_size,
                   nodes,
                   t_min, 
                   t_max,
                   delta_t,
                   type_coagulation_function = 'Product', 
                   type_of_initial_condition = 'ExponentialDistribution', 
                   temporal_solver = 'RK45')
    ax2.semilogx(x,x*num_density[-1,:],linestyles[i],markevery=mk[i],
                 markersize = 7, 
                 label = r'Numerical ($N_x = {}$)'.format(nodes))
    i = i+1
x,x_node_boundaries,delta_x = GridX(minimum_particle_size,minimum_particle_size,1000).UniformLogarithmic()
num_den_ana = AnalyticalSolution(x, t_max, delta_x = delta_x).ProductKernelCoagulationNormExpInitialCondition()
ax2.semilogx(x,x*num_den_ana, ':k', label = 'Analytical')
plt.legend(fontsize = 20, frameon = False)
ax2.tick_params(axis ='both', labelsize = 20,grid_alpha=0.5)
ax2.set_xlabel(r'$x$',fontsize=32)
ax2.set_ylabel(r'$g$',fontsize=32, rotation = 'horizontal')
ax2.xaxis.set_label_coords(0.55,-0.06)
ax2.yaxis.set_label_coords(-0.125,0.5)
ax2.yaxis.offsetText.set_fontsize(20)
ax2.set_title(r'Case 4', fontsize = 32)
ax2.set_xlim(minimum_particle_size,maximum_particle_size)
ax2.set_ylim(0,)
ax2.grid()
ax2.ticklabel_format(axis = 'y', style = 'sci')
fig.savefig('Comparision_g_pure_aggregation', dpi=300,
        transparent=True, bbox_inches='tight' )


#Prediction of moments 


""" Error Analysis """


##### Error with time ############
f = 24
fig, ((ax1,ax2)) = plt.subplots(1,2)
fig.subplots_adjust(wspace = 0.2, hspace = 0.2)
fig.set_figheight(15)
fig.set_figwidth(15)
linestyles = ['--ko', '-r*','-.gv','-b^',':c>',':m<',':ks']
mk = [1,1,1,1,1]
i =  0
for nodes in [8,16,32,64,128,128*2]:
    os.chdir('Error/PureBreakage/WithTime')
    errortimeExpDist = np.load('L2_errorPureAggregationN' + str(nodes) +'ConstantUnityInitialCond.npy')
    os.chdir('../../..')
    ax1.semilogy(errortimeExpDist[:,0],errortimeExpDist[:,1]/(nodes**2),linestyles[i], markevery = 2,
             label = r'$N_x =$ {}'.format(nodes))
    i = i+1
    ax1.tick_params(axis ='both', labelsize = f,grid_alpha=0.5)
    ax1.set_xlabel(r'$t$',fontsize=f+8)
    ax1.set_ylabel(r'$L_{2}$',fontsize=f, rotation = 'horizontal')
    ax1.xaxis.set_label_coords(0.5,-0.06)
    ax1.yaxis.set_label_coords(-0.125,0.5)
    ax1.set_xlim(0,0.9)
    # ax1.set_ylim(0,0.3)
    ax1.set_title(r'Case 3', fontsize = f+4, fontweight = 'bold')
    # ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
    ax1.yaxis.offsetText.set_fontsize(f)
    ax1.legend(fontsize = f, frameon  = False, ncol = 2)
    ax1.grid()
    
i = 0
for nodes in [8,16,32,64,128,128*2,128*4]:
    os.chdir('Error/PureBreakage/WithTime')
    errortimeExpDist = np.load('L2_errorPureAggregationN' + str(nodes) +'ProductInitialCond.npy')
    os.chdir('../../..')
    ax2.semilogy(errortimeExpDist[:,0],errortimeExpDist[:,1]/(nodes**2),linestyles[i], markevery = 2,
             label = r'$N_x =$ {}'.format(nodes))
    i = i+1
    ax2.tick_params(axis ='both', labelsize = f,grid_alpha=0.5)
    ax2.set_xlabel(r'$t$',fontsize=f+8)
    ax2.set_ylabel(r'$L_{2}$',fontsize=f, rotation = 'horizontal')
    ax2.set_title(r'Case 4', fontsize = f+4, fontweight = 'bold')
    ax2.xaxis.set_label_coords(0.5,-0.06)
    ax2.yaxis.set_label_coords(-0.125,0.5)
    # ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
    ax2.yaxis.offsetText.set_fontsize(f)
    ax2.set_xlim(0,0.9)
    # ax1.set_ylim(0,0.3)
    ax2.grid()


fig.savefig('L1_L2_error_time_Aggregation', dpi=300,
        transparent=True, bbox_inches='tight' )


### Analysing Moments
fig, ax1 = plt.subplots(1,1)
fig.set_figheight(10)
fig.set_figwidth(10)
t_max = 0.3
delta_t = 0.001
linestyles = ['--ko', '-b*','-.gv','-b^',':c>']
mk = [1,5,12,24,48]
i = 0
k = 1 # refers to the moment
for nodes in [32,64,128]:
    x,delta_x,t,num_density,total_mass,total_number_of_particles = PureAggregtionPBESolver(
                   minimum_particle_size, 
                   maximum_particle_size,
                   nodes,
                   t_min, 
                   t_max,
                   delta_t,
                   type_coagulation_function = 'ConstantUnity', 
                   type_of_initial_condition = 'ExponentialDistribution', 
                   temporal_solver = 'RK45')
    
    ax1.plot(t,total_mass-1, linestyles[i],
             markevery = 5, label = r'$N_x =$ {}'.format(nodes))
    i = i+1
    
zeroth_moment_ana = moments.analytical_moment_PBE(maximum_particle_size,minimum_particle_size,nodes,t_min,t_max,delta_t,
                          moment = k,
                          which_selection_function = None,
                          which_coagulation_function = 'ConstantUnity',
                          which_disrtibution = 'ExponentialDistribution')
ax1.plot(zeroth_moment_ana[:,0],zeroth_moment_ana[:,1]-1,':r', markevery = 5, label = 'Analytical')
ax1.legend(fontsize = 20, frameon = False)
ax1.tick_params(axis ='both', labelsize = 20,grid_alpha=0.5)
ax1.set_xlabel(r'$t$',fontsize=32)
ax1.set_ylabel(r'$M^{n}_{1}$',fontsize=32, rotation = 'horizontal')
ax1.xaxis.set_label_coords(0.55,-0.06)
ax1.yaxis.set_label_coords(-0.12,0.48)
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
ax1.set_title('Case 3',fontsize = 32)
ax1.yaxis.offsetText.set_fontsize(20)
ax1.set_xlim(0,0.3)
# ax1.set_ylim(-0.8,-0.94)
ax1.grid()
# L2_error_zeroth_moment = (1/zeroth_moment_ana[-1,1])*np.sqrt(np.mean(abs(zeroth_moment_ana[:-1,1]**2 - 
#                                              total_mass**2)))
fig.savefig('first_moment_ConstantUnity_exp_dis', dpi=300,
        transparent=True, bbox_inches='tight' )




