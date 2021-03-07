# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 18:28:49 2020

@author: kanishk

This is the post processing file the PBE solver can be used here to generate plot of desired quality

Implementation of PBE solver

PureBreakagePBESolver(minimimum_particle_size, 
               maximum_particle_size,
               no_of_nodes,
               t_min, 
               t_max,
               delta_t,
               type_of_selection_function,type_of_initial_condition)
There are two types of possible Selection functions 1. "Linear" 2. "Squared"
There are two types of possible initial conditions 1. "OneLargeParticle" 2. "ExponentialDistribution"
"""
import matplotlib.pyplot as plt
import matplotlib
# import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np
import os as os
from Grid_X import GridX
from MainSolver import PureBreakagePBESolver
from AnalyticalSolution import AnalyticalSolution

### managing fonts for plots 

plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'
##############################



minimimum_particle_size = 1e-8
# maximum particle size 
maximum_particle_size = 1
#initial time

no_of_nodes = 31
t_min = 0 
# Final Time
t_max = 2000
# Save solution after Delta_t time 
delta_t = 10


""" Comparision with analytical solution """

### Comparing number density


t_max = 1000
fig, (ax1,ax2) = plt.subplots(1,2)
fig.subplots_adjust(wspace = 0.2)
fig.set_figheight(6)
fig.set_figwidth(14)
linestyles = ['--ko', '-r*','-.gv','-b^',':c>']
mk = [1,5,12,24,48]
i = 0
for nodes in [8,16,32]:
    x, delta_x,num_density = PureBreakagePBESolver(minimimum_particle_size, 
                    maximum_particle_size,
                    nodes,
                    t_min, 
                    t_max,
                    delta_t,
                    'Linear', 'ExponentialDistribution')
    ax1.semilogx(x,num_density[:,-1],linestyles[i],markevery=mk[i],
                 markersize = 7, 
                 label = r'Numerical ($N_x = {}$)'.format(nodes))
    i = i+1
    
num_den_ana = AnalyticalSolution(x,t_max).BinaryBreakageLinearSelectionExpInitialCondition()
ax1.semilogx(x,num_den_ana, ':k', label = 'Analytical')
plt.legend(fontsize = 20)
ax1.tick_params(axis ='both', labelsize = 20,grid_alpha=0.5)
ax1.set_xlabel(r'$x$',fontsize=32)
ax1.set_ylabel(r'$n$',fontsize=32, rotation = 'horizontal')
ax1.xaxis.set_label_coords(0.55,-0.06)
ax1.yaxis.set_label_coords(-0.125,0.5)
ax1.yaxis.offsetText.set_fontsize(20)
ax1.set_title(r'Case 1', fontsize = 32)
ax1.set_xlim(1e-8,1)
ax1.set_ylim(0,)
ax1.grid()
ax1.ticklabel_format(axis = 'y', style = 'sci')

i =0
for nodes in [8,16,32]:
    x, delta_x,num_density = PureBreakagePBESolver(minimimum_particle_size, 
                    maximum_particle_size,
                    nodes,
                    t_min, 
                    t_max,
                    delta_t,
                    'Squared', 'ExponentialDistribution')
    ax2.semilogx(x,num_density[:,-1],linestyles[i],markevery=mk[i],
                 markersize = 7, 
                 label = r'$N_x = {}$'.format(nodes))
    i = i+1
    
num_den_ana = AnalyticalSolution(x,t_max).BinaryBreakageSquareSelectionExpInitialCondition()
ax2.semilogx(x,num_den_ana, ':k', label = 'Analytical')
plt.legend(fontsize = 20, frameon = False)
ax2.tick_params(axis ='both', labelsize = 20,grid_alpha=0.5)
ax2.set_xlabel(r'$x$',fontsize=32)
ax2.set_ylabel(r'$n$',fontsize=32, rotation = 'horizontal')
ax2.xaxis.set_label_coords(0.55,-0.06)
ax2.yaxis.set_label_coords(-0.125,0.55)
ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
ax2.yaxis.offsetText.set_fontsize(20)
ax2.set_xlim(1e-8,1)
ax2.set_ylim(0,)
ax2.set_title(r'Case 2', fontsize = 32)
ax2.grid()
fig.savefig('Comparision_num_ana', dpi=300,
        transparent=True, bbox_inches='tight' )


### Comparing Particle mass density

t_max = 1000
fig, (ax1,ax2) = plt.subplots(1,2)
fig.subplots_adjust(wspace = 0.2)
fig.set_figheight(6)
fig.set_figwidth(14)
linestyles = ['--ko', '-r*','-.gv','-b^',':c>']
mk = [1,2,4,8,16]
i = 0
for nodes in [8,16,32,64]:
    x, delta_x,num_density = PureBreakagePBESolver(minimimum_particle_size, 
                    maximum_particle_size,
                    nodes,
                    t_min, 
                    t_max,
                    delta_t,
                    'Linear', 'ExponentialDistribution')
    ax1.semilogx(x,x*num_density[:,-1],linestyles[i],markevery=mk[i],
                 markersize = 7, 
                 label = r'Numerical ($N_x = {}$)'.format(nodes))
    i = i+1

x = np.logspace(np.log10(minimimum_particle_size),np.log10(maximum_particle_size),1000)
num_den_ana = AnalyticalSolution(x,t_max).BinaryBreakageLinearSelectionExpInitialCondition()
ax1.semilogx(x,x*num_den_ana, ':ks', label = 'Analytical',markevery = 100, markersize = 5)
plt.legend(fontsize = 20)
ax1.tick_params(axis ='both', labelsize = 20,grid_alpha=0.5)
ax1.set_xlabel(r'$x$',fontsize=32)
ax1.set_ylabel(r'$g$',fontsize=32, rotation = 'horizontal')
ax1.xaxis.set_label_coords(0.55,-0.06)
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
ax1.yaxis.set_label_coords(-0.125,0.5)
ax1.yaxis.offsetText.set_fontsize(20)
ax1.set_title(r'Case 1', fontsize = 32)
ax1.set_xlim(1e-8,1)
ax1.set_ylim(0,)
ax1.grid()
# ax1.ticklabel_format(axis = 'y', style = 'sci')

i =0
for nodes in [8,16,32,64]:
    x, delta_x,num_density = PureBreakagePBESolver(minimimum_particle_size, 
                    maximum_particle_size,
                    nodes,
                    t_min, 
                    t_max,
                    delta_t,
                    'Squared', 'ExponentialDistribution')
    ax2.semilogx(x,x*num_density[:,-1],linestyles[i],markevery=mk[i],
                 markersize = 7, 
                 label = r'$N_x = {}$'.format(nodes))
    i = i+1

x = np.logspace(np.log10(minimimum_particle_size),np.log10(maximum_particle_size),1000)
num_den_ana = AnalyticalSolution(x,t_max).BinaryBreakageSquareSelectionExpInitialCondition()
ax2.semilogx(x,x*num_den_ana, ':ks', label = 'Analytical',markevery = 100, markersize = 5)
plt.legend(fontsize = 20, frameon = False)
ax2.tick_params(axis ='both', labelsize = 20,grid_alpha=0.5)
ax2.set_xlabel(r'$x$',fontsize=32)
ax2.set_ylabel(r'$g$',fontsize=32, rotation = 'horizontal')
ax2.xaxis.set_label_coords(0.55,-0.06)
ax2.yaxis.set_label_coords(-0.125,0.55)
ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
ax2.yaxis.offsetText.set_fontsize(20)
ax2.set_xlim(1e-8,1)
ax2.set_ylim(0,)
ax2.set_title(r'Case 2', fontsize = 32)
ax2.grid()
fig.savefig('Comparision_particle_mass_density', dpi=300,
        transparent=True, bbox_inches='tight' )



###L1 and L2 error plots 
#################################################################################
# points to annotate 
i= 28
n = np.array([31,62,124,248,496]) - 4
fig, ax4 = plt.subplots(1,1)
fig.subplots_adjust(wspace = 0.2, hspace = 0.2)
fig.set_figheight(12)
fig.set_figwidth(8)
os.chdir('Error/PureBreakage/WithNodes')
errorExpDist_lin = np.load('L2_errorPureBreakageLinearSelectionInitialCondExponentialDistribution.npy')
errorExpDist = np.load('L2_errorPureBreakageSquaredSelectionInitialCondExponentialDistribution.npy')
AerrorExpDist_lin = np.load('L2_errorPureAggregationConstantUnitycoagulationFunctionInitialCond.npy')
AerrorExpDist = np.load('L2_errorPureAggregationProductcoagulationFunctionInitialCond.npy')
os.chdir('../../..')
node_sq = np.power(range(3,len(errorExpDist)+3),2)
# ax4.loglog(range(3,len(errorOneParticle)+3), errorOneParticle/node_sq,'--k', linewidth = 2)
ax4.loglog(range(3,len(errorExpDist)+3), errorExpDist_lin/node_sq,'-.b', linewidth = 2, label = 'Case 1')
ax4.loglog(range(3,len(errorExpDist)+3), errorExpDist/node_sq,'-.r', linewidth = 2,label = 'Case 2')

ax4.loglog(range(3,len(AerrorExpDist_lin)+3), AerrorExpDist_lin,'-.g', linewidth = 2,label = 'Case 3')
ax4.loglog(range(3,len(AerrorExpDist)+3), AerrorExpDist,'-.c', linewidth = 2, label = 'Case 4')
# ax4.scatter([34,65,125], 
#             [errorExpDist_lin[31]/node_sq[31],errorExpDist_lin[62]/node_sq[62],
#              errorExpDist_lin[124]/node_sq[122]],
#             c="b", marker='^',s = 200)
# ax4.scatter([34,65,125], 
#             [errorExpDist[31]/node_sq[31],
#              errorExpDist[62]/node_sq[62],errorExpDist[124]/node_sq[122]],
#             c="r", marker='v',s = 200)
ax4.tick_params(axis ='both', labelsize = i,grid_alpha=0.5)
ax4.set_xlabel(r'$N_{x}$',fontsize=i)
ax4.set_ylabel(r'$L_{2}$',fontsize=i, rotation = 'horizontal')
ax4.xaxis.set_label_coords(0.5,-0.06)
ax4.yaxis.set_label_coords(-0.125,0.6)
first_order_error = np.logspace(4,1,100)
third_order_error = np.logspace(2,-4,100)
node_reduction_th = np.logspace(0,2,100)
node_reduction = np.logspace(0,3,100)

# ax4.plot(node_reduction,first_order_error,':k', label = r'First order')
# ax4.plot(node_reduction_th,third_order_error,'--k', label = r'Third order')
ax4.plot(node_reduction,first_order_error,':k')
ax4.plot(node_reduction_th,third_order_error,'--k')
ax4.grid()
ax4.set_xlim(1,1e3)
ax4.set_ylim(1e-5,1e6)
ax4.legend(frameon = False,fontsize = i, ncol = 1, columnspacing = 0.2)
fig.savefig('L1_L2_error', dpi=300,
        transparent=True, bbox_inches='tight' )

################################################################################
##### Error with time ############
fig, (ax2,ax4) = plt.subplots(1,2)
fig.subplots_adjust(wspace = 0.2, hspace = 0.2)
fig.set_figheight(15)
fig.set_figwidth(15)
linestyles = ['--ko', '-r*','-.gv','-b^',':c>',':m<',':ks']
mk = [1,9,18,36,72]
f = 24
i = 0
for nodes in [8,16,31,62,124,248]:
    os.chdir('Error/PureBreakage/WithTime')
    errortimeExpDist = np.load('L2_errorPureBreakageN'+ str(nodes)
                           +'LinearSelectionInitialCondExponentialDistribution.npy')
    os.chdir('../../..')
    ax2.semilogy(errortimeExpDist[:,0],errortimeExpDist[:,1]/(nodes**2),linestyles[i], markevery = 100,
             label = r'$N_x =$ {}'.format(nodes))
    i = i+1
    ax2.tick_params(axis ='both', labelsize = f,grid_alpha=0.5)
    ax2.set_xlabel(r'$t$',fontsize=f+8)
    ax2.set_ylabel(r'$L_{2}$',fontsize=f, rotation = 'horizontal')
    ax2.set_title(r'Case 1', fontsize = f+4, fontweight = 'bold')
    ax2.xaxis.set_label_coords(0.5,-0.06)
    ax2.yaxis.set_label_coords(-0.125,0.5)
    # ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
    ax2.yaxis.offsetText.set_fontsize(f)
    ax2.legend(fontsize = f, frameon  = False, ncol = 2)
    ax2.set_xlim(0,1000)
    ax2.grid()


i = 0
for nodes in [8,16,31,62,124,248]:
    os.chdir('Error/PureBreakage/WithTime')
    errortimeExpDist = np.load('L2_errorPureBreakageN'+ str(nodes)
                           +'SquaredSelectionInitialCondExponentialDistribution.npy')
    os.chdir('../../..')
    ax4.semilogy(errortimeExpDist[:,0],errortimeExpDist[:,1]/(nodes**2),linestyles[i], markevery = 100,
             label = r'$N_x =$ {}'.format(nodes))
    i = i+1
    ax4.tick_params(axis ='both', labelsize = f,grid_alpha=0.5)
    ax4.set_xlabel(r'$t$',fontsize=f+8)
    ax4.set_ylabel(r'$L_{2}$',fontsize=f, rotation = 'horizontal')
    ax4.xaxis.set_label_coords(0.5,-0.06)
    ax4.yaxis.set_label_coords(-0.125,0.5)
    # ax4.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
    ax4.set_title(r'Case 2', fontsize = f+4, fontweight = 'bold')
    ax4.yaxis.offsetText.set_fontsize(f)
    ax4.set_xlim(0,1000)
    ax4.grid()

fig.savefig('L1_L2_error_time', dpi=300,
        transparent=True, bbox_inches='tight' )





""" Evaluating moments """ 

def analytical_moment_PBE(x_max,x_min,nodes,t_min,t_max,delta_t,
                          moment = 0,
                          which_selection_function = 'Linear', 
                          which_disrtibution = 'OneLargeParticle'):
 
    x,x_node_boundaries,delta_x = GridX(x_min,x_max,nodes).UniformLogarithmic()
    timesteps = (t_max - t_min)/delta_t
    time = np.linspace(t_min, t_max,int(timesteps)+1)
    moment_ana = np.zeros([len(time),2])
    moment_ana[:,0] = time
    for timestep in range(len(time)):
        if which_selection_function == 'Linear' and which_disrtibution == 'OneLargeParticle':
            moment_ana[timestep,1] = np.sum(np.power(x,moment)*delta_x*
            AnalyticalSolution(x,timestep*delta_t).BinaryBreakageLinearSelection())
        elif which_selection_function == 'Squared' and which_disrtibution == 'OneLargeParticle':
            moment_ana[timestep,1] = np.sum(np.power(x,moment)*delta_x*
            AnalyticalSolution(x,timestep*delta_t).BinaryBreakageSquareSelection())
        elif which_selection_function == 'Linear' and which_disrtibution == 'ExponentialDistribution':
            moment_ana[timestep,1] = np.sum(np.power(x,moment)*delta_x*
            AnalyticalSolution(x,timestep*delta_t).BinaryBreakageLinearSelectionExpInitialCondition())
        elif which_selection_function == 'Squared' and which_disrtibution == 'ExponentialDistribution':
            moment_ana[timestep,1] = np.sum(np.power(x,moment)*delta_x*
            AnalyticalSolution(x,timestep*delta_t).BinaryBreakageSquareSelectionExpInitialCondition())
        else:
            print('Selection function or Initial condition not mentioned properly')
    return moment_ana
        
    

    


# The zeroth moment represents the total number of the particles
fig, ax1 = plt.subplots(1,1)
fig.set_figheight(10)
fig.set_figwidth(10)
t_max = 2000
delta_t = 10
linestyles = ['--ko', '-b*','-.gv','-b^',':c>']
mk = [1,5,12,24,48]
i = 0
k = 0 # refers to the moment
for nodes in [32,64,128]:
    x, delta_x,num_density = PureBreakagePBESolver(minimimum_particle_size, 
                    maximum_particle_size,
                    nodes,
                    t_min, 
                    2000,
                    delta_t,
                    'Linear', 'ExponentialDistribution')
    
    zeroth_moment_num = np.zeros([len(num_density[-1,:]),2])
    for timestep in range(len(num_density[-1,:])):
        zeroth_moment_num[timestep,0] = timestep*delta_t
        zeroth_moment_num[timestep,1] = np.sum(np.power(x,k)*num_density[:,timestep]*delta_x)
    ax1.plot(zeroth_moment_num[:,0],zeroth_moment_num[:,1]-1, linestyles[i],
             markevery = 5, label = r'$N_x =$ {}'.format(nodes))
    i = i+1
    
zeroth_moment_ana = analytical_moment_PBE(1,1e-8,1000,0,2000,delta_t,moment = k,
                                          which_selection_function='Linear',
                                          which_disrtibution = 'ExponentialDistribution')
ax1.plot(zeroth_moment_ana[:,0],zeroth_moment_ana[:,1]-1,':r', markevery = 5, label = 'Analytical')
ax1.legend(fontsize = 20, frameon = False)
ax1.tick_params(axis ='both', labelsize = 20,grid_alpha=0.5)
ax1.set_xlabel(r'$t$',fontsize=32)
ax1.set_ylabel(r'$M_{0}}$',fontsize=32, rotation = 'horizontal')
ax1.xaxis.set_label_coords(0.55,-0.06)
ax1.yaxis.set_label_coords(-0.12,0.52)
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(-4,-5))
ax1.set_title('Case 1',fontsize = 32)
ax1.yaxis.offsetText.set_fontsize(20)
ax1.set_xlim(0,2000)
# ax1.set_ylim(-0.0001,0.0001)
ax1.grid()
# ax1.ticklabel_format(axis = 'y', style = 'sci')
# L2_error_zeroth_moment = (1/zeroth_moment_ana[-1,1])*np.sqrt(np.mean(abs(zeroth_moment_ana[:-1,1]**2 - 
#                                              zeroth_moment_num[:,1]**2)))
fig.savefig('zeroth_moment_Linear_exp_dis', dpi=300,
        transparent=True, bbox_inches='tight' )

















