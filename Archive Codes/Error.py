# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 21:48:24 2020

@author: kanishk

Evaluating l2 and l1 norm error and saving arrays for post processing script
to generate needed plot. 

"""

""" Evaluating error for a binary pure breakage with respect to number of nodes """ 

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import numpy as np
import os as os
from MainSolver import PureBreakagePBESolver
from MainSolver import PureAggregtionPBESolver
from AnalyticalSolution import AnalyticalSolution

### managing fonts for plots 

plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'
##############################
def ErrorForPureBreakage():
    minimimum_particle_size = 1e-8
    # maximum particle size 
    maximum_particle_size = 1
    #initial time
    
    no_of_nodes = 31
    t_min = 0 
    # Final Time
    t_max = 1000
    # Save solution after Delta_t time 
    delta_t = 10
    
    """ Evaluating error for a binary pure breakage with respect to number of nodes """ 
    
    maximum_possible_nodes = 496
    minimum_possible_nodes = 3
    
    L1_error = np.zeros(len(range(minimum_possible_nodes,maximum_possible_nodes))) 
    L2_error = np.zeros(len(range(minimum_possible_nodes,maximum_possible_nodes))) 
    for selection_function in ['Linear','Squared']:
        for initial_condition in ['OneLargeParticle', 'ExponentialDistribution']:
            for nodes in range(minimum_possible_nodes,maximum_possible_nodes):
                x, delta_x,num_density = PureBreakagePBESolver(minimimum_particle_size, 
                           maximum_particle_size,
                           nodes,
                           t_min, 
                           t_max,
                           delta_t,
                           selection_function, initial_condition)
                if selection_function == 'Linear' and initial_condition == 'OneLargeParticle':
                    num_den_ana = AnalyticalSolution(x,t_max).BinaryBreakageLinearSelection()
                elif selection_function == 'Squared' and initial_condition == 'OneLargeParticle':
                    num_den_ana = AnalyticalSolution(x,t_max).BinaryBreakageSquareSelection()
                elif selection_function == 'Linear' and initial_condition == 'ExponentialDistribution':
                    num_den_ana = AnalyticalSolution(x,t_max).BinaryBreakageLinearSelectionExpInitialCondition()
                elif selection_function == 'Squared' and initial_condition == 'ExponentialDistribution':
                    num_den_ana = AnalyticalSolution(x,t_max).BinaryBreakageSquareSelectionExpInitialCondition()
                else:
                    print('Selection function or Initial condition not mentioned properly')
                
                L1_error[nodes-minimum_possible_nodes] = np.sum(abs((num_density[:,-1] - num_den_ana)) * delta_x)
                L2_error[nodes-minimum_possible_nodes] = np.sqrt(np.mean(
                    abs(np.power(num_density[:,-1],2) - np.power(num_den_ana,2))))/(nodes+1)
            os.chdir('Error\PureBreakage\WithNodes')
            np.save('L1_errorPureBreakage'+selection_function+'SelectionInitialCond'+initial_condition+'.npy', L1_error)
            np.save('L2_errorPureBreakage'+selection_function+'SelectionInitialCond'+initial_condition+'.npy', L2_error)   
            os.chdir('..\..\..')
    
    for selection_function in ['Linear','Squared']:
        for initial_condition in ['OneLargeParticle', 'ExponentialDistribution']:
            delta_t = 1
            t_max = 1000
            for nodes in [8,16,31,62,124,248]:
                x, delta_x,num_density = PureBreakagePBESolver(minimimum_particle_size, 
                           maximum_particle_size,
                           nodes,
                           t_min, 
                           t_max,
                           delta_t,
                           selection_function, initial_condition)
                L1_error_time = np.zeros([len(num_density[-1,:]),2]) 
                L2_error_time = np.zeros([len(num_density[-1,:]),2]) 
                for timestep in range(len(num_density[-1,:])):
                    L1_error_time[timestep,0] = timestep*delta_t
                    L2_error_time[timestep,0] = timestep*delta_t# at time 
                    ## Selecting nalytical solution depending on the choice of kernels
                    if selection_function == 'Linear' and initial_condition == 'OneLargeParticle':
                        num_den_ana = AnalyticalSolution(x,L1_error_time[timestep,0]).BinaryBreakageLinearSelection()
                    elif selection_function == 'Squared' and initial_condition == 'OneLargeParticle':
                        num_den_ana = AnalyticalSolution(x,L1_error_time[timestep,0]).BinaryBreakageSquareSelection()
                    elif selection_function == 'Linear' and initial_condition == 'ExponentialDistribution':
                        num_den_ana = AnalyticalSolution(x,L1_error_time[timestep,0]).BinaryBreakageLinearSelectionExpInitialCondition()
                    elif selection_function == 'Squared' and initial_condition == 'ExponentialDistribution':
                        num_den_ana = AnalyticalSolution(x,L1_error_time[timestep,0]).BinaryBreakageSquareSelectionExpInitialCondition()
                    else:
                        print('Selection function or Initial condition not mentioned properly')
                    
                    L1_error_time[timestep,1] = np.sum(abs(num_den_ana - num_density[:,timestep])*delta_x)
                    L2_error_time[timestep,1] = np.sqrt(np.mean(
                    abs(np.power(num_density[:,timestep],2) - np.power(num_den_ana,2))))/(nodes+1)
                os.chdir('Error\PureBreakage\WithTime')
                np.save('L1_errorTimePureBreakageN'+str(nodes)+selection_function+'SelectionInitialCond'+initial_condition+'.npy', L1_error_time)
                np.save('L2_errorPureBreakageN'+str(nodes)+selection_function+'SelectionInitialCond'+initial_condition+'.npy', L2_error_time)   
                os.chdir('..\..\..')
    

"""Error for pure aggregation cases """
minimimum_particle_size = 1e-4
# maximum particle size 
maximum_particle_size = 100
#initial time

no_of_nodes = 31
t_min = 0 
# Final Time
t_max = 0.3
# Save solution after Delta_t time 
delta_t = 0.01

maximum_possible_nodes = 496
minimum_possible_nodes = 3


L1_error = np.zeros(len(range(minimum_possible_nodes,maximum_possible_nodes))) 
L2_error = np.zeros(len(range(minimum_possible_nodes,maximum_possible_nodes))) 
for coagulation_function in ['ConstantUnity','Product']:
    for nodes in range(minimum_possible_nodes,maximum_possible_nodes):
        x,delta_x,t_solve,num_density,total_mass,total_number_of_particles = PureAggregtionPBESolver(minimimum_particle_size, 
               maximum_particle_size,
               no_of_nodes,
               t_min, 
               t_max,
               delta_t,
               type_coagulation_function = coagulation_function,
               type_of_initial_condition = 'ExponentialDistribution',
               temporal_solver = 'RK45')
        if coagulation_function == 'ConstantUnity':
            num_den_ana = AnalyticalSolution(x,t_max, delta_x = delta_x).ConstantUnityCoagulationNormExpInitialCondition()
        elif coagulation_function == 'Product':
            num_den_ana = AnalyticalSolution(x,t_max, delta_x = delta_x).ProductKernelCoagulationNormExpInitialCondition()
        else:
            print('Selection function or Initial condition not mentioned properly')
        
        L1_error[nodes-minimum_possible_nodes] = np.sum(abs((num_density[-1,:] - num_den_ana)) * delta_x)
        L2_error[nodes-minimum_possible_nodes] = np.sqrt(np.mean(
            abs(np.power(num_density[-1,:],2) - np.power(num_den_ana,2))))/(nodes+1)
    os.chdir('Error\PureBreakage\WithNodes')
    np.save('L1_errorPureAggregation'+coagulation_function+'coagulationFunctionInitialCond.npy', L1_error)
    np.save('L2_errorPureAggregation'+coagulation_function+'coagulationFunctionInitialCond.npy', L2_error)   
    os.chdir('..\..\..')


for coagulation_function in ['ConstantUnity','Product']:
        delta_t = 0.05
        t_max = 1
        for nodes in [8,16,32,64,128,128*2,128*4]:
            x,delta_x,t_solve,num_density,total_mass,total_number_of_particles = PureAggregtionPBESolver(minimimum_particle_size, 
               maximum_particle_size,
               nodes,
               t_min, 
               t_max,
               delta_t,
               type_coagulation_function = coagulation_function,
               type_of_initial_condition = 'ExponentialDistribution',
               temporal_solver = 'RK45')
            L1_error_time = np.zeros([len(num_density[:,-1]),2]) 
            L2_error_time = np.zeros([len(num_density[:,-1]),2]) 
            for timestep in range(len(num_density[:,-1])):
                L1_error_time[timestep,0] = timestep*delta_t
                L2_error_time[timestep,0] = timestep*delta_t# at time 
                ## Selecting nalytical solution depending on the choice of kernels
                if coagulation_function == 'ConstantUnity':
                    num_den_ana = AnalyticalSolution(x,t_max, delta_x = delta_x).ConstantUnityCoagulationNormExpInitialCondition()
                elif coagulation_function == 'Product':
                    num_den_ana = AnalyticalSolution(x,t_max, delta_x = delta_x).ProductKernelCoagulationNormExpInitialCondition()
                else:
                    print('Selection function or Initial condition not mentioned properly')
                
                L1_error_time[timestep,1] = np.sum(abs(num_den_ana - num_density[timestep,:])*delta_x)
                L2_error_time[timestep,1] = np.sqrt(np.mean(
                abs(np.power(num_density[timestep,:],2) - np.power(num_den_ana,2))))/(nodes+1)
            os.chdir('Error\PureBreakage\WithTime')
            np.save('L1_errorTimePureAggregationN'+str(nodes)+coagulation_function+'InitialCond.npy', L1_error_time)
            np.save('L2_errorPureAggregationN'+str(nodes)+coagulation_function+'InitialCond.npy', L2_error_time)   
            os.chdir('..\..\..')



