# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 00:25:39 2020

@author: kanishk

The file contins a class of function that can be used to derive moments


"""
import numpy as np
from Grid_X import GridX
from AnalyticalSolution import AnalyticalSolution


class moments: 
    def analytical_moment_PBE(x_max,x_min,nodes,t_min,t_max,delta_t,
                          moment = 0,
                          which_selection_function = 'Linear',
                          which_coagulation_function = None,
                          which_disrtibution = 'OneLargeParticle'):
 
        x,x_node_boundaries,delta_x = GridX(x_min,x_max,nodes).UniformLogarithmic()
        timesteps = (t_max - t_min)/delta_t
        time = np.linspace(t_min, t_max,int(timesteps)+1)
        moment_ana = np.zeros([len(time),2])
        moment_ana[:,0] = time
        for timestep in range(len(time)):
            if which_coagulation_function == None and which_selection_function != None:
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
            elif which_coagulation_function == 'ConstantUnity' and which_selection_function == None:
                moment_ana[timestep,1] = np.sum(np.power(x,moment)*delta_x*
                    AnalyticalSolution(x,timestep*delta_t,delta_x=delta_x).ConstantUnityCoagulationNormExpInitialCondition())
            elif which_coagulation_function == 'Product' and which_selection_function == None:
                moment_ana[timestep,1] = np.sum(np.power(x,moment)*delta_x*
                    AnalyticalSolution(x,timestep*delta_t,delta_x=delta_x).ProductKernelCoagulationNormExpInitialCondition())
            else:
                print('Mention appropriate Coagulation function and set selection function to None')
        return moment_ana
        