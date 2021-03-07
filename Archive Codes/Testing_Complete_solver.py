# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 13:35:20 2020

@author: kanishk
"""

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import time
### managing fonts for plots 

plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'
##############################

import numpy as np
import pandas as pd
from MainSolver import PBEBrkAggSolver
from scipy.integrate import solve_ivp
from scipy.integrate import Radau
from scipy.integrate import odeint
from scipy.integrate import quad
from time import perf_counter 
from Grid_X import GridX
from TemporalGrid import TemporalGrid
from BreakageFunction import BreakageFunction
from SelectionFunction import SelectionFunction
from CoagulationFunction import CoagulationFunction
from InitialCondition import InitialConditionNumberDensity
from OptimisedMainSolver import PBEBrkAggSolver,APureAggregation, APureBreakage

minimimum_particle_size = 1e-4
# maximum particle size 
maximum_particle_size = 1e4
#initial time

no_of_nodes = 6
t_min = 0 
# Final Time
t_max = 5
# Save solution after Delta_t time 
delta_t = 0.5

for temporal_solver in ['BDF', 'LSODA','RK45', 'RK23']:
    for no_of_nodes in [4,8,16,32,64,128]:
        g_solve,num_density = PBEBrkAggSolver(
            minimimum_particle_size, 
                        maximum_particle_size,
                        no_of_nodes,
                        t_min, 
                        t_max,
                        delta_t,
                        type_of_selection_function = 'CandT',
                        type_of_breakage_function = 'CandT',
                        type_coagulation_function = 'CandT', 
                        type_of_initial_condition = 'LND', 
                        temporal_solver = temporal_solver,
                        Status_Update = None,
                        save_data = True)

# ax1.ticklabel_format(axis = 'y', style = 'sci')
















