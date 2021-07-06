# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:55:29 2021

@author: kanishk
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from OptimisedMainSolver import PBEBrkAggSolver
from AnalyticalSolution import AnalyticalSolution
# from InitialCondition import InitialConditionNumberDensity
# from AnalyticalSolution import AnalyticalSolution
import os
import seaborn as sns
sns.set_theme(context = 'paper', style = 'ticks', font_scale = 2, palette='bright')
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
minimimum_particle_size = 1e-8
# maximum particle size 
maximum_particle_size = 1
#initial time
plt.figure(figsize = (10,10))
no_of_nodes = 16
t_min = 0 
# Final Time
t_max = 1000
# Save solution after Delta_t time 
delta_t = 1
# for top in ['pAgg', 'pBrk', 'PBrkAgg']:
#         for C3 in [0.001, 0.004, 0.01]:
for C3 in [0.001, 0.002]:
    for C4 in [0.1, 0.2, 0.3]:
        for dissi in np.arange(0,1,0.1):
            g_solve, df = PBEBrkAggSolver(minimimum_particle_size, 
                           maximum_particle_size,
                           no_of_nodes,
                           t_min, 
                           t_max,
                           delta_t,
                           grid_type = 'logspace',
                           r1 = 12, ## grid genration factors
                           r2 = 0.9,
                           vf = 1, ## Volume factor, to scale the droplet volume
                           type_of_problem = 'pBrk', ## Three types possible PureBreakage
                           # i.e = pBrk, Pure Aggregation = pAgg or Breakage and Aggregation combined = BrkAgg
                           type_of_selection_function = 'CandT',
                           type_of_breakage_function = 'CandT',
                           type_coagulation_function = 'CandT', 
                           type_of_initial_condition = 'SLND',
                           mean = 50, std = 15,
                           temporal_solver = 'BDF', 
                           Status_Update = False,
                           save_data = None,
                           C1=2, C2 = 2, ### Constants for Breakage function/Daughter size Distribution
                           C3 = C3, C4 = C4, C5= 2, ### Constants for Selection Function/Breakage frequency
                           C6 = 0.053, C7 = 12, ### Constants for Aggregation function
                           phiDP = 0.1,
                                  muCP = 4e-4, muDP = 0.03, sigma = 5.5e-3, rhoCP = 980,
                                  rhoDP = 860,DissipationRate= dissi, D = 12e-7, We = 0.8)
        
        # plt.plot(df['x'], df['g'], label = r'C3 = {}'.format(C3),
        #          marker = '*')
        

# plt.semilogx(df['x'].dropna(),df['g_initial'].dropna(),'-k', label = 'initial', marker = 'o')
# plt.ylabel(r'$g$')
# plt.xlabel(r'$x$')

## directly from solsvik paper
# n_solvik = pd.read_csv('Validation/Solsvik_Init_Breakage_Dominated_Cases.csv', names = ['x','g'])
# kx,ky = 1e12,1e7
# plt.plot(n_solvik['x']*1, n_solvik['g']*ky,label = 'solsvik Paper')
plt.legend(frameon = False)
path = 'G:\My Drive\Research\Popullation Balance Equation Solver\Images\AllKernelsValidation'
# plt.savefig(os.path.join(path, 'pBrk'+'C3'+str(no_of_nodes)+'FinalTime' +str(t_max)+'alopeaus.jpg'),
#             dpi = 300, bbox_inches = 'tight')

# Using Analytical Solution
# x = np.logspace(-8, 1, 1000)
# k = AnalyticalSolution(x, t_max).BinaryBreakageLinearSelection()
# plt.semilogx(x, k)

# pdf = InitialConditionNumberDensity(df['x'].dropna(),df['DeltaX'].dropna(),no_of_nodes).InitialConditionBasedMeanandStd(50,14.6)
# plt.semilogx(df['x'].dropna(),pdf*df['x'].dropna())
### Evaluating D43 and D32. 
# d = np.power((3*df['x'])/(4*np.pi),1/3)
# mu10 = np.sum(np.power(df['x'],1)*df['NumberDensity'])/np.sum(np.power(df['x'],2)*df['NumberDensity'])
# D32 = np.sum(np.power(d,3)*df['NumberDensity']*df['DeltaX'])/np.sum(np.power(d,2)*df['NumberDensity']*df['DeltaX'])
# D43 = np.sum(np.power(d,4)*df['NumberDensity']*df['DeltaX'])/np.sum(np.power(d,3)*df['NumberDensity']*df['DeltaX'])
# initD32 = np.sum(np.power(d,3)*n*df['DeltaX'])/np.sum(np.power(d,2)*n*df['DeltaX'])
# initD43 = np.sum(np.power(d,4)*n*df['DeltaX'])/np.sum(np.power(d,3)*n*df['DeltaX'])
# g = InitialConditionNumberDensity(df['x'],df['DeltaX']).Solsvik_LogNormalDistribution()
# plt.plot(df['x']*kx,g)

#####
# from scipy.stats import lognorm
# mu, sigma = 50, 14.6 #in micrometers
# lnsigma = np.sqrt(np.log( np.square(sigma) / np.square(mu) + 1) );
# lnmu = np.log(mu/np.exp(0.5*np.square(lnsigma)));
# ## random will give me approximate droplet sizes
# droplet_counts = np.random.lognormal(lnmu,lnsigma,3000)*1e-6 # as my droplet size is in microns
# # volume_counts = (4/3)*np.pi*np.power(droplet_counts,3)
# # coverting to volume of particles

# shape,loc,scale = lognorm.fit(droplet_counts)
# d = np.logspace(np.log10(droplet_counts.min()),np.log10(droplet_counts.max()),24)
# pdf = lognorm.pdf(d, shape, loc, scale)
# volume = (4/3)*np.pi*np.power(d,3)
# x = np.logspace(-13,-12, 24)
# plt.plot(x, pdf, 'r')

# plt.xscale('log')
# plt.hist(droplet_counts,200, density = True)









