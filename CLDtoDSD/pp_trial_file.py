# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:19:11 2020

@author: kanishk
"""
# Extracting DSD from CLD

import matplotlib.pyplot as plt
from CLDfromExpData import ExperimentalCLD
from CLDtoDSD import CLDtoDSDMethod1,CLDtoDSDMethod2


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

fig, ax1 = plt.subplots(1,1)
fig.set_figheight(8)
fig.set_figwidth(17)
marker = ['b--','g--','--r','--k']
marker_cld = ['-b','-g','-r','-k']
i = 0
for time in [int(800),int(1600),int(2400),'Last Time']:
    CLD = ExperimentalCLD('Experiment 2020-11-27 10-34 Default.csv',time)
    DSD = CLDtoDSDMethod2('Experiment 2020-11-27 10-34 Default.csv',time)
    if isinstance(time, int):
        ax1.semilogx(DSD['Diameter'].values , DSD['Counts'].values,marker[i], label = f'DSD t = {time}s')
        ax1.semilogx(CLD.iloc[:,0].values , CLD.iloc[:,1].values,marker_cld[i],label = f'CLD t = {time}s')
    else:
        ax1.semilogx(DSD['Diameter'].values , DSD['Counts'].values,marker[i], label = 'DSD at Steady state')
        ax1.semilogx(CLD.iloc[:,0].values , CLD.iloc[:,1].values,marker_cld[i],label = 'CLD at Steady state')
    
    i = i+1
ax1.tick_params(axis ='both', labelsize = 24,grid_alpha=0.5)
ax1.set_xlabel(r'$x(\mu m)$',fontsize=32)
ax1.set_ylabel(r'$n$',fontsize=32, rotation = 'horizontal')
ax1.xaxis.set_label_coords(0.55,-0.06)
ax1.yaxis.set_label_coords(-0.05,0.5)
ax1.yaxis.offsetText.set_fontsize(24)
ax1.set_title(r'Experimental Data', fontsize = 32)
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))
ax1.set_xlim(1,1000)
ax1.set_ylim(0,)
ax1.grid()
plt.legend(fontsize = 24, frameon = False)
fig.savefig('Comparision_num_ana_pure_aggregation', dpi=300,
        transparent=True, bbox_inches='tight')









