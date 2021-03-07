# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 12:22:50 2021

@author: kanishk 
"""
from AnalyticalSolution import AnalyticalSolution
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os as os
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker
import numpy as np

sns.set_theme(context='paper', 
                  style='ticks', 
                  font='Times New Roman', font_scale=2)
x = np.logspace(-8,0,1000)
final_time = 1

fig, axs = plt.subplots(ncols = 2, nrows = 2, figsize = (10,10), sharex = True)
for ax, final_time in zip(axs.reshape(-1), [1,10,100,1000]):
    print(ax,final_time)
    num_den = AnalyticalSolution(x,final_time).BinaryBreakageLinearSelection()
    ax.plot(x, num_den, linewidth = 2, color = 'black', linestyle = '--')
    ax.set_xlim(1e-8,1)
    ax.set_ylim(0,)
    ax.set_title('t = {} s'.format(final_time))
    ax.set_xscale('log')

path = 'G:\My Drive\Research\Popullation Balance Equation Solver\Images\AnalyticalSolutuionImages'
plt.savefig(os.path.join(path, 'MonodisperseLinearSelectionBinaryBreakage.jpg'),
                         dpi = 300, bbox_inches = 'tight')


sns.set_theme(context='paper', 
                  style='ticks', 
                  font='Times New Roman', font_scale=2)
x = np.logspace(-8,0,1000)
final_time = 1

fig, axs = plt.subplots(ncols = 2, nrows = 2, figsize = (10,10), sharex = True)
for ax, final_time in zip(axs.reshape(-1), [1,10,100,1000]):
    print(ax,final_time)
    num_den = AnalyticalSolution(x,final_time).BinaryBreakageLinearSelectionExpInitialCondition()
    ax.plot(x, num_den, linewidth = 2, color = 'black', linestyle = '--')
    ax.set_xlim(1e-8,1)
    ax.set_ylim(0,)
    ax.set_title('t = {} s'.format(final_time))
    ax.set_xscale('log')

path = 'G:\My Drive\Research\Popullation Balance Equation Solver\Images\AnalyticalSolutuionImages'
plt.savefig(os.path.join(path, 'ExpConditionLinearSelectionBinaryBreakage.jpg'),
                         dpi = 300, bbox_inches = 'tight')


fig, axs = plt.subplots(ncols = 2, nrows = 2, figsize = (10,10), sharex = True)
for ax, final_time in zip(axs.reshape(-1), [1,10,100,1000]):
    print(ax,final_time)
    num_den = AnalyticalSolution(x,final_time).BinaryBreakageSquareSelectionExpInitialCondition()
    ax.plot(x, num_den, linewidth = 2, color = 'black', linestyle = '--')
    ax.set_xlim(1e-8,1)
    ax.set_ylim(0,)
    ax.set_title('t = {} s'.format(final_time))
    ax.set_xscale('log')

path = 'G:\My Drive\Research\Popullation Balance Equation Solver\Images\AnalyticalSolutuionImages'
plt.savefig(os.path.join(path, 'ExpInitialDistSquareSelectionBinaryBreakage.jpg'),
                         dpi = 300, bbox_inches = 'tight')


sns.set_theme(context='paper', 
                  style='ticks', 
                  font='Times New Roman', font_scale=2)
x = np.logspace(-8,0,1000)
final_time = 1

fig, axs = plt.subplots(ncols = 2, nrows = 2, figsize = (10,10), sharex = True)
for ax, final_time in zip(axs.reshape(-1), [1,10,100,1000]):
    print(ax,final_time)
    num_den = AnalyticalSolution(x,final_time).BinaryBreakageSquareSelection()
    ax.plot(x, num_den, linewidth = 2, color = 'black', linestyle = '--')
    ax.set_xlim(1e-8,1)
    ax.set_ylim(0,)
    ax.set_title('t = {} s'.format(final_time))
    ax.set_xscale('log')

path = 'G:\My Drive\Research\Popullation Balance Equation Solver\Images\AnalyticalSolutuionImages'
plt.savefig(os.path.join(path, 'MonoDisperseBinarySquareSelectionBinaryBreakage.jpg'),
                         dpi = 300, bbox_inches = 'tight')


















