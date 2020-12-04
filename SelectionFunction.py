# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:39:28 2020

@author: kanishk

The file contains possible selection functions.
"""

import numpy as np

def SelectionFunction(x,type_s = None):
    if type_s == 'Linear' or type_s == None:
        return x
    elif type_s == 'Squared':
        return x*x
    elif type_s == 'CandT': # Coulaloglou and Tavlarides
        c1 = 0
        c2 = 0
        epsilon = 0.35 # turbulent dissipation rate in W/kg
        rho_d = 860 # mass density of dispersed phase
        sigma_1 = 5.5e-3 # interfacial tension 
        k = (c1*((np.power(epsilon,1/3))/(np.power(x,2/9)))*
             (np.power(np.pi/6,2/9))*
             np.exp(-(c2*sigma_1)/(rho_d*np.power(epsilon,2/3)*np.power(x,5/9)*np.power(np.pi/6,5/9))))
        return k
