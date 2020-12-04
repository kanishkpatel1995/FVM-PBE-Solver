# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:32:28 2020

@author: kanishk
The file contains the possible breakage functions that can be used. i.e
Daughter Particle Distribution Function 
"""
import numpy as np

def BreakageFunction(u,v,type_of_breakage_function = None):
    if type_of_breakage_function == None or 'BinaryBreakage':
        return 2/v
    elif type_of_breakage_function == 'CandT': # Coulaloglou and Tavlarides
        nu = 2 # number of particles produced upon breakage 
        return nu*(2.4/v)*np.exp(-(4.5*np.power(2*u - v,2))/(v*v))
        
        