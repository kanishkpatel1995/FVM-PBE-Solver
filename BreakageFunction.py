# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:32:28 2020

@author: kanishk
The file contains the possible breakage functions that can be used. i.e
Daughter Particle Distribution Function 
"""
import numpy as np
from scipy.special import gamma
from SelectionFunction import SelectionFunction
def BreakageFunction(u,v,type_of_breakage_function = None, C1=2, C2 = 2, phiDP = None,
                      muCP = None, muDP = None, sigma = None, rhoCP = None,
                      rhoDP = None,DissipationRate= None, D = None, We = None):
    di = np.power((3*u)/(4*np.pi),1/3)
    dj = np.power((3*v)/(4*np.pi),1/3)
    if type_of_breakage_function == 'BinaryBreakage':
        return 2/v
    elif type_of_breakage_function == 'CandT' or type_of_breakage_function == None : # Coulaloglou and Tavlarides 
    # parameters a and b = 2 for the caase of binary breakage.
        return ((1/dj)*(gamma(C1+C2)/(gamma(C1)*gamma(C2))) * 
                np.power(di/dj,(1/3)*(C1-1))*
                np.power(1 - (di/dj), C2-1))
    
    elif type_of_breakage_function == 'alopeaus':
        # a = 0.5 # in this case a = geometric standard deviation
        #its value equal to 0.5 is assumed for dilute systems from paper of 
        #3 This kernel can also be used for densed systems i.e high 
        #volume fraction
        #becker et. al. 2011
        return ((1/(u*np.sqrt(2*np.pi*np.power(C1,2))))*
                np.exp(-np.power(np.log(u) - np.log(v/2) + C1*C1,2)/(2*C1*C1)))
    elif type_of_breakage_function == 'BandP':
        # a = 3
        ## this assumes normal distribution of particles after 
        ### Breakage
        return ((1/(u*np.sqrt(2*np.pi)))*
                np.exp(-np.power((u - 0.5*u)*2*C1,2)/(2*v*v)))
    elif type_of_breakage_function == 'SandR':
        # a = 0.0577## s4 original value from the paper of Becker et.al. 2011
        # b = 0.558  # s5 parameter original value from the paper of Becker et.al. 2011
        alpha = np.exp(C1*np.log(muCP/muDP) - C2)
        ratio = (SelectionFunction(v,type_s = 'SandR', C3 = None, C4 = None, C5= None, phiDP = phiDP,
                      muCP = muCP, muDP = muDP, sigma = sigma, rhoCP = rhoCP,
                      rhoDP = rhoDP,DissipationRate= DissipationRate, D = D, We = We)/
                 SelectionFunction(u,type_s = 'SandR', C3 = None, C4 = None, C5= None, phiDP = phiDP,
                      muCP = muCP, muDP = muDP, sigma = sigma, rhoCP = rhoCP,
                      rhoDP = rhoDP,DissipationRate= DissipationRate, D = D, We = We))
        nume = np.power(ratio, alpha)
        denom = (1 - 0.25*alpha) + (0.25*alpha*
                                    np.power(ratio, 4))
        return nume/denom
    
    elif type_of_breakage_function == 'SolsvikValidationcase':
        # print('Applied Breakage func: {}'.format(type_of_breakage_function))
        k = (4.8/v) * np.exp(-(4.5 *np.square(2*u - v))/(np.square(v)))
        # print(u,v, k)
        return k
    
    # elif type_of_breakage_function == 'newone':
    #     return we
    
        
        
        
    
                    
        
        