# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:39:01 2020

@author: kanishk

This file contains the coagulation function for aggregation

"""
import numpy as np
# from scipy.integrate import quad

def CoagulationFunction(u,v,type_coagulation_function= None, C6 = 0.5, C7 = 0.5, phiDP = None,
                      muCP = None, muDP = None, sigma = None, rhoCP = None,
                      rhoDP = None,DissipationRate= None, D = None, We = None):
    if type_coagulation_function == "ConstantUnity":
        if isinstance(u, np.ndarray) == False:
            return 1
        else:
            return np.ones(len(u))
    elif type_coagulation_function == "Sum":
        return u+v
    elif type_coagulation_function == "Product":
        return u*v
    elif type_coagulation_function == "CandT":
        c3 = C6
        c4 = C7
        epsilon = DissipationRate
        mu_c = muCP
        rho_c = rhoCP  # mass density of Continuous phase
        sigma_1 = sigma
        # particle collision frequency
        beta = (c3*np.power(epsilon,1/3)*
                np.power(np.power(u,1/3) + np.power(v,1/3), 2)*
                np.power(np.power(u,2/9) + np.power(v,2/9), 1/2))
        # Coalescence eefficiency
        cf_lambda = np.exp(-((c4*mu_c*rho_c*epsilon)/(np.power(sigma_1,2)))*
                           np.power((np.power(u*v,1/3))/(np.power(u,1/3)+np.power(v,1/3)),4))
        return beta*cf_lambda
    elif type_coagulation_function == 'SolsvikValidationcase':
        beta = (C6*np.power(DissipationRate,1/3)*np.square(
            np.power(u,1/3) + np.power(v,1/3))*np.sqrt(
                np.power(u,2/9) + np.power(v,2/9)))
        k1 = (C7*muCP*rhoCP*DissipationRate)/(np.square(sigma))
        k2 = np.power(np.power(u*v,1/3)/(np.power(u,1/3)+ np.power(v,1/3)),4)
        return beta* np.exp(-k1*k2)
# CoagulationFunction(1e-8,1,"CandT")







