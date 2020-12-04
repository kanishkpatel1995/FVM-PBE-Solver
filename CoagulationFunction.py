# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:39:01 2020

@author: kanishk

This file contains the coagulation function for aggregation

"""
import numpy as np
# from scipy.integrate import quad

def CoagulationFunction(u,v,type_coagulation_function):
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
        c3 = 0.053
        c4 = 12
        epsilon = 0.35
        mu_c = 4e-4
        rho_c = 980  # mass density of Continuous phase
        sigma_1 = 0.55e-2
        # particle collision frequency
        beta = (c3*np.power(epsilon,1/3)*
                np.power(np.power(u,1/3) + np.power(v,1/3), 2)*
                np.power(np.power(u,2/9) + np.power(v,2/9), 1/2))
        # Coalescence eefficiency
        cf_lambda = np.exp(-((c4*mu_c*rho_c*epsilon)/(np.power(sigma_1,2)))*
                           np.power((np.power(u*v,1/3))/(np.power(u,1/3)+np.power(v,1/3)),4))
        return beta*cf_lambda

CoagulationFunction(1e-8,1,"CandT")







