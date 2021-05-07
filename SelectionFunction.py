# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:39:28 2020

@author: kanishk

The file contains possible selection functions/breakup frequency.
"""

import numpy as np
from scipy.special import polygamma
import scipy.integrate as integrate
from scipy.special import erf

def SelectionFunction(x,type_s = None, C3 = 4.87e-3, C4 = 0.0552, C5= None, phiDP = None,
                      muCP = None, muDP = None, sigma = None, rhoCP = None,
                      rhoDP = None,DissipationRate= None, D = None, We = None): 
    d = np.power((3*x)/(4*np.pi),1/3) #diameter of the particle
    if type_s == 'Linear':
        return x
    elif type_s == 'Squared':
        return x*x
    elif type_s == 'CandT': # Coulaloglou and Tavlarides 1977
        ##--This model is based on the fact that the turbulent kinetic energy of
        #the drop is greater than the critical value.
        #--It assumes a normal distribution of turbulent kinetic energy within
        # isotropic turbulence field.
        # For the case of gas-liquid mixture the density rho_d should be 
        #replaced with density rho_c  
        # developed specifically for liquid-liquid dispersions
        
         # interfacial tension 
        k = (C3 * np.power(d,-2/3)*np.power(DissipationRate,1/3)*np.power(1+phiDP,-1)*
             np.exp((C4*sigma)/
                    (rhoDP *np.power(DissipationRate,2/3)*np.power(d,5/3))))
        return k
    elif type_s == 'BandP':
        if C3 == None or C4 == None or C5 == None:
            C3 = 0.56 # B1 parameter from Becker et.al. 2011
            C4 = 1/C3  ### D is the Diameter of the propeller/mixer
        Li = 0.05 * D
        f1 = (0.16*muDP)/(rhoCP*np.power(DissipationRate,1/3)*np.power(Li,1/3)*d)
        f2 = (0.35*sigma)/(rhoCP*np.power(DissipationRate,2/3)*np.power(Li,2/3)*d)
        f3 = np.log(Li/d)
        alpha_d = 3*np.log(2*np.power(f1 +np.sqrt(f1*f1 + f2),-1))*(1/f3)
        def integral(alpha):
            f_alpha = 1 - (np.power(alpha - 1.117,2)/0.468)
            return np.power(d/Li, (1/3)*(alpha + 2 - 3*f_alpha))
        integral_val = integrate.quad(integral, 0.12, alpha_d)
        return (0.0035*C3*np.sqrt(np.log(Li/d))*
                (np.power(C4*DissipationRate,1/3)/np.power(d,2/3))*integral_val[0])
    elif type_s == 'alopeaus':
        if C3 == None or C4 == None or C5 == None:
            C3 = 0.986
            C4 = 0.892e-3
            C5 = 0.2
        f1 = C4*((sigma)*np.power(rhoCP*np.power(DissipationRate,2/3)*
                                 np.power(d,5/3),-1))
        f2 = C5*((muDP)*np.power(np.sqrt(rhoCP*rhoDP)*np.power(DissipationRate,1/3)*
                                 np.power(d,4/3),-1))
        return C3*np.power(DissipationRate, 1/3)*erf(np.sqrt(f1+f2))
    
    elif type_s == 'SandR':
        if C3 == None or C4 == None or C5 == None:
            C3 = 0.422
            C4 = 0.247
            C5 = 2.154
        if We == None :
            omega = 100
            We = omega**2 * D**3 *rhoCP * (1/sigma) 
        f1 = C4*np.square(np.log(We*np.power(x/(D*D*D), 5/9) * np.power(muCP/muDP, 0.2)))
        f2 = C5* np.log(We* np.power(x/(D*D*D), 5/9) * np.power(muCP/muDP, 0.2))
        return C3 * np.sqrt(sigma/(rhoCP*x))* np.exp(-f1 + f2)
        
    elif type_s == "Chatzi": ##modification of CandT model proposed by chatzi
        # in 1983
        # He ssumes that the distribution of kinetic energy is according to 
        # Maxwell's law.
        d = np.power((3*x)/(4*np.pi),1/3) #diameter of the particle
        c1 = 0
        c2 = 0
        phi = 0.2 # volume fraction of the dispersed phase
        epsilon = 0.35 # turbulent dissipation rate in W/kg
        rho_d = 860 # mass density of dispersed phase
        sigma = 5.5e-3
        ev = 1.1283791670955126 # never changes, 2/srt(pi)
        k = (c1*np.power(d,-2/3)*np.power(epsilon,1/3)*ev*
        polygamma(1.5,(c2*sigma)/(rho_d *np.power(epsilon,2/3)*np.power(d,5/3))))
    elif type_s == 'SolsvikValidationcase':
        k1 = C3*(np.power(DissipationRate,1/3)/np.power(x,2/9))*np.power(np.pi/6,2/9)
        k2 = np.exp(-(C4*sigma)/
                    (rhoDP*np.power(DissipationRate,2/3)*np.power(x,5/9)*np.power(6/np.pi,5/9)))
        return k1*k2
        
        
        
        
        
        
        
        
        
        
        
        