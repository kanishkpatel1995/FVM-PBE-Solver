# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:45:13 2020

@author: kanishk

Optimisd Main Solver

"""

###Imported modules - online 
import numpy as np
import pandas as pd
import time
import json
from tqdm import tqdm
import time
import os as os
from scipy.integrate import solve_ivp
from scipy.integrate import quad
from time import perf_counter 
from Grid_X import GridX
from TemporalGrid import TemporalGrid
from BreakageFunction import BreakageFunction
from SelectionFunction import SelectionFunction
from CoagulationFunction import CoagulationFunction
from InitialCondition import InitialConditionNumberDensity
### timestmp
timestr = time.strftime("%Y%m%d%H%M%S")

# class APureBreakage:
#         def __init__(self, x, x_node_boundaries, delta_x, type_of_selection_function, 
#                       type_of_breakage_function):
#             self.x = x
#             self.x_node_boundaries = x_node_boundaries
#             self.delta_x = delta_x
#             self.type_s = type_of_selection_function
#             self.type_of_breakage_function = type_of_breakage_function
#         def derec(self,k,i):
#             d_k_i = 0
#             inner_integral = 0
#             if i>=0:
#                 for j in range(i+1): 
#                     inner_integral = inner_integral + self.x[j]*BreakageFunction(self.x[j],
#                                                                                   self.x[k],
#                                                                                   self.type_of_breakage_function)*self.delta_x[j]
#             else :
#                 inner_integral = 0
#             d_k_i = SelectionFunction(self.x[k],self.type_s)*(1/self.x[k])*self.delta_x[k]*inner_integral
#             return d_k_i
    
#         def A(self):
#             A = np.zeros((len(self.x),len(self.x)))
#             for i in range(len(self.x)):
#                 #print(i) 
#                 A[i,i] = -(1/self.delta_x[i])*self.derec(i,i-1) # diagonal elements
#                 for k in range(i+1,len(self.x)):
#                     A[i,k] = -(1/self.delta_x[i]) * (self.derec( k, i-1) - 
#                                               self.derec( k, i))
#             return A

# class APureAggregation:
#     ### Getting spatial matrix A pure aggregation ###
#     # we need to add g array from previous timestep as the solution is now dependent on g 
#     def __init__(self, x, x_node_boundaries, delta_x,g,type_coagulation_function):
#         self.x = x
#         self.x_node_boundaries = x_node_boundaries
#         self.delta_x = delta_x
#         self.g = g # it is the mass density matrix
#         self.type_coagulation_function = type_coagulation_function
#     def derec(self,i,k):
#         ### Evaluating alpha
#         x = self.x
#         x_node_boundaries = self.x_node_boundaries
#         delta_x = self.delta_x
#         g = self.g
#         value = x_node_boundaries[i+1] - x[k]
#         array = np.asarray(x_node_boundaries)
#         idx = (np.abs(array - value)).argmin()
#         if value > array[idx]:
#             alpha = idx + 1
#         elif value == 0:
#             alpha = idx + 1
#             ## on the last cell the node boundary value is same as the node value
#             #henceforth value = x_node_boundaries[i+1] - x[k] came out to be zero.
#             ## For an integral with 1/x a zero value will result in sigularity. 
#             ## to avoid this the value of 0 is made equivalent to the size of the smallest particle.
#             value = x_node_boundaries[0] ## making value equal to smallest particle
#             # value = 1e-12
#         else:
#             alpha = idx
#         # print(x[alpha:len(x)],alpha)
#         rhs_term1 = np.sum(g[alpha:]*(1/x[alpha:])*
#                            CoagulationFunction(x[alpha:], x[k], self.type_coagulation_function)
#                            *delta_x[alpha:])
#         # print(CoagulationFunction(x[alpha:], x[k], self.type_coagulation_function))
#         def integrand(x,a):
#             return (1/x)*CoagulationFunction(x,a,self.type_coagulation_function)
#         # using sci py quad function for integration 
#         a = x[k]
#         I = quad(integrand, value, x_node_boundaries[alpha],args=(a))
#         rhs_term2 = g[alpha - 1] * I[0]
        
#         # # using mid point rule instead of integrand
#         # x_midpoint = (value + x_node_boundaries[alpha])/2
#         # rhs_term2 = integrand(x_midpoint, x[k])*(x_node_boundaries[alpha] - value)
#         # print(I,alpha,value,x_node_boundaries[alpha])
#         return rhs_term1 +rhs_term2
         

#     def A(self):
#         A = np.zeros((len(self.x),len(self.x)))
#         J = np.zeros((len(self.x),len(self.x)))
#         for i in range(len(self.x)):
#             # print(i)
#             for j in range(i+1):
#                 J[i,j] = self.delta_x[j] * self.derec(i,j)
#         for i in range(len(A[:,0])):
#             if i == 0:
#                 A[i,:] = (-1/self.delta_x[i]) * J[i,:]
#             else:
#                 A[i,:] = (-1/self.delta_x[i]) * (J[i,:] - J[i-1,:])
#         # print(A)
#         return A
    



def PBEBrkAggSolver(minimimum_particle_size, 
               maximum_particle_size,
               no_of_nodes,
               t_min, 
               t_max,
               delta_t,
               grid_type = 'owngrid',
               r1 = 0.75, ## grid genration factors
               r2 = 0.75,
               vf = 1, ### Volume factor, multiplies with x to make solution physical
               type_of_problem = None, ## Three types possible PureBreakage
               # i.e = pBrk, Pure Aggregation = pAgg or Breakage and Aggregation combined = BrkAgg
               type_of_selection_function = None,
               type_of_breakage_function = None,
               type_coagulation_function = None, 
               type_of_initial_condition = None, 
               mean = None,
               std = None,
               temporal_solver = None, 
               Status_Update = None,
               save_data = None,
               C1=2, C2 = 2, ### Constants for Breakage function/Daughter size Distribution
               C3 = 4.87e-3, C4 = 0.0552, C5= None, ### Constants for Selection Function/Breakage frequency
               C6 = None, C7 = None, ### Constants for Aggregation function
               phiDP = None,
                      muCP = None, muDP = None, sigma = None, rhoCP = None,
                      rhoDP = None,DissipationRate= None, D = None, We = None):
    
    """ Solves popullation balance equation with the choice of aggregation and 
    brekage kernel"""
    
    print("Creating a uniform logarithmic grid")
    
    grid_gen_start = perf_counter()
    #### Selecting the type of grid and parameters
    if grid_type == 'Uniform':
        x, x_node_boundaries,delta_x = GridX(minimimum_particle_size,
                                             maximum_particle_size,
                                             no_of_nodes).Uniform()
    elif grid_type == 'logspace':
        x, x_node_boundaries,delta_x = GridX(minimimum_particle_size,
                                             maximum_particle_size,
                                             no_of_nodes).UniformLogarithmic()
    elif grid_type == 'owngrid':
        x, x_node_boundaries,delta_x = GridX(minimimum_particle_size,
                                             maximum_particle_size,
                                             no_of_nodes).owngrid(r1,r2)
    elif grid_type == 'bigeometric':
        x, x_node_boundaries,delta_x = GridX(minimimum_particle_size,
                                             maximum_particle_size,
                                             no_of_nodes).bigeometric(r1,r2)
    grid_gen_end = perf_counter()
    t = grid_gen_end-grid_gen_start
    print(f"Grid Generation ended, Time Taken to generate grid = {t}")
    
    
    print("Selecting initial condition")
    sel_ini_cond_start = perf_counter() ### Calculating time for implementing initial condition
    
    if type_of_initial_condition == 'OneLargeParticle':
        print('Initial Condition Applied, ExponentialDistribution')
        n = InitialConditionNumberDensity(x,delta_x).FilbetAndLaurencotExponentialDistributionForProductKernel()
        g0 = n*x
        print(g0)
    elif type_of_initial_condition == 'SLND':
        print('Initial Condition Applied, Solsvik Lognormal disribution')
        g0 = InitialConditionNumberDensity(x, delta_x).Solsvik_LogNormalDistribution()
    elif type_of_initial_condition == 'mimicReality':
        if mean == None or std == None:
            mean = 50 ### mean particle dimeter size 
            std = 14.6
        g0 = InitialConditionNumberDensity(x, delta_x, no_of_nodes).InitialConditionBasedMeanandStd(mean,std)
    
    # g0[g0<1e-18] = 1e-18
    # Initial particle mass density
    ##################################################################################
    # Evaluating initial mean droplet diameters
    d = np.power((3*(x*vf))/(4*np.pi),1/3) 
    InitialD32 = np.sum(np.power(d,3)*(g0/(x))*delta_x)/np.sum(np.power(d,2)*(g0/(x))*delta_x)
    InitialD43 = np.sum(np.power(d,4)*(g0/(x))*delta_x)/np.sum(np.power(d,3)*(g0/(x))*delta_x)
    print(g0)
    
    
    sel_ini_cond_end = perf_counter() ### Initial condition timer ends
    
    print(f"Applying intial condition took {sel_ini_cond_end-sel_ini_cond_start} seconds")
#############################################################################################   
    ###### Pure Breakage class adding the parameters #############
    class APureBreakage:
        def __init__(self, x, x_node_boundaries, delta_x, type_of_selection_function, 
                     type_of_breakage_function):
            self.x = x
            self.x_node_boundaries = x_node_boundaries
            self.delta_x = delta_x
            self.type_s = type_of_selection_function
            self.type_of_breakage_function = type_of_breakage_function
        def derec(self,k,i):
            d_k_i = 0
            inner_integral = 0
            if i>=0:
                for j in range(i+1): 
                    inner_integral = inner_integral + (self.x[j]*
                                                       BreakageFunction(self.x[j],
                                                                        self.x[k],
                    type_of_breakage_function = self.type_of_breakage_function,
                    C1 = C1,
                    C2 = C2,
                    phiDP = phiDP,
                    muCP = muCP, 
                    muDP = muDP, 
                    sigma = sigma, 
                    rhoCP = rhoCP,
                    rhoDP = rhoDP,DissipationRate= DissipationRate, D = D, We = We
                                                                        )*self.delta_x[j])
                    # print('type_of_breakage_function = {} a = {},b = {},phiDP = {},muCP = {}, muDP = {}, sigma = {}'.format(type_of_breakage_function,C1,C2,phiDP,muCP,muDP,sigma))
                    # print(self.type_s,self.type_of_breakage_function )
            else :
                inner_integral = 0
            d_k_i = SelectionFunction(self.x[k],type_s = self.type_s, 
                                      C3 = C3, C4 = C4, C5= C5,
                                       phiDP = phiDP,
                                        muCP = muCP, 
                                        muDP = muDP, 
                                        sigma = sigma, 
                                        rhoCP = rhoCP,
                                        rhoDP = rhoDP,
                                        DissipationRate= DissipationRate, 
                                        D = D, 
                                        We = We
                                      )*(1/self.x[k])*self.delta_x[k]*inner_integral
            return d_k_i
    
        def A(self):
            A = np.zeros((len(self.x),len(self.x)))
            for i in range(len(self.x)):
                #print(i) 
                A[i,i] = -(1/self.delta_x[i])*self.derec(i,i-1) # diagonal elements
                for k in range(i+1,len(self.x)):
                    A[i,k] = -(1/self.delta_x[i]) * (self.derec( k, i-1) - 
                                              self.derec( k, i))
            return A
###############################################################################################  
##### Aggregation cLASS #############################################################
    class APureAggregation:
    ### Getting spatial matrix A pure aggregation ###
    # we need to add g array from previous timestep as the solution is now dependent on g 
        def __init__(self, x, x_node_boundaries, delta_x,g,type_coagulation_function):
            self.x = x
            self.x_node_boundaries = x_node_boundaries
            self.delta_x = delta_x
            self.g = g # it is the mass density matrix
            self.type_coagulation_function = type_coagulation_function
        def derec(self,i,k):
            ### Evaluating alpha
            x = self.x
            x_node_boundaries = self.x_node_boundaries
            delta_x = self.delta_x
            g = self.g
            value = x_node_boundaries[i+1] - x[k]
            array = np.asarray(x_node_boundaries)
            idx = (np.abs(array - value)).argmin()
            if value > array[idx]:
                alpha = idx + 1
            elif value == 0:
                alpha = idx + 1
                ## on the last cell the node boundary value is same as the node value
                #henceforth value = x_node_boundaries[i+1] - x[k] came out to be zero.
                ## For an integral with 1/x a zero value will result in sigularity. 
                ## to avoid this the value of 0 is made equivalent to the size of the smallest particle.
                value = x_node_boundaries[0] ## making value equal to smallest particle
                # value = 1e-12
            else:
                alpha = idx
            # print(x[alpha:len(x)],alpha)
            rhs_term1 = np.sum(g[alpha:]*(1/x[alpha:])*
                               CoagulationFunction(x[alpha:], x[k],
                            type_coagulation_function= self.type_coagulation_function,
                        C6 = C6, 
                        C7 = C7,
                        phiDP = phiDP,
                                        muCP = muCP, 
                                        muDP = muDP, 
                                        sigma = sigma, 
                                        rhoCP = rhoCP,
                                        rhoDP = rhoDP,
                                        DissipationRate= DissipationRate, 
                                        D = D, 
                                        We = We)
                               *delta_x[alpha:])
            # print('g = {},h = {},phiDP = {},muCP = {}, muDP = {}, sigma = {}'.format(C6,C7,phiDP,muCP,muDP,sigma))

            # print(CoagulationFunction(x[alpha:], x[k], self.type_coagulation_function))
            def integrand(x,a):
                return (1/x)*CoagulationFunction(x,a,
                        type_coagulation_function= self.type_coagulation_function,
                        C6 = C6, 
                        C7 = C7,
                        phiDP = phiDP,
                                        muCP = muCP, 
                                        muDP = muDP, 
                                        sigma = sigma, 
                                        rhoCP = rhoCP,
                                        rhoDP = rhoDP,
                                        DissipationRate= DissipationRate, 
                                        D = D, 
                                        We = We)
            # using sci py quad function for integration 
            a = x[k]
            I = quad(integrand, value, x_node_boundaries[alpha],args=(a))
            rhs_term2 = g[alpha - 1] * I[0]
            
            # # using mid point rule instead of integrand
            # x_midpoint = (value + x_node_boundaries[alpha])/2
            # rhs_term2 = integrand(x_midpoint, x[k])*(x_node_boundaries[alpha] - value)
            # print(I,alpha,value,x_node_boundaries[alpha])
            return rhs_term1 +rhs_term2
         

        def A(self):
            A = np.zeros((len(self.x),len(self.x)))
            J = np.zeros((len(self.x),len(self.x)))
            for i in range(len(self.x)):
                # print(i)
                for j in range(i+1):
                    J[i,j] = self.delta_x[j] * self.derec(i,j)
            for i in range(len(A[:,0])):
                if i == 0:
                    A[i,:] = (-1/self.delta_x[i]) * J[i,:]
                else:
                    A[i,:] = (-1/self.delta_x[i]) * (J[i,:] - J[i-1,:])
            # print(A)
            return A
############################################Classess End###################

    """Linear Algebric solvers """
    ## Implementing EulerExplicitSolver as A mtrix depends on time 
    ## The solver evaluates the particle mass density.
    
    t_solve = TemporalGrid(t_min,t_max,delta_t).Uniform()

    if Status_Update == True:
        
        def pbe_ode(t,y,pbar,state):
            # state is a list containing last updated time t:
            # state = [last_t, dt]
            # I used a list because its values can be carried between function
            # calls throughout the ODE integration
            last_t, dt = state
            
            # let's subdivide t_span into 1000 parts
            # call update(n) here where n = (t - last_t) / dt
            time.sleep(0.1)
            n = int((t - last_t)/dt)
            pbar.update(n)
            
            # we need this to take into account that n is a rounded number.
            state[0] = last_t + dt * n
            
            #### Selecting the type of problem
            if type_of_problem == 'pBrk':
                dy_dt = np.matmul((APureBreakage(x, x_node_boundaries,delta_x,
                                    type_of_selection_function,type_of_breakage_function).A()),y)
                return dy_dt
            elif type_of_problem == 'pAgg':
                dy_dt = np.matmul((APureAggregation(x, x_node_boundaries,delta_x,y,
                                                    type_coagulation_function).A()),y)
                return dy_dt
            else:
                dy_dt = np.matmul((APureAggregation(x, x_node_boundaries,delta_x,y,type_coagulation_function).A() + 
                              APureBreakage(x, x_node_boundaries,delta_x,
                                type_of_selection_function,type_of_breakage_function).A()),y)
                return dy_dt
    
        t_starting_solve_ivp_temporal_solver = perf_counter()
    
        with tqdm(total=1000, unit="‰") as pbar:
            g_solve = solve_ivp(pbe_ode, [t_min,t_max],g0,
                                method = temporal_solver,
                                dense_output=True,
                                t_eval = t_solve, atol = 1e-3,
                                args=[pbar, [t_min, (t_max-t_min)/1000]])
        # g_solve = solve_ivp(pbe_ode, [t_min,t_max],g0,
        #                         method = temporal_solver,
        #                         dense_output=True,
        #                         t_eval = t_solve, atol = 1e-3)
        
        t_ending_solve_ivp_temporal_solver = perf_counter()
        
    else:
        if type_of_problem == 'pBrk':
            def pbe_ode(t,y):
                    A_mat = APureBreakage(x, x_node_boundaries,delta_x,
                                        type_of_selection_function,type_of_breakage_function).A()
                    A_mat[A_mat<1e-18] = 1e-18
                    dy_dt = np.matmul(A_mat,y)
                    return dy_dt
        elif type_of_problem == 'pAgg':
            def pbe_ode(t,y):
                dy_dt = np.matmul((APureAggregation(x, x_node_boundaries,delta_x,y,
                                                    type_coagulation_function).A()),y)
                return dy_dt
        else:
            def pbe_ode(t,y):
                dy_dt = np.matmul((APureAggregation(x, x_node_boundaries,delta_x,y,type_coagulation_function).A() + 
                              APureBreakage(x, x_node_boundaries,delta_x,
                                type_of_selection_function,type_of_breakage_function).A()),y)
                return dy_dt
        # print(type_of_selection_function,type_of_breakage_function)
        t_starting_solve_ivp_temporal_solver = perf_counter()
        
        g_solve = solve_ivp(pbe_ode, [t_min,t_max],g0,
                                method = temporal_solver,
                                dense_output=True,
                                t_eval = t_solve, atol = 1e-3)
        
        t_ending_solve_ivp_temporal_solver = perf_counter()
        
    timeTakenBySolver = t_ending_solve_ivp_temporal_solver-t_starting_solve_ivp_temporal_solver
    print(f"The solve_ivp Solver took {t_ending_solve_ivp_temporal_solver-t_starting_solve_ivp_temporal_solver} seconds using " + temporal_solver + " method.")

    num_density_solve_ivp = np.zeros_like(g_solve.y)
    for i in range(len(g_solve.y[1,:])):
        num_density_solve_ivp[:,i] =  1*g_solve.y[:,i]/(x)
    
    num_density = num_density_solve_ivp.transpose()
    total_mass = np.zeros(len(num_density[:,0]))
    total_number_of_particles = np.zeros(len(num_density[:,0]))
    for i in range(len(num_density[:,1])):
        total_mass[i] =  np.sum(num_density[i,:]*x*delta_x)
        total_number_of_particles[i] = np.sum(num_density[i,:]*delta_x)
        
    """  Linear Algebric Solver ends """
    
    """ Creating DataFrame to save results in proper manner ## Logging"""
    
    df = pd.DataFrame({'x':pd.Series(x),
                       'DeltaX':pd.Series(delta_x),
                       'g_initial': pd.Series(g0),
                       'NumberDensity':pd.Series(num_density[-1,:]),
                       'g':pd.Series(num_density[-1,:]*x),
                        'DataEvalatTime':pd.Series(t_solve),
                        'TotalNumberofParticles':pd.Series(total_number_of_particles),
                        'TotalMass': pd.Series(total_mass), 
                        'TimeTakenBySolver' : pd.Series(timeTakenBySolver)})
    
    
    df_num_density = pd.concat([pd.DataFrame({'DataEvalatTime':pd.Series(t_solve)}),
                               pd.DataFrame(num_density)], axis = 1)
    
    
    FinalD32 = np.sum(np.power(d,3)*(df['g'].dropna()/(x))*delta_x)/np.sum(np.power(d,2)*(df['g'].dropna()/(x))*delta_x)
    FinalD43 = np.sum(np.power(d,4)*(df['g'].dropna()/(x))*delta_x)/np.sum(np.power(d,3)*(df['g'].dropna()/(x))*delta_x)
    # FinalD32 =0
    # FinalD43 = 0
    
    """Save Output"""
    
    if save_data == True:
        path = "Data/" + temporal_solver
        if os.path.isdir(path) == False:
            os.mkdir(path)
            os.chdir(path)
            filename = (temporal_solver + str(no_of_nodes) + 
                        str(int(delta_t*100))+
                        type_of_initial_condition + 
                        type_of_selection_function + 
                        type_of_breakage_function + 
                        type_coagulation_function + '.csv')
            filename_nd = ('num_den_diff_times'+ temporal_solver + str(no_of_nodes) + 
                        str(int(delta_t*100)) +
                        type_of_initial_condition + 
                        type_of_selection_function + 
                        type_of_breakage_function + 
                        type_coagulation_function + '.csv')
            df.to_csv(filename)
            df_num_density.to_csv(filename_nd)
            os.chdir('../..')
        else:
            os.chdir(path)
            filename = (temporal_solver + 
                        str(no_of_nodes) + 
                        str(int(delta_t*100)) +
                        type_of_initial_condition + 
                        type_of_selection_function + 
                        type_of_breakage_function + 
                        type_coagulation_function + '.csv')
            filename_nd = ('num_den_diff_times'+ temporal_solver + str(no_of_nodes) + 
                        str(int(delta_t*100)) +
                        type_of_initial_condition + 
                        type_of_selection_function + 
                        type_of_breakage_function + 
                        type_coagulation_function + '.csv')
            df.to_csv(filename)
            df_num_density.to_csv(filename_nd)
            os.chdir('../..')
    dict_data = {'timestamp':timestr,'minimimum_particle_size':minimimum_particle_size, 
           'maximum_particle_size':maximum_particle_size,
           'no_of_nodes':no_of_nodes,
           't_min':t_min, 
           't_max':t_max,
           'delta_t':delta_t,
           'type_of_problem':type_of_problem,
           'type_of_selection_function' :type_of_selection_function,
           'type_of_breakage_function' : type_of_breakage_function,
           'type_coagulation_function' : type_coagulation_function, 
           'type_of_initial_condition' : type_of_initial_condition, 
           'temporal_solver' : temporal_solver,
           'C1':C1, 
           'C2':C2, 
           'C3':C3, 
           'C4':C4, 
           'C5':C5,
           'C6':C6, 
           'C7':C7,
           'InitialD32':InitialD32,
           'InitialD43':InitialD43,
           'phiDP' : phiDP,
           'muCP' :muCP, 
           'muDP' :muDP, 
           'sigma':sigma, 
           'rhoCP': rhoCP,
           'rhoDP':rhoDP,
           'DissipationRate':DissipationRate, 
           'D': D, 
           'We': We,
           'FinalD32':FinalD32,
           'FinalD43':FinalD43}
    try:
        os.chdir('DataML2')
        with open(timestr+'.txt', 'w') as outfile:
            json.dump(dict_data, outfile)
        os.chdir('..')
    except:
        os.mkdir('DataML2')
        os.chdir('DataML2')
        with open( timestr+'.txt', 'w') as outfile:
            json.dump(dict_data, outfile)
        os.chdir('..')
            
    return g_solve, df
    # return g_solve