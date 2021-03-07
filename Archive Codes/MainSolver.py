# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 18:17:55 2020

@author: kanishk 

Main solver 
"""
###Imported modules - online 
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import matplotlib.pyplot as plt
from Grid_X import GridX
from TemporalGrid import TemporalGrid
from BreakageFunction import BreakageFunction
from SelectionFunction import SelectionFunction
from CoagulationFunction import CoagulationFunction
from AMatrix import APureBreakage
from AnalyticalSolution import AnalyticalSolution
from InitialCondition import InitialConditionNumberDensity

def PureBreakagePBESolver(minimimum_particle_size, 
               maximum_particle_size,
               no_of_nodes,
               t_min, 
               t_max,
               delta_t,
               type_of_selection_function, type_of_initial_condition):
    #########Initial Conditions and boundary conditions ########
    # # Minimum particle size 
    # minimimum_particle_size = 1e-8
    # # maximum particle size 
    # maximum_particle_size = 1
    # #initial time
    # t_min = 0 
    # # Final Time
    # t_max = 1000
    # # Save solution after Delta_t time 
    # delta_t = 10
    
    
    
    # # Grid Generation 
    # no_of_nodes = 31
    x, x_node_boundaries,delta_x = GridX(minimimum_particle_size,maximum_particle_size,no_of_nodes).UniformLogarithmic()
    
    #solution parameters 
    #number density
    if type_of_initial_condition == 'OneLargeParticle':
        n = InitialConditionNumberDensity(x,delta_x).OneLargeParticle()
    elif type_of_initial_condition == 'ExponentialDistribution':
        n = InitialConditionNumberDensity(x,delta_x).ExponentialDistribution()
    # particle mass density
    g = x*n  
    
    ### Getting spatial matrix A pure breakage ###
    class APureBreakage:
        def __init__(self, x, x_node_boundaries, delta_x, type_s):
            self.x = x
            self.x_node_boundaries = x_node_boundaries
            self.delta_x = delta_x
            self.type_s = type_s
        
        def derec(self,k,i):
            d_k_i = 0
            inner_integral = 0
            if i>=0:
                for j in range(i+1): 
                    inner_integral = inner_integral + self.x[j]*BreakageFunction(self.x[j],self.x[k])*self.delta_x[j]
            else :
                inner_integral = 0
            d_k_i = SelectionFunction(self.x[k],self.type_s)*(1/self.x[k])*self.delta_x[k]*inner_integral
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
    
    
    A = APureBreakage(x, x_node_boundaries,delta_x,type_of_selection_function).A()
    
    
    """Linear Algebric solver """
    ## Implementing Python inbuilt fourth order Runga Kutta solver 
    ## The solver evaluates the particle mass density.
    def pbe_ode(t,y):
        return np.matmul(A,y)
    
    ## time array defining timesteps at which solution wil be saved.
    t_solve = TemporalGrid(t_min,t_max,delta_t).Uniform()
    
    g_solve = solve_ivp(pbe_ode, [t_min,t_max],g,t_eval = t_solve, vectorized=True)
    
    """  Linear Algebric Solver ends """
    
    
    num_density = np.zeros_like(g_solve.y)
    for i in range(len(g_solve.y[1,:])):
        num_density[:,i] =  1*g_solve.y[:,i]/(x)
    
    return x,delta_x,num_density


###############################################################################
############################################################################
###############################################################################

""" Aggregation Solver """

#############################################################################

def PureAggregtionPBESolver(minimimum_particle_size, 
               maximum_particle_size,
               no_of_nodes,
               t_min, 
               t_max,
               delta_t,
               type_coagulation_function = None, type_of_initial_condition = None, temporal_solver = None):
    
    if type_coagulation_function == None or type_coagulation_function == 'ConstantUnity':
        type_coagulation_function == 'ConstantUnity'
    else: 
        pass
    
    #########Initial Conditions and boundary conditions ########
    # # Minimum particle size 
    # minimimum_particle_size = 1e-8
    # # maximum particle size 
    # maximum_particle_size = 1
    # #initial time
    # t_min = 0 
    # # Final Time
    # t_max = 1000
    # # Save solution after Delta_t time 
    # delta_t = 10
    
    
    
    # # Grid Generation 
    # no_of_nodes = 31
    x, x_node_boundaries,delta_x = GridX(minimimum_particle_size,maximum_particle_size,no_of_nodes).UniformLogarithmic()
    
    #solution parameters 
    #number density

    if type_of_initial_condition == 'ExponentialDistribution' and type_coagulation_function == 'Product':
        n = InitialConditionNumberDensity(x,delta_x).FilbetAndLaurencotExponentialDistributionForProductKernel()
    elif type_of_initial_condition == 'ExponentialDistribution' or None and type_coagulation_function != 'Product':
        n = InitialConditionNumberDensity(x,delta_x).FilbetAndLaurencotExponentialDistribution()
    elif type_of_initial_condition == 'LND':
        n = InitialConditionNumberDensity(x,delta_x).LogNormalDistribution()
    # Initial particle mass density
    
    g0 = x*n  
    
    ### Getting spatial matrix A pure aggregation ###
    class APureAggregation:
        ### Getting spatial matrix A pure aggregation ###
        # we need to add g array from previous timestep as the solution is now dependent on g 
        def __init__(self, x, x_node_boundaries, delta_x,g):
            self.x = x
            self.x_node_boundaries = x_node_boundaries
            self.delta_x = delta_x
            self.g = g # it is the mass density matrix
        
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
                               CoagulationFunction(x[alpha:], x[k], type_coagulation_function)
                               *delta_x[alpha:])
            
            def integrand(x,a):
                return (1/x)*CoagulationFunction(x,a,type_coagulation_function)
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
                #print(i)
                for j in range(i+1):
                    J[i,j] = self.delta_x[j] * self.derec(i,j)
            for i in range(len(A[:,0])):
                if i == 0:
                    A[i,:] = (-1/self.delta_x[i]) * J[i,:]
                else:
                    A[i,:] = (-1/self.delta_x[i]) * (J[i,:] - J[i-1,:])
            return A

# g_at_previous_time = g
    
    # g_at_previous_time = g
    # A = APureAggregation(x, x_node_boundaries,delta_x,type_coagulation_function,g_at_previous_time).A()
    
    
    """Linear Algebric solvers """
    ## Implementing EulerExplicitSolver as A mtrix depends on time 
    ## The solver evaluates the particle mass density.
    
    
    if temporal_solver == 'Euler Explicit':
        """ Euler Explicit Solver start """
        # creating time array 
        t = TemporalGrid(t_min,t_max,delta_t).Uniform()
        t_solve = t
        g_euler_eplicit = np.zeros([len(t),len(g0)])
        g_euler_eplicit[0,:] = g0
        for time in range(len(t)):
            A = APureAggregation(x, x_node_boundaries,delta_x,type_coagulation_function,g_euler_eplicit[time]).A()
            g_euler_eplicit[time+1,:] = g_euler_eplicit[time,:] + (t[time+1] - t[time])*np.matmul(A, g_euler_eplicit[time,:])
        
        num_density = np.zeros_like(g_euler_eplicit)
        total_mass = np.zeros(len(num_density[:,0]))
        total_number_of_particles = np.zeros(len(num_density[:,0]))
        for i in range(len(g_euler_eplicit[:,1])):
            num_density[i,:] =  g_euler_eplicit[i,:]/(x)
            total_mass[i] =  np.sum(num_density[i,:]*x*delta_x)
            total_number_of_particles[i] = np.sum(num_density[i,:]*delta_x)
    
    else:
        
        def pbe_ode(t,y):
            dy_dt = np.matmul(APureAggregation(x, x_node_boundaries,delta_x,y).A(),y)
            return dy_dt
            
        t_solve = TemporalGrid(t_min,t_max,delta_t).Uniform()
        
        g_solve = solve_ivp(pbe_ode, [t_min,t_max],g0,method='RK45',t_eval = t_solve)
        
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
    
    
    return x,delta_x,t_solve,num_density,total_mass,total_number_of_particles


def PBEBrkAggSolver(minimimum_particle_size, 
               maximum_particle_size,
               no_of_nodes,
               t_min, 
               t_max,
               delta_t,
               type_of_selection_function = None,
               type_of_breakage_function = None,
               type_coagulation_function = None, type_of_initial_condition = None, temporal_solver = None):
    
    
    x, x_node_boundaries,delta_x = GridX(minimimum_particle_size,
                                         maximum_particle_size,
                                         no_of_nodes).UniformLogarithmic()
    
    
    
    if type_of_initial_condition == 'ExponentialDistribution' and type_coagulation_function == 'Product':
        n = InitialConditionNumberDensity(x,delta_x).FilbetAndLaurencotExponentialDistributionForProductKernel()
    elif type_of_initial_condition == 'ExponentialDistribution' or None and type_coagulation_function != 'Product':
        n = InitialConditionNumberDensity(x,delta_x).FilbetAndLaurencotExponentialDistribution()
    elif type_of_initial_condition == None or type_of_initial_condition == 'Exp':
        n = InitialConditionNumberDensity(x, delta_x).ExponentialDistribution()
    elif type_of_initial_condition == 'LND':  
        n = InitialConditionNumberDensity(x, delta_x).LogNormalDistribution()
    # Initial particle mass density
    
    g0 = x*n
    
    
    
    
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
                    inner_integral = inner_integral + self.x[j]*BreakageFunction(self.x[j],
                                                                                 self.x[k],
                                                                                 self.type_of_breakage_function
                                                                                 )*self.delta_x[j]
            else :
                inner_integral = 0
            d_k_i = SelectionFunction(self.x[k],self.type_s)*(1/self.x[k])*self.delta_x[k]*inner_integral
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
                               CoagulationFunction(x[alpha:], x[k], self.type_coagulation_function)
                               *delta_x[alpha:])
            # print(CoagulationFunction(x[alpha:], x[k], self.type_coagulation_function))
            def integrand(x,a):
                return (1/x)*CoagulationFunction(x,a,self.type_coagulation_function)
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
                print(i)
                for j in range(i+1):
                    J[i,j] = self.delta_x[j] * self.derec(i,j)
            for i in range(len(A[:,0])):
                if i == 0:
                    A[i,:] = (-1/self.delta_x[i]) * J[i,:]
                else:
                    A[i,:] = (-1/self.delta_x[i]) * (J[i,:] - J[i-1,:])
            # print(A)
            return A
    

    """Linear Algebric solvers """
    ## Implementing EulerExplicitSolver as A mtrix depends on time 
    ## The solver evaluates the particle mass density.
    
    
    if temporal_solver == 'Euler Explicit':
        """ Euler Explicit Solver start """
        # creating time array 
        t = TemporalGrid(t_min,t_max,delta_t).Uniform()
        t_solve = t
        g_euler_eplicit = np.zeros([len(t),len(g0)])
        g_euler_eplicit[0,:] = g0
        for time in range(len(t)-1):
            print(time)
            A = (APureAggregation(x, x_node_boundaries,delta_x,
                                 g_euler_eplicit[time],type_coagulation_function).A() + 
                 APureBreakage(x, x_node_boundaries,delta_x,
                               type_of_selection_function,type_of_breakage_function).A())
            g_euler_eplicit[time+1,:] = g_euler_eplicit[time,:] + (t[time+1] - t[time])*np.matmul(A, g_euler_eplicit[time,:])
        
        num_density = np.zeros_like(g_euler_eplicit)
        total_mass = np.zeros(len(num_density[:,0]))
        total_number_of_particles = np.zeros(len(num_density[:,0]))
        for i in range(len(g_euler_eplicit[:,1])):
            num_density[i,:] =  g_euler_eplicit[i,:]/(x)
            total_mass[i] =  np.sum(num_density[i,:]*x*delta_x)
            total_number_of_particles[i] = np.sum(num_density[i,:]*delta_x)
    
    else:
        
        def pbe_ode(t,y):
            dy_dt = np.matmul((APureAggregation(x, x_node_boundaries,delta_x,y,type_coagulation_function).A() + 
                              APureBreakage(x, x_node_boundaries,delta_x,
                               type_of_selection_function,type_of_breakage_function).A()),y)
            return dy_dt
            
        t_solve = TemporalGrid(t_min,t_max,delta_t).Uniform()
        
        g_solve = solve_ivp(pbe_ode, [t_min,t_max],g0,method='RK45',t_eval = t_solve)
        
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
    
    
    return x,delta_x,t_solve,num_density,total_mass,total_number_of_particles





