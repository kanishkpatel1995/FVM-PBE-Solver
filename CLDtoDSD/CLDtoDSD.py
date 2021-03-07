# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:13:09 2020

@author: Kanishk
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CLDfromExpData import ExperimentalCLD

##### Method 1 
def LogNormalFunc(D,sigma,D_0,D_max):
    f = np.zeros(len(D))
    i  = 0 
    D_0 = D_0
    for d in D:
        if d>D_max:
            f[i] = 0
        else:
            f[i] = ((1/(sigma*np.sqrt(2*np.pi)*d))*
                    np.exp(-(np.power(np.log(d/D_0),2))/(2*np.power(sigma,2))))
        i = i+1
    return f

def CLDtoDSDMethod1(specify_csv_filename,specify_time, provide_mean_lengths = None):
    
    
    def LogNormalFunc(D,sigma,D_0,D_max):
        f = np.zeros(len(D))
        i  = 0 
        D_0 = D_0
        for d in D:
            if d>D_max:
                f[i] = 0
            else:
                f[i] = ((1/(sigma*np.sqrt(2*np.pi)*d))*
                        np.exp(-(np.power(np.log(d/D_0),2))/(2*np.power(sigma,2))))
            i = i+1
        return f
    
    ### Getting the needed Experimental data 
    df = ExperimentalCLD(specify_csv_filename,specify_time)
    total_number_lengths_measured = df['Counts'].sum()
    #### Normalising the experimental data. 
    ## By dividing the frequency with the total number of counts.
    df['Counts'] = df['Counts']/np.sum(df['Counts'])
    
    ## empirical constants taken from 
    ####Schümann, H., Khatibi, M., Tutkun, M., H. Pettersen, B., Yang, Z., & Nydal, O. J. (2015). 
    ###Droplet Size Measurements in Oil–Water Dispersions: A Comparison Study Using FBRM and PVM. 
    ##Journal of Dispersion Science and Technology, 36(10), 1432–1443. 
    k43 = 4.72
    k32 = 5.49
    ### Getting surface and volume chord length 
    L_32 = (np.sum(df.iloc[:,1].values * np.power(df.iloc[:,0].values,3))/
         np.sum(df.iloc[:,1].values * np.power(df.iloc[:,0].values,2)))
    
    L_43 = (np.sum(df.iloc[:,1].values * np.power(df.iloc[:,0].values,4))/
         np.sum(df.iloc[:,1].values * np.power(df.iloc[:,0].values,3)))
    #### Using empirical constants to obtain mean diameters
    D_32 = k32 * L_32
    D_43 = k43 * L_43
    ### Obtaining parameter for assumed log normal distribution of dropet Diameters
    sigma = np.sqrt(np.log(D_43/D_32))
    D_0 = np.power(D_32*D_43,0.5) * np.power(D_32/D_43,3)
    
    ## Define maximum possible droplet size to avoid the curve from predicitng 
    ### Unecessary counts
    c  = 1/0.48
    D_max = c*D_32
    #diving the space to numerically obtain log normal distribution 
    D = np.logspace(0,3,101)
    ### Using an iterative progression method to make sure that the values of mean 
    # diameters obtained from the assumed log normal distribution match the experimentally 
    #predicted values.
    unreal_D32 = 1
    unreal_D43 = 1
    max_iterations = 100
    beta = np.ones(max_iterations+1)
    Iter = 0 
    step = 10
    tol = 1e-4
    while (abs((unreal_D32 - D_32)/D_32)>tol and abs((unreal_D43 - D_43)/D_43)>tol) and Iter < max_iterations:
        print(Iter)
        f = LogNormalFunc(D,beta[Iter]*sigma,D_0, D_max)
        f = f/f.sum()
        f = f*total_number_lengths_measured
        # print(beta[Iter]*sigma)
        DSD = pd.DataFrame({'Diameter':pd.Series(D),'Counts':pd.Series(f)})
        
        unreal_D43 = (np.sum(DSD.iloc[:,1].values * np.power(DSD.iloc[:,0].values,4))/
         np.sum(DSD.iloc[:,1].values * np.power(DSD.iloc[:,0].values,3)))
        
        unreal_D32 = (np.sum(DSD.iloc[:,1].values * np.power(DSD.iloc[:,0].values,3))/
         np.sum(DSD.iloc[:,1].values * np.power(DSD.iloc[:,0].values,2)))
        
        sigma_DSD = np.sqrt(np.log(unreal_D43/unreal_D32))
        # print(sigma_DSD)
        
        if (unreal_D43 - D_43) > 0 :
            beta[Iter+1] = beta[Iter] - (1/step)
            if beta[Iter - 1] == beta[Iter + 1]:
                step = step*10
            else:
                pass
        else:
            beta[Iter+1] = beta[Iter] + (1/step)
            if beta[Iter - 1] == beta[Iter + 1]:
                step = step*10
            else:
                pass
        Iter = Iter + 1
    
    ## Returning the dataframe with first coloumn as diameter and the second coloumn
    # containing the frequency of each diameter.
    if provide_mean_lengths == True:
        return L_32,L_43, D_32, D_43, unreal_D43, unreal_D32
    else:
        return DSD

    
### Method 2

def CLDtoDSDMethod2(specify_csv_filename,specify_time, provide_mean_lengths = None):
    
    
    df = ExperimentalCLD(specify_csv_filename,specify_time)
    total_number_lengths_measured = df['Counts'].sum()
    #### Normalising the experimental data. 
    ## By dividing the frequency with the total number of counts.
    # df['Counts'] = df['Counts']/np.sum(df['Counts'])
    
    ## empirical constants taken from 
    ####Schümann, H., Khatibi, M., Tutkun, M., H. Pettersen, B., Yang, Z., & Nydal, O. J. (2015). 
    ###Droplet Size Measurements in Oil–Water Dispersions: A Comparison Study Using FBRM and PVM. 
    ##Journal of Dispersion Science and Technology, 36(10), 1432–1443.
    k43 = 4.72
    k32 = 5.49
    
    L_32 = (np.sum(df.iloc[:,1].values * np.power(df.iloc[:,0].values,3))/
          np.sum(df.iloc[:,1].values * np.power(df.iloc[:,0].values,2)))
    
    L_43 = (np.sum(df.iloc[:,1].values * np.power(df.iloc[:,0].values,4))/
          np.sum(df.iloc[:,1].values * np.power(df.iloc[:,0].values,3)))
    
    L_0 = np.power(L_32 * L_43,0.5) * np.power(L_32/L_43,3)
    sigma_L = np.sqrt(np.log(L_43/L_32))
    D_32 = k32 * L_32
    D_43 = k43 * L_43  
    sigma = np.sqrt(np.log(D_43/D_32))
    D_0 = np.power(D_32*D_43,0.5) * np.power(D_32/D_43,3)

    
    C = (np.log10(df['Chord Length'].values) - np.log10(L_0))/sigma_L
    D = np.power(10,np.log10(D_0) + C*sigma)
    
    F = np.zeros(len(D))
    
    for i in range(len(D)-1):
        F[i] = (df['Counts'].values[i] * (np.log10(df['Chord Length'].values[i+1])-
                                      np.log10(df['Chord Length'].values[i])))/(
        np.log10(D[i+1])-np.log10(D[i]))
    
    DSD = pd.DataFrame({'Diameter':pd.Series(D),'Counts':pd.Series(F)})
    
    unreal_D43 = (np.sum(DSD.iloc[:,1].values * np.power(DSD.iloc[:,0].values,4))/
         np.sum(DSD.iloc[:,1].values * np.power(DSD.iloc[:,0].values,3)))
        
    unreal_D32 = (np.sum(DSD.iloc[:,1].values * np.power(DSD.iloc[:,0].values,3))/
         np.sum(DSD.iloc[:,1].values * np.power(DSD.iloc[:,0].values,2)))
    
    if provide_mean_lengths == True:
        return L_32,L_43, D_32, D_43, unreal_D43, unreal_D32
    else:
        return DSD

#### TRIAL USAGE #########


# DSD = CLDtoDSDMethod1('Experiment 2020-11-27 10-34 Default.csv','Last Time')
# L_32,L_43, D_32, D_43, unreal_D43, unreal_D32 = CLDtoDSDMethod2('Experiment 2020-11-27 10-34 Default.csv','Last Time',
#                                                                 provide_mean_lengths=True)
# CLD = ExperimentalCLD('Experiment 2020-11-27 10-34 Default.csv','Last Time')
# plt.semilogx(CLD.iloc[:,0].values , CLD.iloc[:,1].values,label = f'CLD')
# plt.semilogx(DSD['Diameter'].values , DSD['Counts'].values,label = f'Method 1') 
# plt.semilogx(DSD_m2['Diameter'].values , DSD_m2['Counts'].values,label = f'Method 2')   
    
    
    
    
        
