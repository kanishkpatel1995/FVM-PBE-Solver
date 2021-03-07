# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:26:28 2020

@author: kanishk 
"""
""" Post processing CLD Data from UofS"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm


def ExperimentalCLD(specify_csv_filename,specify_time, raw_data = None):
    ### Will return pdf for the experimental data if raw_data = None or else will
    ## return the original experimental data with midpoints of hist bar cannected by 
    # a line
    usecols = [i for i in range(4,104)]
    df_exp = pd.read_csv(specify_csv_filename,skiprows = 4, 
                         usecols = usecols, error_bad_lines = False)
    
    df_channel_midpoints = pd.read_csv('Experiment 2020-11-27 10-34 Default.csv',skiprows = range(2,), 
                         usecols = usecols, error_bad_lines = False)
    L = df_channel_midpoints.iloc[0,:].values
    
    df_exp.reset_index(drop=True, inplace=True)
    
    columns = [str(i) for i in range(len(df_exp.iloc[0,:]))]
    new_cols = dict(zip(df_exp.columns, columns)) 
    
    df_exp = df_exp.rename(columns = new_cols)
    
    
    Time = np.zeros(len(df_exp.iloc[:,1]))
    for t in range(len(Time)):
        Time[t]= t*10
    
    df_exp.insert(0,"Time (seconds)",Time)
    # L = np.logspace(0,3,len(df_exp.iloc[10,1:]) + 1)
    if specify_time == 'Last Time':
        counts = df_exp.iloc[-1,1:].values
    else:
        counts = df_exp.iloc[int(specify_time/10),1:].values
    df = pd.DataFrame({'Chord Length': pd.Series(L),'Counts':pd.Series(counts)})
    if raw_data == True or raw_data == None:
        return df
    else:
        x = L 
        mean = (np.sum(
            np.log(df['Chord Length'].values)*df['Counts'].values)
        /np.sum(df['Counts'].values))
        std = np.sqrt(
            np.sum(
                df['Counts'].values*
                np.power(
                    np.log(df['Chord Length'].values) - mean,2))
            /df['Counts'].sum())
        
        mu = mean
        sigma = std
        pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))/ (x * sigma * np.sqrt(2 * np.pi)))
        pdf_at_time_chord_length_dist = pd.DataFrame({'Chord Length': pd.Series(L),
                                                      'Probability':pd.Series(pdf)})
        return pdf_at_time_chord_length_dist
    
color = ['r','y','g','k','b','c','m','brown','y','gold']
i = 0
for time in [700,800,1600,2500,2800,2900,2910,'Last Time']:
    df = ExperimentalCLD('Experiment 2020-11-27 10-34 Default.csv',time, raw_data=True)
    plt.semilogx(df['Chord Length'].values,df['Counts'].values, color[i], label = f't = {time}')
    i = i+1
plt.legend(frameon = False, fontsize = 24)
    
time = 2900
df_cld = ExperimentalCLD('Experiment 2020-11-27 10-34 Default.csv',time, raw_data=False)
df = ExperimentalCLD('Experiment 2020-11-27 10-34 Default.csv',time, raw_data=True)
mu_original = np.sum(df['Chord Length'].values*df['Counts'].values)/np.sum(df['Counts'].values)
sigma_original = np.sqrt(np.sum(df['Counts'].values
                                *np.power(df['Chord Length'].values - 
                                          mu_original,2))/np.sum(df['Counts'].values))


### Plotting histogram of the droplets i.e DSD.
bins = np.logspace(0,3, 101)
# restoring data to obtain histogram, frequency data is converted to sample data 
restored = [[d]*int(df['Counts'].values[n]) for n,d in enumerate((bins[1:]+bins[:-1])/2)]
restored = [item for sublist in restored for item in sublist]
counts, bins, ignored = plt.hist(restored,bins = bins, density=True,color ='bisque', align='mid')
plt.xscale('log')

## Ealuating mean from restored data
act_mean = np.mean(np.log(restored))
act_std = np.std(np.log(restored))

# evaluating mean and standard deviation for the frequency distribution
mu_from_hist = np.sum(np.log(df['Chord Length'].values)*counts)/np.sum(counts)
sigma_from_hist = np.sqrt(np.sum(counts*np.power(np.log(df['Chord Length'].values) - mu_from_hist,2)
                                 )/np.sum(counts))

##evaluating from the regular exp distribution with no normalisation
mu = np.sum(np.log(df['Chord Length'].values)*df['Counts'].values)/np.sum(df['Counts'].values)
sigma = np.sqrt(np.sum(df['Counts'].values*np.power(np.log(df['Chord Length'].values) - mu,2))/np.sum(df['Counts'].values))

x = np.logspace(0,3,10000)

def pdf(x,mean,sigma):
    func = (1/(sigma*np.sqrt(2*np.pi)*x))*np.exp(-np.power((np.log(x)-mean),2)/(2*sigma*sigma))
    return func

#The below two must be same slight variation acceptable given log values have been introduced
plt.semilogx(x, pdf(x,act_mean,act_std), linewidth=2, color='k')
plt.semilogx(x, pdf(x,mu_from_hist,sigma_from_hist), linewidth=2, color='r')

## why different. 
plt.semilogx(x, pdf(x,mu,sigma), linewidth=2, color='g')

### Now using lognormal fit to fit the restotred data. 
shape,loc,scale = lognorm.fit(restored)

x = np.logspace(0, 5, 200)
pdf = lognorm.pdf(x, shape, loc, scale)

## I recieve a lognormal distribution with different mean and standard deviation 
plt.plot(x, pdf, 'c')
plt.xscale('log')

####Why does the fit differ from my distribution 

# mu = 3
# sigma = 1.1
# s = np.random.lognormal(mu, sigma, 100000)
# mean_randon_data = np.mean(s)
# std_random_data = np.std(s)
# # how to get normal mean and standard deviation from the log normal data 
# normal_mean_from_lognormal_data = np.exp(mu + 0.5*sigma*sigma)
# normal_std_from_log_normal_data = np.sqrt(np.exp(2*mu + sigma*sigma) * (np.exp(sigma*sigma)-1) )

# ## how to get mu and sigma from the lognormal distribution mean and std
# mu_formula = 2*np.log(mean_randon_data) - 0.5*np.log(std_random_data*std_random_data 
#                                                      +mean_randon_data*mean_randon_data)
# std_formula = np.sqrt(-2*np.log(mean_randon_data) + np.log(std_random_data*std_random_data 
#                                                      +mean_randon_data*mean_randon_data))
# count, bins, ignored = plt.hist(s, 100, density=True, align='mid')
# plt.xscale('log')
# #obtaining curve by fitting the distribution
# shape,loc,scale = lognorm.fit(s)
# x = np.logspace(0, 5, 200)
# pdf = lognorm.pdf(x, shape, loc, scale)
# plt.plot(x, pdf, 'r')
# #obtaining curve by evaluating mean and sigma
# x = np.linspace(min(bins), max(bins), 10000)
# mu_from_pdf = np.sum(np.log(x)*pdf)/np.sum(pdf)
# std_from_pdf = np.sqrt(np.sum(pdf
#                                 *np.power(np.log(x) - 
#                                           mu_from_pdf,2))/np.sum(pdf))

# pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
#         / (x * sigma * np.sqrt(2 * np.pi)))
# plt.plot(x, pdf, 'b')
# plt.xscale('log')











