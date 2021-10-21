#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:02:46 2021

@author: teagannorrgard
"""

# load modules
import numpy as np
import pandas as pd

# risk-free Treasury rate
R_f = 0.0175 / 252

# read in the market data
data = pd.read_csv('capm_market_data.csv')

### Look at some records  
### SPY is an ETF for the S&P 500 (the "stock market")  
### AAPL is Apple  
### The values are closing prices, adjusted for splits and dividends



### Drop the date column

data = data.drop('date', 1)

### Compute daily returns (percentage changes in price) for SPY, AAPL  
### Be sure to drop the first row of NaN  
### Hint: pandas has functions to easily do this

data.drop(0, axis=0)
data_returns = data.pct_change()

#### 1. (1 PT) Print the first 5 rows of returns

data_returns.head(5)

### Save AAPL, SPY returns into separate numpy arrays  
#### 2. (1 PT) Print the first five values from the SPY numpy array, and the AAPL numpy array

spynp = np.array(data['spy_adj_close'])
spynp[0:5]

aaplnp = np.array(data['aapl_adj_close'])
aaplnp[0:5]

##### Compute the excess returns of AAPL, SPY by simply subtracting the constant *R_f* from the returns.
##### Specifically, for the numpy array containing AAPL returns, subtract *R_f* from each of the returns. Repeat for SPY returns.

NOTE:  
AAPL - *R_f* = excess return of Apple stock  
SPY - *R_f* = excess return of stock market


aapl_excess = aaplnp - R_f
spy_excess = spynp - R_f

#### 3. (1 PT) Print the LAST five excess returns from both AAPL, SPY numpy arrays


aapl_excess[-5:]

spy_excess[-5:]

#### 4. (1 PT) Make a scatterplot with SPY excess returns on x-axis, AAPL excess returns on y-axis####
### Matplotlib documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

import matplotlib.pyplot as plt

plt.scatter(x=spy_excess, y=aapl_excess)

#### 5. (3 PTS) Use Linear Algebra (matrices) to Compute the Regression Coefficient Estimate, \\(\hat\beta_i\\)

### Hint 1: Here is the matrix formula where *x′* denotes transpose of *x*.

### \begin{aligned} \hat\beta_i=(x′x)^{−1}x′y \end{aligned} 

### Hint 2: consider numpy functions for matrix multiplication, transpose, and inverse. Be sure to review what these operations do, and how they work, if you're a bit rusty.

x = spy_excess.reshape(-1, 1)
y = aapl_excess.reshape(-1, 1)

x_transpose = x.transpose()
mult = np.matmul(x_transpose, x)
inverse = mult**(-1)
inverse_times_transpose = inverse * x_transpose
beta_est = np.matmul(inverse_times_transpose, y)
beta_est

x = spy_excess.reshape(-1, 1)
y = aapl_excess.reshape(-1, 1)

beta_est = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.transpose(), x)), x.transpose()), y)[0][0]
beta_est

x = spy_excess.reshape(-1, 1)
y = aapl_excess.reshape(-1, 1)

mult = np.matmul(x.transpose(), x)
inv = np.linalg.inv(mult)
inv_trans = np.matmul(inv, x.transpose())
beta_est = np.matmul(inv_trans, y)
beta_est

### You should have found that the beta estimate is greater than one.  
### This means that the risk of AAPL stock, given the data, and according to this particular (flawed) model,  
### is higher relative to the risk of the S&P 500.




#### Measuring Beta Sensitivity to Dropping Observations (Jackknifing)

### Let's understand how sensitive the beta is to each data point.   
### We want to drop each data point (one at a time), compute \\(\hat\beta_i\\) using our formula from above, and save each measurement.

#### 6. (3 PTS) Write a function called `beta_sensitivity()` with these specs:

### - take numpy arrays x and y as inputs
### - output a list of tuples. each tuple contains (observation row dropped, beta estimate)

### Hint: **np.delete(x, i).reshape(-1,1)** will delete observation i from array x, and make it a column vector

def beta_sensitivity(x, y):
    '''
    PURPOSE
    INPUTS
    OUTPUTS
    '''
    
    for i in x:
        x_revised = np.delete(x, i).reshape(-1, 1)
        x_revised_transpose = x_revised.transpose()
        mult = np.matmul(x_revised_transpose, x_revised)
        inverse = mult**(-1)
        inverse_times_transpose = inverse * x_revised_transpose
        beta_est = np.matmul(inverse_times_transpose, y)
        return beta_est

#### Call `beta_sensitivity()` and print the first five tuples of output.

beta_sensitivity(spy_excess, aapl_excess)

