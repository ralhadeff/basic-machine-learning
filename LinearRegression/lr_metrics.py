'''
Linear regression metrics tools
'''

import numpy as np

def mae(real_y,predictions):
    '''
    Mean absolute error
    '''
    return (np.abs(real_y-predictions)).mean()

def mse(real_y,predictions):
    '''
    Mean squared error
    '''
    return ((real_y-predictions)**2).mean()

def rss(real_y,predictions):
    '''
    Residual sum of squares
    also called sum of squared residuals
    also SSE
    '''
    return ((real_y-predictions)**2).sum()

def r_squared(real_y,predictions):
    '''
    R squared value
    '''
    rss = ((real_y-predictions)**2).sum()
    y_mean = real_y.mean()
    tss = ((real_y-y_mean)**2).sum()
    return 1-(rss/tss)

def adjusted_r_squared(X=None, real_y=None,predictions=None,r2=None,sample_size=None,no_of_features=None):
    '''
    Adjusted R squared
    input can either be X, real_y and predictions (must provide also X to count the number of features)
       OR
    the R squared, sample size n and number of features p
    '''
    if (X is None or real_y is None or predictions is None):
        r = r2
        n = sample_size
        p = no_of_features
    else:
        r = r_squared(real_y,predictions)
        n,p = X.shape
        # check if the first column of X is ones
        if (X[0,0]==1):
            if (np.unique(X[:,0]).shape==(1,)):
                p-=1
        
    return 1-(1-r)*(n-1)/(n-p-1)

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
