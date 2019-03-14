'''
This tool produces a learning curve for a given estimator
The estimator must have a .score(X,y) function
'''

import numpy as np

def learning_curve(estimator,X_train,y_train,X_validation,y_validation,steps=1,fitting_params=None):
    '''
    Calculate the learning curve, train vs validation
    steps defines the step size for iterating over datapoints
    fitting_params is an optional library of parameters to pass to the fit function of the specific estimator
    '''
    # number of points that will be generated:
    n = len(range(X_train.shape[1],X_train.shape[0],steps))
    # score array
    learn = np.zeros((n-1,3))

    # fit to subsets of train and calculate metrics 
    for i in range(0,n-1):
        # skip the first point (error tends to explode when training set is minuscule)
        j = (i+1)*steps+X_train.shape[1]
        learn[i,0] = j
        # fit, passing fit arguments if needed
        if (fitting_params==None):
            estimator.fit(X_train[:j,:],y_train[:j])
        else:
            estimator.fit(X_train[:j,:],y_train[:j],**fitting_params)
        # update scores
        learn[i,1] = estimator.score(X_train[:j,:],y_train[:j])
        # validation scores
        learn[i,2] = estimator.score(X_validation,y_validation)
        
    return learn

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
