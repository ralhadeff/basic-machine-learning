'''
This tool allows the performance of k-fold cross-validation using any estimator and any score callable
'''

import numpy as np

def k_fold_cv(estimator, X, y, k, fit_param=None, scoring=None):
    '''
    Perform k-fold cross-validation on data X and y using estimator to fit
    User can provide fit parameters for the estimator using a dictionary
    Scoring is using the default estimator.score() function. Otherwise, the user can provide a callable
    '''
    output = []
    # start from 0
    previous = 0
    for i in np.linspace(0,len(X),k+1)[1:]:
        # last index of current slice
        current = int(round(i))
        # generate slices data
        c_x = np.concatenate((X[0:previous],X[current:]),axis=0)
        c_y = np.concatenate((y[0:previous],y[current:]),axis=0)
        # fit estimator
        if (fit_param==None):
            # default parameters
            estimator.fit(c_x,c_y)
        else:
            estimator.fit(c_x,c_y,**fit_param)
        # score vs validation slices
        v_x = X[previous:current]
        v_y = y[previous:current]
        if (scoring==None):
            # default scoring
            output.append(estimator.score(v_x,v_y))
        else:
            # user provided scoring function
            output.append(scoring(v_y,estimator.predict(v_x)))
        # first index for next iteration
        previous = current

    return np.asarray(output).mean()

if (__name__ == '__main__'):
    print('This module is not intended to run by iself')
