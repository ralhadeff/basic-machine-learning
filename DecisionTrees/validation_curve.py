'''
A tool for calculating the data for a validation curve
The tool takes an estimator and runs a k-fold CV on a range of values for one hyper-parameters
    And returns an array for ploting the validation curve
    hyper-parameter    training_score    cross-validation_score
'''

import numpy as np
from collections import Iterable
import kfold

def validation_curve(estimator, X, y, params, cv_k=5, scoring=None, param_for_fit=True):
    '''
    Performs fitting of estimator using the parameters and all values of the hyper-parameter
        provided in a dictionary
    Performance is measured by k-fold cross-validation,
    using the specified number of folds,
    and the provided scoring function (or estimator.score by default)
    by default the params dictionary is provided to the estimator.fit method
        however, user can flag param_for_fit as false, and the dictionary 
        will be provided to the estimator __init__ method instead
        either way, user should provide an instance of estimator

    Returns an array of the results
    '''
    # find the key of the target hyper-parameter
    for i in params:
        if (isinstance(params[i],Iterable)):
            # key of hyper-parameters
            x = i
    # copy dictionary (to later modify it without affecting the input)
    args = dict(params)
    # values for the curve to return
    curve = np.zeros((len(args[x]),3))
    # hyper-parameter values requested
    curve[:,0] = args[x]
    dtype = type(args[x][0])
    # score each combination
    i=0
    for hyp in curve[:,0]:
        # insert a single value into args dictionary
        args[x] = dtype(hyp)
        if (param_for_fit):
            # parameters are providing in fit
            estimator.fit(X,y,**args)
        else:
            # parameters are provided on creation
            estimator = estimator.__class__(**args)
            estimator.fit(X,y)
        curve[i,1] = estimator.score(X,y)
        if (param_for_fit):
            curve[i,2] = kfold.k_fold_cv(estimator,X,y,cv_k,args,scoring=scoring)    
        else:
            curve[i,2] = kfold.k_fold_cv(estimator,X,y,cv_k,scoring=scoring)                
        i+=1
    return curve      

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
