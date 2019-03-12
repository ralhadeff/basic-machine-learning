'''
A tool for implementing GridSearch
The tool takes an estimator and runs a k-fold CV on all combinations of parameters requested (the grid search) and returns which parameters scored best
'''

import numpy as np
from collections import Iterable
import itertools
import kfold

def grid_search(estimator, X, y, params, cv_k=5, scoring=None,maximize=True):
    '''
    Performs fitting of estimator using all the combinations of parameters provided in params (a dictionary)
    Performance is measured by k-fold cross-validation,
    using the specified number of folds,
    and the provided scoring function (or estimator.score by default)
    User can specify maximize=False if the scoring function should be minimized rather than maximized

    Returns a dictionary with the best combination of parameters
    '''
    # generate list of dictionaries for all possible combinations
    # this will first generate a list of single item dictionaries, one least for each key (argument)
    # and in the list, one dictionary for each possiblity requested
    arguments = []
    for i in params:
        options = []
        if (isinstance(params[i],Iterable) and type(params[i])!=str):
            # iterate over list
            for j in params[i]:
                options.append([i,j])
        else:
            # single option
            options.append([i,params[i]])
        arguments.append(options)
    # generate all combinations
    dictionaries = list(itertools.product(*arguments))  
    
    scores = []
    # score each combination
    for i in dictionaries:
        # transform list to dictionary
        args = {}
        for d in i:
            args[d[0]] = d[1]
        scores.append(kfold.k_fold_cv(estimator,X,y,cv_k,args,scoring=scoring).mean())
    
    # get index of best combination
    if (maximize):
        best = np.argmax(np.asarray(scores))
    else:
        best = np.argmin(np.asarray(scores))
    # return best combination, as a dictionary
    return dict(dictionaries[best])

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
