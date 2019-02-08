"""
Label encoder
"""

import numpy as np
import pandas as pd

def encode(X,omit_one=None):
    """
    Label encoder - will transform a categorical column into a sparse matrix of labels
        by default, skip the last value (all 0's)
    User can provide a numpy array or a pandas DataFrame and output will match the type
    """
    # number of samples
    m = len(X)
    # unique labels
    labels = np.unique(X)
    skip = None
    # by default, omit one (which will be all 0's)
    if (omit_one is None):
        n = len(labels)-1
        # print out for user's interest
        print('omitting',labels[-1])
    elif (type(omit_one) is str):
        # omit specific by name
        n = len(labels)
        # get index of the requested skip
        skip = list(labels).index(omit_one)
        print('omitting',labels[skip])
    elif (type(omit_one) is int):
        # omit specific by index
        n = len(labels)
        skip = omit_one
        print('omitting',labels[skip])
    else:
        n = len(labels)
    # initialize results array
    L = np.zeros((m,n),dtype=int)
    # check whether output should be a numpy array or a pandas DataFrame
    pandas = False
    if (type(X) is pd.DataFrame):# or type(X) is pd.Series):
        # transform X to 1d array for later use
        X = X.values.ravel()
        pandas = True
    # populate sparse matrix
    for i in range(n):
        if (skip is None):
            L[X==labels[i],i] = 1
        else:
            if (i != skip):
                L[X==labels[i],i] = 1
    # remove empty column if needed
    if (skip is not None):
        L = np.delete(L,skip,axis=1)
        labels = np.delete(labels,skip)
    if (pandas):
        # convert back to DataFrame, using labels as column names
        L = pd.DataFrame(L,columns=labels[:n])
    return L

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
