"""
This tool splits a dataset into training and test sets
    can also split into a third group: validation
    input seed can be provided for reproducibility
"""

import numpy as np
import pandas as pd
import random

def split_dataset(X,y,test_ratio=0.33, validation_ratio=0,seed=42):
    """
    Randomly splits data into training set and test set. X and y can be dataframes or numpy arrays
    optional: splits the data 3 ways, including a validation set as well
    random seed can be provided for reproducibility
    """
    # set random seed
    random.seed(seed)
    
    # set to np arrays if needed
    if (type(X)==pd.DataFrame or type(X)==pd.Series):
        X = X.values
    if (type(y)==pd.Series):
        y = y.values
    
    # split training set
    train_ratio = 1-test_ratio-validation_ratio
    size = X.shape[0]
    if (not (0<train_ratio<1)):
        raise ValueError("Training set size out of bounds")
    train = random.sample(range(size),round(size*train_ratio))
    train = np.sort(train)
    # opposite indices
    test = np.ones(len(X),np.bool)
    test[train] = 0
        
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]
    if (validation_ratio!=0):
        # further split test into test and validation
        # this second seed is necessary, otherwise the next call of split_dataset will use the seed of 42 again
        next_seed = random.randint(0,1000)
        new_ratio = test_ratio/(test_ratio+validation_ratio)
        X_val,y_val, X_test, y_test = split_dataset(X_test,y_test,test_ratio=new_ratio,seed=next_seed)
        return (X_train,y_train,X_test,y_test,X_val,y_val)
    
    return (X_train,y_train,X_test,y_test)

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
