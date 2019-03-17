'''
Simple logistic regressor that uses batch/mini-batch/stochastic gradient descent
'''

import pandas as pd
import numpy as np
# try to import metrics module
try:
    import lr_metrics
except:
    pass


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegressor(object):

    def __init__(self):
        self.coeff=None
    
    def fit(self,X,y,add_x0=True,method='stochastic',learning_rate=0.001,epochs=100,starting_coeff=None,bin_size=1):
        '''
        This is a simple logistic regressor
        it can use batch gradient decent, or stochastic gradient decent to find the coeffcients
        it also automatically adds the first column x0 (all 1's) if it is missing

        starting coefficients can be provided
        
        bin_size is set to all dataset for 'batch'
        '''
        # convert dataframe to ndarray
        if (type(X) is pd.DataFrame):
            X = X.values
        if (type(y) is pd.Series):
            y = y.values
        # if only using one feature:
        if (len(X.shape)==1):
            X.shape = (X.shape[0],1)
        # unless add_x0 is set to False, figure out whether a new column of 1's needs to be added
        if (add_x0) and (X[0,0]!=1 or len(np.unique(X[:,0]))>1):
            # n - number of samples, m - number of features (excluding the 1's)
            n,m = X.shape
            x0 = np.ones((n,1))
            x = np.hstack((x0,X))
            # update m
            m+=1
        else:
            n,m = X.shape
            x = X
        # perform regression, based on method
        if (method=='batch'):
            bin_size = n
            
        if (starting_coeff is None):
            # initialize coefficients as 0's
            self.coeff = np.zeros(m)
        else:
            # use provided coefficients as a starting point
            self.coeff = np.asarray(starting_coeff)
        # start descent
        previous_error = float('inf')
        for epoch in range(epochs):
            # accumulating derivatives
            dc = np.zeros(m)
            error = 0 
            # +1 in case n is not a natural multiplication of bin_size
            for batch in (range(int(n/bin_size)+1)):
                # run one batch
                for j in range(bin_size):
                    # index of sample from entire dataset
                    i=batch*bin_size+j
                    # full iteration through dataset done
                    if (i>=n):
                        break
                    # x values
                    xs = x[i]
                    # label
                    y_real = y[i]
                    # prediction
                    z = np.dot(self.coeff,xs)
                    h = sigmoid(z)
                    # derivation
                    dc+= (h-y_real)*xs
                # update coefficients
                self.coeff-= dc * (learning_rate/n)
            # end of batch/mini-batch loop
        # end of epochs loop

    def predict(self,X, probability=False, threshold=0.5):
        '''
        Predict and returns the values h for all samples provided
        probability is whether the user wants output as probabilities or labels
        threshold is the threshold for labeling '1', only applied if probability is set to False
        '''
        # convert dataframe to ndarray if needed
        if (type(X) is pd.DataFrame):
            X = X.values
        # add 1's as the first column if needed
        # here simply check if there is only one column missing
        if (X.shape[1]==self.coeff.shape[0]-1):
            n = X.shape[0]
            x0 = np.ones((n,1))
            X = np.hstack((x0,X))
        prob = np.vectorize(sigmoid)(np.squeeze(np.asarray(np.matmul(X,self.coeff))))
        if (probability):
            return prob
        else:
            return np.vectorize(lambda x: 1 if x>=threshold else 0)(prob)

    def score(self,X,y):
        '''
        Return the score on the given test set
        '''
        # try to use metrics' accuracy
        try:
            # the reason this is better than the built-in function is in case I ever want to modify lr_metrics module
            return log_metrics.accuracy(y,self.predict(X))
        except:
            # default built-in metrics
            pred = self.predict(X)
            return (y==pred).mean()

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
