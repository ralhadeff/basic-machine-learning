"""
This is a simple linear regression that can use the normal equation, batch gradient descent or stochastic gradient decent
It uses a simple learning algorithm, with a fairly primitive self correcting learning rate,

Note - because it is simplistic, feature standartization (or normalization) is crucial. Also, convergence is very slow
"""

import numpy as np
import pandas as pd

class LinearRegressor(object):

    def __init__(self):
        self.coeff=None
    
    def fit(self,X,y,add_x0=True,method='normal',learning_rate=0.0001,epochs=100,starting_coeff=None,tolerance=
                         0.01, bin_size=10):
        """
        This is a simple linear regressor
        it uses the normal equation, batch gradient decent, or stochastic gradient decent to find the coeffcients
        it also automatically adds the first column x0 (all 1's) if it is missing

        if using batch gradient descent, a learning rate and number of ephochs can be provided
           these have no effect for normal equation
        also, starting coefficients can be provided
        when doing gradient decent, learning rate will gradually decrease if diverging
        if loss function improvement is smaller than tolance (in percentages) learning rate will increase

        if using stochastic gradient descent, same as batch, but with bins instead of the whole set

        """
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
        if (method=='normal'):
            # normal equation
            self.coeff = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.transpose(),x)),x.transpose()),y)
        else:
            # batch is same as mini-batch but the size is the total number of samples
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
                        h = np.dot(self.coeff,xs)
                        # derivation
                        dc+= (h-y_real)*xs
                        # error function (MSE)
                        error+= (h-y_real)**2
                    # update coefficients
                    self.coeff-= dc * (learning_rate/n)
                # end of batch/mini-batch loop
                # self correcting learning rate only for batch - in mini-batch MSE is not always going down on each iteration 
                if (method=='batch'):
                    # calculate total error and compare to previous error
                    if (error<previous_error):
                        # improvement
                        if (previous_error/error-1<tolerance):
                            # improvement is very small, increase learning rate
                            learning_rate *= 1.1
                    else:
                        # diverging
                        learning_rate/=2
                    previous_error = error
            # end of epochs loop

    def predict(self,X):
        """
        Predict and returns the values h for all samples provided
        """
        # convert dataframe to ndarray if needed
        if (type(X) is pd.DataFrame):
            X = X.values
        # add 1's as the first column if needed
        # here simply check if there is only one column missing
        if (X.shape[1]==self.coeff.shape[0]-1):
            n = X.shape[0]
            x0 = np.ones((n,1))
            X = np.hstack((x0,X))
        return np.squeeze(np.asarray(np.matmul(X,self.coeff)))

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
