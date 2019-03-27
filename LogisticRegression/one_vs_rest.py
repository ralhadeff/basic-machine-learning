'''
One-vs-Rest implementation using n Logistic_regressors
Treat the ovr object like a regressor, it generates, trains and predicts using n regressors behind the scenes
'''

import numpy as np
import logistic_regressor


class OvR(object):
    
    def __init__(self):
        # use dictionary where the key will be the flag, and the value will be a regressor
        self.regressors = {}
        # keep track of flags
        self.unique = np.asarray([])
    
    def fit(self,X,y,add_x0=True,method='stochastic',learning_rate=0.0001,
            epochs=100,starting_coeff=False,bin_size=10):
        '''
        Fits the data into n logistic regressors, using the same parameters provided here
        starting coefficients set to True will iterate all regressors using each's own current coefficients
        '''
        
        # determine how many regressors are needed
        self.unique = np.unique(y)
        # generate regressors and train each on a modified training set
        for key in self.unique:
            y_current = np.vectorize(lambda x: 1 if x==key else 0)(y)
            self.regressors[key] = logistic_regressor.LogisticRegressor()
            # if requested, give each regressor its own current coefficients
            if (starting_coeff):
                init = self.regressors[key].coeff
            else:
                init = None
            self.regressors[key].fit(X,y_current,add_x0,method,learning_rate,epochs,init,bin_size)

    def predict(self,X):
        # save predictions in a dictionary
        predictions = {}
        # get predictions one regressor at a time
        for key in self.unique:
            predictions[key] = self.regressors[key].predict(X,True)
        # construct predictions array, starting with a dummy column
        pred = np.empty([X.shape[0],1])
        # keep track of index for each key
        keys = []
        for key in predictions:
            pred = np.hstack([pred,predictions[key][:,None]])
            keys.append(key)
        # remove dummy column
        pred = pred[:,1:]
        # extract the highest prediction in each row
        argmax = np.argmax(pred,axis=1)
        # convert index to label
        argmax = np.vectorize(lambda x: keys[x])(argmax)
        return argmax

if (__name__ == '__main__'):
    print('This module is not intended to run by iself')
