"""
Generic One-vs-One implementation
Treat the OvO object like a regressor. Internally, it generates, trains, and predicts 
    using k(k-1)/2 regressors.
"""

import numpy as np
import inspect
from collections import Counter

class OvO():
    
    def fit(self,estimator,X,y,args=None):
        """
        Fits the data into k(k-1)/2 estimators of the provided type, using the args parameter dictionary
        Estimator should be the class of the estimator that is desired (not an instance)
        """   
        # estimator must be the class itself, and not an instance
        # if an instance is given, print a warning
        if (not inspect.isclass(estimator)):
            print('Warning: estimator was given as instance, not class.')
            print('New instances of the class will be generated, but the estimator itself will be ignored')
            estimator = estimator.__class__
        # determine how many regressors are needed
        self.labels = np.unique(y)
        # generate list of unique tuples
        self.tuples = []
        for i in range(len(self.labels)):
            for j in range(i+1,len(self.labels)):
                self.tuples.append((self.labels[i],self.labels[j]))
        # dict of regressors
        self.regressors = {}
        for pair in self.tuples:
            # split and join data
            x_c = np.concatenate((X[y==pair[0]],X[y==pair[1]]),axis=0)
            y_c = np.concatenate((y[y==pair[0]],y[y==pair[1]]))
            # generate a regressor 
            self.regressors[pair] = estimator()
            # add a pointer to the same regression for the inverted pair
            # this will prevent crashes in case any pair is provided in the wrong order
            self.regressors[pair[::-1]] = self.regressors[pair]
            # train the regressor
            if (args is None):
                self.regressors[pair].fit(x_c,y_c)
            else:
                self.regressors[pair].fit(x_c,y_c,**args)                

    def predict(self,X):
        """
        Predicts the labels for the given data (using internally saved labels)
        """        
        # save predictions in an array
        predictions = np.zeros((len(X),len(self.tuples)),dtype=self.labels.dtype)
        # get predictions one regressor at a time
        i=0
        for pair in self.tuples:
            predictions[:,i] = self.regressors[pair].predict(X)
            i+=1
        # count votes and retun winners
        winners = predictions[:,0]
        for i in range (len(predictions)):
            winners[i] = Counter(predictions[i]).most_common(1)[0][0]
        return winners
    
    def score(self,X,true_y,score=None):
        """
        
        """
        pred = self.predict(X)
        if (score is not None):
            return score(true_y,pred)
        else:
            # default is accuracy
            return (pred==true_y).mean()
            
if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
