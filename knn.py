"""
k-nearest neighbors classifier, using Euclidean distances
"""

import numpy as np
try:
    import lr_metrics
except:
    pass

class KNN():
    """
    k-nearest neighbors classifier, based on Euclidean distances
    """
    
    def __init__(self):
        self.X = None
        self.y = None
        # default value for k
        self.k = 5
    
    def fit(self,X,y):
        """
        No actual fitting is done for this lazy implementation, saves the data for later use
        """
        self.X = X
        self.y = y
    
    def predict(self,X,k):
        """
        Label X dataset using k neighbors
        """
        if (k%2==0):
            print("Note: KNN is not advised using an even number of neighbors")
        
        y = np.zeros(len(X),int)
        # iterate and label input
        for i in range(len(X)):
            x = X[i]
            # distances array
            distances = ((self.X-x.T)**2).sum(axis=1)
            # draw indices of k closest neighbors
            i_neighbors = distances.argsort()[:k]
            votes = (self.y[i_neighbors]).astype(int)
            winner = np.argmax(np.bincount(votes))
            y[i] = winner
        # save k for future scoring
        self.k = k
        return y
    
    def score(self,X,y,k=None):
        """
        Return the score on the given test set
        using the last fitted k, or a newly given k
        """
        # set current k
        if (k==None):
            k = self.k
        # try to use metrics' accuracy
        try:
            # the reason this is better than the built-in function is in case I ever want to modify lr_metrics module
            return log_metrics.accuracy(y,self.predict(X,k))
        except:
            # default built-in metrics
            pred = self.predict(X,k)
            return (y==pred).mean()        

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
