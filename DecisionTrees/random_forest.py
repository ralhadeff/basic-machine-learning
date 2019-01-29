"""
Random forest classifier implementation
"""

import numpy as np
import math
from collections import Counter

import decision_tree

class RandomForest():
    
    def __init__(self,n_trees=10,n_samples='all',n_features='auto'):
        """
        Define the number of trees to be used
            the number of sample to use for each tree (with bootstrapping)
            the number of features to use
        Defaults are 10, all samples (but with bootstrap), and sqrt(X.shape[1])
        """
        self.n_trees = n_trees
        self.n_samples = n_samples
        self.n_features = n_features
        
    def fit(self,X,y,max_depth=15):
        """
        Fit the data to all trees
        """
        # save labels
        self.labels = np.unique(y)
        # determine n_samples
        if (self.n_samples=='all' or self.n_samples>len(X)):
            self.n_samples = len(X)
        # determine n_features
        if (self.n_features=='auto'):
            self.n_features=int(math.sqrt(X.shape[1]))
        elif (self.n_features=='all' or self.n_features>X.shape[1]):
            self.n_features=X.shape[1]
        # QA
        if (self.n_samples<=0 or self.n_features<=0 or self.n_trees<2):
            raise ValueError('There is an error in your input values')
        
        # generate n trees and fit them
        self.trees = []
        for i in range(self.n_trees):
            mask = np.random.choice(np.arange(len(X)),self.n_samples,replace=True)
            tree = decision_tree.DecisionTree(max_depth,max_features=self.n_features)
            tree.fit(X[mask],y[mask])
            self.trees.append(tree)

    def predict(self,X):
        """
        Predict labels
        """
        # see ovo.py
        predictions = np.zeros((len(X),len(self.trees)),dtype=self.labels.dtype)
        i=0
        for tree in self.trees:
            predictions[:,i] = tree.predict(X)
            i+=1
        # count votes and retun winners
        winners = predictions[:,0]
        for i in range (len(predictions)):
            winners[i] = Counter(predictions[i]).most_common(1)[0][0]
        return winners
        
    def score(self,X,y):
        """Score data and true labels using total accuracy"""
        pred = self.predict(X)
        return (y==pred).mean()

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")        