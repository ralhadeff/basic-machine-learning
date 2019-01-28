"""
rbf kernel wrapper for the svm.py SVC
"""

import numpy as np
import math

import svm

class SvmRbf():

    def fit(self,X,y,gamma=1,landmarks=None,learning_rate=1,epochs=1000,C=1):
        """
        Fits the data to the SVM using an rbf kernel
        gamma (1/sigma**2) can be speficied by user (default=1)
        the number of landmarks can also be specified by user (default is len(X))
            if the number of landmarks is specified, the landmarks are randomly chosen from X
        Other parameters for the SVM can also be specified here (and are passed to the linear SVM)
        """
        self.gamma = gamma
        # remove row on constants 1 if present (saves memory and a tiny bit of time)
        if (X[0,0]==1 and len(np.unique(X[:,0]))==1):
            X = X[:,1:]
        # generate and store landmarks
        if (landmarks==None or landmarks>=len(X)):
            self.landmarks = X
        else:
            self.landmarks = X[np.random.choice(np.arange(len(X),dtype=int),landmarks,replace=False)]
        # generate matrix of features
        X = self.transform(X)
        # fit to a linear SVM
        self.svm = svm.SVM()
        # no x0 appears to perform better
        self.svm.fit(X, y, learning_rate, epochs,C,add_x0=False)

    def transform(self,X):
        """
        Transform data to higher dimension using rbf kernel with the pre-saved landmarks from the fitting
        """
        n = len(X)
        matrix = np.zeros((n,len(self.landmarks)))
        for i in range(n):
            for j in range(len(self.landmarks)):
                matrix[i][j] = math.exp(-self.gamma*((X[i]-self.landmarks[j])**2).sum())
        return matrix

    def predict(self,X):
        """
        Predict labels of given dataset
        """
        if (X[0,0]==1 and len(np.unique(X[:,0]))==1):
            X = X[:,1:]
        X = self.transform(X)
        return self.svm.predict(X)

    def score(self,X,y):
        """
        Predict and score the given data and labels
        """
        if (X[0,0]==1 and len(np.unique(X[:,0]))==1):
            X = X[:,1:]
        X = self.transform(X)
        return self.svm.score(X,y)

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
