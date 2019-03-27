'''
Naive Bayes (Gaussian) classifier
'''

import numpy as np
import math

class NaiveBayes():
    
    def fit(self,X,y):
        '''
        X samples and y labels to train on
            y labels can be of any type
        '''
        self.labels = np.unique(y)
        # initialize for first label
        mean = X[y==self.labels[0]].mean(axis=0)
        std = X[y==self.labels[0]].std(axis=0)
        # add all other labels
        for l in self.labels[1:]:
            mean = np.concatenate((mean,X[y==l].mean(axis=0)))
            std = np.concatenate((std,X[y==l].std(axis=0)))
        # reshape
        self.means = mean.reshape(len(self.labels),-1)
        self.stdevs = std.reshape(len(self.labels),-1)

    def predict(self,X):
        # initialize predictions array
        pred = np.zeros(len(X),dtype=self.labels.dtype)
        # calculate once
        c = math.sqrt(2*math.pi)
        for i in range(len(X)):
            # for each sample
            x = X[i]
            # calculate probabilities for each label
            probs = np.zeros(len(self.labels))
            for l in range(len(self.labels)):
                # probability for each feature (independent)
                exp = np.exp(-(x-self.means[l])**2 / (2*(self.stdevs[l]**2)))
                p = exp / (c * self.stdevs[l])
                # overall probability
                probs[l] = np.prod(p)
            # predict the most probable label
            pred[i] = self.labels[np.argmax(probs)]
        return pred
    
    def score(self,X,y):
        '''Return the total accuracty for given X and true y labels'''
        # get predictions
        return (self.predict(X) == y).mean()
    
if (__name__ == '__main__'):
    print('This module is not intended to run by iself')
