"""
Support vector machine implmentation (Pegasos version)
"""
# currently using only the 'linear kernel' and only binary classification

import numpy as np
try:
    import lr_metrics
except:
    pass

class SVM():
    
    def __init__(self):
        self.labels=None
 
    def fit(self,X,y,learning_rate=1,epochs=1000,C=1,add_x0=True):
        """
        Fit the SVM using data X and labels y (1,-1)
        Learning rate and regularization parameter C can be specified
        Number of epochs can be specified
        
        Note: regularization decays as 1/epoch
        """
        # read labels, change y to -1,1 and save labels for predictions later
        labels = np.unique(y)
        if (labels[0]!=-1 or labels[1]!=1):
            if (len(labels)!=2):
                raise ValueError('Error in label array (too many or too few unique labels)')
            else:
                # save labels for later
                self.labels = labels
                # update y
                y = np.vectorize({labels[0]:-1,labels[1]:1}.get)(y)
        
        # add constant to X if necessary (first column)
        if (add_x0) and (X[0,0]!=1 or len(np.unique(X[:,0]))>1):
            X = np.concatenate((np.ones((len(X),1)),X),1)

        # initialize weights vector
        w = np.zeros(X.shape[1])
        
        # run SGD
        for epoch in range(1,epochs):
            # shuffle data, using a mask (shuffle indices)
            mask = np.random.permutation(len(X))
            X = X[mask]
            y = y[mask]
            # external index is fastest
            i=0
            for x in X:
                if ((y[i]*np.dot(w,x))<=1):
                    # for mismatch, update weights
                    w-=learning_rate*(w/epoch-C*x*y[i])
                else:
                    # otherwise, do only regulaization
                    # note, regularization doesn't affect the line in itself at all
                    # the effect is by making the weights smaller, and the effect of updates on mismatches bigger
                    w*=(1-learning_rate/epoch)
                i+=1
        
        # save weights
        self.weights = w
    
    def predict(self,X):
        """
        Predict the labels on a given dataset
        """
        # add 1 to X if needed (only check if one column is missing, and assume that it is the constants)
        if (X.shape[1]==len(self.weights)-1):
            X = np.concatenate((np.ones((len(X),1)),X),1)
        # predict using -1 and 1 as output
        pred = np.dot(X,self.weights)
        pred[pred>=0]=1
        pred[pred<0]=-1
        # convert predictions to the labels
        if (self.labels is not None):
            pred = np.vectorize({-1:self.labels[0],1:self.labels[1]}.get)(pred)
        return pred
        
    def score(self,X,y):
        """
        Return the score on the given test set
        """
        # same as LogisticRegressor
        try:
            return log_metrics.accuracy(y,self.predict(X))
        except:
            pred = self.predict(X)
            return (y==pred).mean()

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
