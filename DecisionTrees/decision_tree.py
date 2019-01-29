"""
Decision tree classifier
"""

import numpy as np
from collections import Counter

class DecisionTree():
    """
    A decision tress classifier
    """
    
    def __init__(self,max_depth=15):
        """Maximum depth to avoid over-recursion"""
        self.max_depth = max_depth

    def fit(self,X,y):
        """Find the best splits recursively until the limit, or a prefect classification is reached"""
        # save labels and convert all labels to numbers
        self.labels = np.unique(y)
        # use separate array to avoid conflict if a label is e.g. '1'
        new_y = np.zeros(len(y),dtype=int)
        for i in range(len(self.labels)):
            new_y[y==self.labels[i]] = i
        y = new_y
        # first branch
        self.head = Branch(self.max_depth)
        self.head.split(X,y)
        
    def predict(self,X):
        """Predict the labels of give data"""
        # setup numerical labels vector
        y = np.zeros(len(X),dtype=int)
        # external counter is faster
        i = 0
        for x in X:
            # predict recursively for each point
            c = self.head.get_label(x)
            # mar label
            y[i] = c
            i+=1
        # convert numerical labels back to original labels
        predictions = np.zeros(len(y),dtype=self.labels.dtype)
        for i in range(len(self.labels)):
            predictions[y==i] = self.labels[i]        
        return predictions
    
    def score(self,X,y):
        """Predict and give a score for the current data and true labels"""
        # default is total accuracy
        pred = self.predict(X)
        return (y==pred).mean()
    
class Branch():
    """
    A branch object for the decision tree - with left and right branches, a splitting criterion, and labels of its own
    """
    
    def __init__(self,depth):
        # left branch is for the <= (le), right branch is for the > (gt)
        self.left = None
        self.right = None
        self.depth = depth
    
    def split(self,X,y):
        """
        Splits the data and generates new branches under current branch as necessary
        """
        # find best split (gini score is not required)
        self.criterion = find_best_split(X,y)[:2]
        # generate left branch data
        Xle = X[X[:,self.criterion[0]]<=self.criterion[1]]
        yle = y[X[:,self.criterion[0]]<=self.criterion[1]]
        # generate right branch data
        Xgt = X[X[:,self.criterion[0]]>self.criterion[1]]
        ygt = y[X[:,self.criterion[0]]>self.criterion[1]]
        
        # pass data to new branches
        # check if left branch needs more splitting
        if (gini(yle)>0):
            # if max_depth has been reach, only proceed it this is a terminal split
            if (self.depth>0 or find_best_split(Xle,yle)[2]==0):
                self.left = Branch(self.depth-1)
                self.left.split(Xle,yle)
        # same with right
        if (gini(ygt)>0):
            if (self.depth>0 or find_best_split(Xgt,ygt)[2]==0):
                self.right = Branch(self.depth-1)
                self.right.split(Xgt,ygt)
        
        # save label information for predictions later
        if (self.left is None):
            if (len(yle)>0):
                self.yle = Counter(yle).most_common(1)[0][0]
        if (self.right is None):
            if (len(ygt)>0):
                self.ygt = Counter(ygt).most_common(1)[0][0]
    
    def get_label(self,x):
        """Make a prediction, recursively looking for the correct leaf"""
        
        # compare to criterion
        if (x[self.criterion[0]]<=self.criterion[1]):
            if (self.left is None):
                return self.yle
            else:
                return self.left.get_label(x)
        else:
            if (self.right is None):
                return self.ygt
            else:
                return self.right.get_label(x)

def gini(y):
    """
    Calculate and return the gini index of the given array of labels
    """
    uniques,counts = np.unique(y,return_counts=True)
    return 1-((counts/len(y))**2).sum()

def find_best_split(X,y,min_group_size=2):
    """
    Finds the best split in X in terms of lowest gini index in y after the split
    """
    # best (column, <=value, gini)
    # starting values
    best = (0,0,1)
    n_total = len(X)
    for c in range(X.shape[1]):
        for x in X[:,c]:
            # try a split
            mask = X[:,c]<=x
            count = mask.sum()
            # weighted gini score
            g = gini(y[mask])*(count/n_total) + gini(y[mask==False])*((n_total-count)/n_total)
            # make a split only if the smallest group is at least min_group_size, except for score of 0.0
            if (g==0 or (g<best[2] and len(y[X[:,c]<=x])>=min_group_size and len(y[X[:,c]>x])>=min_group_size)):
                best = (c,x,g)
    return best

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
