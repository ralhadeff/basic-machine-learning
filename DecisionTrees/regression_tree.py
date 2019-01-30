"""
Decision tree regressor
"""

import numpy as np
from collections import Counter

from decision_tree import DecisionTree, Branch

class DecisionTreeRegressor(DecisionTree):
    """
    A decision tree regressor
    """
    
    def fit(self,X,y):
        """Find the best splits recursively until the limit is reached"""
        if (self.max_features=='all' or self.max_features>=X.shape[1]):
            skip = []
        else:
            skip = np.random.choice(np.arange(X.shape[1]),X.shape[1]-self.max_features)
        # first branch
        self.head = RBranch(self.max_depth)
        self.head.reg_split(X,y,skip)
        
    def predict(self,X):
        """Predict the values of y for the given X"""       
        # setup array for predictions
        y = np.zeros(len(X))
        # external counter is faster
        i = 0
        for x in X:
            # predict recursively for each point
            y[i] = self.head.get_value(x)
            i+=1
        return y
    
    def score(self,X,real_y):
        """Predict and give a score for the current data and true values"""
        # default is mse
        pred = self.predict(X)
        return ((real_y-pred)**2).mean()
    
class RBranch(Branch):
    """
    A modified branch object for the decision tree regressor
    """
    
    def reg_split(self,X,y,skip=[]):
        """
        Splits the data and generates new branches under current branch as necessary
        """
        # find best split (stdev not required)
        self.criterion = find_reg_split(X,y,skip=skip)[:2]
        if (self.criterion[0] == -1):
            # no split was found, this node should act as if it is null
            # it will return the value of y unsplit (same as parent branch, if this node hadn't been created
            self.yle = y.mean()
            self.ygt = self.yle
        else:
            # generate left branch data
            Xle = X[X[:,self.criterion[0]]<=self.criterion[1]]
            yle = y[X[:,self.criterion[0]]<=self.criterion[1]]
            # generate right branch data
            Xgt = X[X[:,self.criterion[0]]>self.criterion[1]]
            ygt = y[X[:,self.criterion[0]]>self.criterion[1]]
        
            # pass data to new branches and keep splitting until reaching max_depth 
            if (self.depth>0):
                self.left = RBranch(self.depth-1)
                self.left.reg_split(Xle,yle,skip)
            if (self.depth>0):
                self.right = RBranch(self.depth-1)
                self.right.reg_split(Xgt,ygt,skip)
        
            # save label information for predictions later
            if (self.left is None):
                if (len(yle)>0):
                    self.yle = yle.mean()
            if (self.right is None):
                if (len(ygt)>0):
                    self.ygt = ygt.mean()
        
    def get_value(self,x):
        """Same as get_label"""
        return Branch.get_label(self,x)
    
def find_reg_split(X,y,min_group_size=2,skip=[]):
    """
    Finds the best split in X in terms of lowest stdev after the split
    """
    # best (column, <=value, stdev)
    # starting values
    best = (-1,0,float('inf'))
    n_total = len(X)
    for c in range(X.shape[1]):
        for x in X[:,c]:
            # skip rows as requested (for random forest)
            if (c not in skip):
                # try a split, using the weighted mse (=var)
                mask = X[:,c]<=x 
                g = 0
                if (len(y[mask])>0):
                    g += y[mask].var() * len(y[mask])/len(y)
                if (len(y[mask==False])>0):
                    g += y[mask==False].var() * len(y[mask==False])/len(y)
                # make a split only if the smallest group is at least min_group_size, except for score of 0.0
                if ((g<best[2] and len(y[X[:,c]<=x])>=min_group_size and len(y[X[:,c]>x])>=min_group_size)):
                    best = (c,x,g)
    return best

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
