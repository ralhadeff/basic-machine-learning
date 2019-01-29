import numpy as np
from collections import Counter

class DecisionTree():
    
    def __init__(self,max_depth):
        self.max_depth = max_depth

    def fit(self,X,y):
        self.head = Branch(self.max_depth)
        self.head.split(X,y)
        
    def predict(self,X):
        y = np.zeros(len(X))
        i = 0
        for x in X:
            c = self.head.get_label(x)
            y[i] = c
            i+=1
        return y

class Branch():
    
    def __init__(self,depth):
        self.left = None
        self.right = None
        self.depth = depth
    
    def split(self,X,y):
        # find best split
        self.criterion = find_best_split(X,y)
        # generate left branch data
        Xle = X[X[:,self.criterion[0]]<=self.criterion[1]]
        self.yle = y[X[:,self.criterion[0]]<=self.criterion[1]]
        # generate right branch data
        Xgt = X[X[:,self.criterion[0]]>self.criterion[1]]
        self.ygt = y[X[:,self.criterion[0]]>self.criterion[1]]
        
        # max depth reached
        if (self.depth==0):
            return
        else:
            # check if a left branch needs further splitting:
            if gini(self.yle)>0:
                self.left = Branch(self.depth-1)
                self.left.split(Xle,self.yle)
            # check if right branch needs further splitting:
            if gini(self.ygt)>0:
                self.right = Branch(self.depth-1)
                self.right.split(Xgt,self.ygt)
    
    def get_label(self,x):
        # compare to criterion
        if (x[self.criterion[0]]<=self.criterion[1]):
            if (self.left is None):
                return Counter(self.yle).most_common(1)[0][0]
            else:
                return self.left.get_label(x)
        else:
            if (self.right is None):
                return Counter(self.ygt).most_common(1)[0][0]
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
            if (g<best[2] and len(y[X[:,c]<=x])>=min_group_size and len(y[X[:,c]>x])>=min_group_size):
                best = (c,x,g)
    return best[0:2]

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
