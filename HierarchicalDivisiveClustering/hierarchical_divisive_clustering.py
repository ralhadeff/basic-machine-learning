'''
Naive implementation of hierarchical divisive clustering
'''

import numpy as np

class HierarchicalDivisive():

    @staticmethod
    def find_farthest(X):
        '''Find the point that is the farthest from all other points (using Euclidean distance)'''
        max_index = -1
        max_distance = 0
        for i in range(len(X)):
            # calculate the mean without the current point
            mean = np.delete(X,i).mean(axis=0)
            # calculate distance of current point from mean
            distance = ((X[i]-mean)**2).sum()
            if (distance>max_distance):
                # farthest so far
                max_distance = distance
                max_index = i
        return max_index

    @staticmethod
    def find_cluster(X,p):
        '''
        Find all points in X that are closer to p than to the remainder of the points in X
            User can specify how many iterations back and forth are desired (to refine the cluster)
        '''
        # new cluster
        cluster = [p]
        for i in range(len(X)):
            # calculate cluster centers
            mean_cluster = X[cluster].mean(axis=0)        
            mean_remainder = np.delete(X,[cluster+[i]],axis=0).mean(axis=0)
            # calculate distances to clusters
            d_cluster = ((X[i]-mean_cluster)**2).sum()
            d_remainder = ((X[i]-mean_remainder)**2).sum()
            if (d_cluster<d_remainder):
                # point is closer to cluster
                cluster.append(i)
        # remove p (which was added a second time in the loop)
        cluster.pop(0)
        return np.array(cluster)

    def fit(self,X,k):
        '''
        Splits data into k clusters
        '''
        clusters = Tree(X)
        while (clusters.count()<k):
            # find biggest cluster tree and get the data
            c = clusters.get_largest()
            x = c.left
            p = self.find_farthest(x)
            new_cluster = self.find_cluster(x,p)
            c.split(x[new_cluster],np.delete(x,new_cluster,axis=0))          
        self.clusters = clusters
    
    def get_tree(self):
        '''Return the hierarchical tree of clusters'''
        return self.clusters
    
    def get_clusters(self):
        '''Return a list of the clusters (hierarchy is lost)'''
        return self.clusters.get_all()
    
class Tree():
    '''Simple tree to sort hierarchy'''

    def __init__(self,left,right=None):
        '''Define left and right branch'''
        self.left = left
        self.right = right

    def count(self):
        '''Count the total number of clusters in this tree'''
        if (self.right is None):
            return 1
        else:
            return self.left.count() + self.right.count()

    def split(self,left,right):
        '''Splits current tree, but transforming each branch to a new tree'''
        test = Tree(None,None)
        self.left = Tree(left)
        self.right = Tree(right)

    def get_largest(self):
        '''Return the branch (a Tree) with the largest cluster in it'''
        if (self.right is None):
            return self
        else:
            left = self.left.get_largest()
            right = self.right.get_largest()
            if (len(left)>len(right)):
                return left
            else:
                return right

    def get_all(self):
        '''Return a list of all the clusters in this tree'''
        if (self.right is None):
            return [self.left]
        else:
            ls = self.left.get_all()
            ls.extend(self.right.get_all())
            return ls
    
    def get_data(self):
        '''Return all points nested under this branch'''
        clusters = self.get_all()
        cluster = clusters.pop(0)
        for c in clusters:
            cluster = np.concatenate((cluster,c))
        return cluster

    def __len__(self):
        '''Return the size of the largest array nested within this tree'''
        if (self.right is None):
            return len(self.left)
        else:
            left = len(self.left)
            right = len(self.right)
            return max(left,right)
        
if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
