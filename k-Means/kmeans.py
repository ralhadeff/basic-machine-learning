'''
Implementation of k-means clustering, including kmeans++ centroid initialization
'''

import numpy as np

class KMeans():
    '''
    Implementation of k-means clustering
    Will find k clusters in the data based on Euclidean distance
    '''
    
    def __init__(self, init='kmeans++'):
        """
        Construct a k-means clusterizer, using initialization function as specified
        kmeans++ is the default, otherwise 'random' can be specified
        """
        self.init = init
        self.centroids = None
    
    def fit(self,X,k):
        """
        Locates k clusters (centroids) in X
        using random initialization
        can also use kmeans++ if specified
        """

        if (self.init=='random'):
            # draw k starting points
            self.centroids = np.random.choice(np.arange(X.shape[0]),k,replace=False)
            self.centroids = X[self.centroids]
        elif (self.init=='kmeans++'):
            # draw first centroid, all others are at infinity
            self.centroids = np.ones((k,X.shape[1]))*float('inf')
            self.centroids[0] = X[np.random.choice(np.arange(X.shape[0]),1)]
            for i in range(1,k):
                # build distance matrix and sort it by distance^2
                # note that distances are already squared
                d = self.predict(X,True)
                d = d[d[:,0].argsort()]
                # build cummulative sum array
                cum = np.cumsum(d[:,0],axis=0)
                # rebuild distance matrix to be an inverted cummulative sum
                d = np.flip(np.concatenate((d[:,1],cum)).reshape((2,len(cum))).T)
                # draw random number, from 0 to the sum of all distances
                rand = np.random.uniform()*(d[0,0])
                # randomly pick point, weighted by distance squared
                for j in d:
                    # iterate until random is smaller than the current cummulative sum
                    if (rand<j[0]):
                        # add new centroid and exit loop
                        self.centroids[i] = X[int(j[1])]
                        break
        else:
            raise ValueError('Unrecognized initialization method')
            
        # initialize labels
        y = np.zeros((X.shape[0],),dtype=int)
        # iterate until converging
        while(True):
            # mark previous labels
            previous_k = y
            # re-label dataset
            y = self.predict(X)
            # check if there was any change
            if ((previous_k==y).all()):
                break
            # update centroids
            new_ks = np.zeros(self.centroids.shape)
            count = np.zeros(len(self.centroids))
            for x in range(len(X)):
                k = y[x]
                new_ks[k]+=X[x]
                count[k]+=1
            for k in range(len(new_ks)):
                new_ks[k]/=count[k]
            self.centroids = new_ks
    
    def predict(self,X,distances=False):
        """
        Label dataset X based on current centroids
        Should be used after fitting
        
        distances can be used to compute the distance to the closest cluster rather than the cluster index
        """
        if (distances):
            # distance and index. Index is necessary because the array will later be sorted
            y = np.zeros((X.shape[0],2),dtype=float)
        else:
            y = np.zeros((X.shape[0],),dtype=int)            
        for x in range(len(X)):
            d_max = float('inf')
            # find closest centroid
            for k in range(len(self.centroids)):
                # note that distances are squared
                d = np.sum((X[x]-self.centroids[k])**2)
                if (d<d_max):
                    d_max = d
                    if (distances):
                        y[x,1]=x
                        y[x,0]=d
                    else:
                        y[x]=k
        return y
    
    def wcss(self,X):
        """
        Calculate within cluster sum of squares, also called intertia
        """
        y = self.predict(X)
        wcss=0
        for x in range(len(X)):
            wcss+=np.sum((X[x]-self.centroids[y[x]])**2)
        return wcss
    
    def iterate(self,X,k,n):
        """
        Repeat the fitting process on for k cluster on X dataset n-times
            pick the fit with the lowest wcss
        """
        centroids = []
        wcss = []
        # run n fits
        for i in range(n):
            self.fit(X,k)
            # save results
            centroids.append(self.centroids)
            wcss.append(self.wcss(X))
        # find lowest wcss and take those centroids
        i = wcss.index(min(wcss))
        self.centroids = centroids[i]

def elbow(X,kmeans,k_max,n=5):
    """
    Elbow method plot generator
    Calculates wcss for each clustering from 1 to k_max
    X is the data, kmeans is the estimator
    n is an optional argument for how many trials per k value should be attempted, to find the best clustering
    """
    x = []
    y = []
    # fit and calculate wcss (k_max included
    for i in range(1,k_max+1):
        x.append(i)
        kmeans.iterate(X,i,n)
        y.append(kmeans.wcss(X))
    
    return x,y

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
