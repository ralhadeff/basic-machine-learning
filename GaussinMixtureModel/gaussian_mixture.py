'''Gaussian mixture model clustering'''

import numpy as np

class GaussianMixtureClustering():
    
    def draw_gaussians(self,X,k):
        '''Randomly generate k gaussians'''
        # mus are within the range of X values
        mus = np.random.uniform(X.min(axis=0),X.max(axis=0),size=(k,X.shape[1]))
        # sigmas start as identity matrices ('perfect gaussian')
        sigmas = np.array([np.identity(X.shape[1]) for s in range(k)])
        # priors start uniform
        priors = np.ones(k)/k
        return (mus, sigmas, priors)
    
    def iterate(self,X,k,mus,sigmas,priors):
        p = np.zeros(shape=(len(X),k))
        # probability to belong to cluster (w bayes rule)
        c = np.zeros(shape=(len(X),k))      
        # weight (importance) of point to each cluster
        w = np.zeros(shape=(len(X),k))
        for x in range(len(X)):
            for g in range(k):
                sample = X[x]
                gaussian = (mus[g],sigmas[g])
                p[x,g] = calculate_p(sample,gaussian)
            # iterate a second time for bayes rule
            for g in range(k):
                c[x,g] = p[x,g]*priors[g]/(p[x]*priors).sum()
        # calculate weights
        w = c/c.sum(axis=0)
        return (w,c)
    
    def fit(self,X,k,max_iterations=100):
        '''
        Cluster the given data X into k gaussians
            User can define the maximum number of iterations, in case convergence is extremely slow
            Otherwise, the clustering will stop as soon as all points are stabaly assigned
        '''
        # generate initial gaussians
        mus, sigmas, priors = self.draw_gaussians(X,k)

        y = np.zeros(len(X))
        # run iterations, refining gaussians:
        for i in range(max_iterations):
            # get weights and probabilities
            # p is probabilities to belong to cluster
            w,p = self.iterate(X,k,mus, sigmas, priors)
            # update gaussians
            d = X.shape[1]
            mus = np.zeros((k,d))
            for g in range(k):
                for x in range(len(X)):
                    # weighted sum of data
                    mus[g]+=X[x]*w[x,g]
            sigmas = np.zeros((k,d,d))
            for g in range(k):
                for a in range(sigmas.shape[1]):
                    for b in range(sigmas.shape[2]):
                        total = 0
                        for x in range(len(X)):
                            total+= w[x,g]*(X[x,a]-mus[g,a])*(X[x,b]-mus[g,b])
                        sigmas[g,a,b] = total
            # update priors (p(cluster))
            priors = np.zeros(k)
            for g in range(k):
                priors[g] = p[:,g].mean()
            # assign points to clusters (for checking convergence):
            n_y = np.argmax(p,axis=1)
            if (y==n_y).all():
                break
            else:
                y = n_y
        # save gaussians for predictions
        self.mus = mus
        self.sigmas = sigmas
        self.priors = priors
        
    def predict(self,X):
        '''Assign clusters to the data'''
        _,p = self.iterate(X,len(self.priors),self.mus, self.sigmas, self.priors)
        return np.argmax(p,axis=1)

# note - taken from AnomalyDetection
def calculate_p(x,gaussian):
    '''
    Calculate the probability x belongs to a gaussian with mu and sigma parameters
    '''
    mu,sigma = gaussian
    # multivariate, meaning there is one multidimensional Gaussian for all features
    n = len(sigma)
    det = np.linalg.det(sigma)
    return 1/((2*np.pi)**(n//2)*det)*np.exp(
        -((x-mu)[:,None].T @ np.linalg.inv(sigma) @ (x-mu)[:,None])/2) 

if (__name__ == '__main__'):
    print('This module is not intended to run by iself')
