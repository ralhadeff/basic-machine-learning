'''
An anomaly detection tool
'''

import numpy as np

class AnomalyDetector():
    
    def __init__(self):
        '''
        An anomaly detector, using univariate or multivariate Gaussians to determine outliers
        '''
        pass
    
    def fit(self,X):
        """
        Provide 'good' data to the detector to establish what are the ranges
            of values that are acceptable. It is ok if the data has some outliers
        """
        self.mu = X.mean(axis=0)
        self.var = X.var(axis=0)
        self.sigma = np.cov(X.T)
    
    def detect_anomalous(self,X,epsilon,variate='univariate'):
        """
        Given a dataset, return the index of all anomalous points within the set
        """
        p = self.calculate_p(X,variate)
        return np.nonzero(p<=epsilon)           
    
    def calculate_p(self,X,variate='univariate'):
        """
        Find the p values of all points in a dataset
        """
        if (variate=='univariate' or variate=='uni'):
            # univariate, meaning each feature has its own p measured separately
            # calculate individual p(x)
            p = (1/np.sqrt(2*np.pi*self.var)) * np.exp(-(X-self.mu)**2 / (2*self.var))
            # calculate the multiplication of all p's to get p(X)
            if (np.ndim(p)>1):
                p = np.prod(p,axis=1)
            else:
                p = np.prod(p)
            return p
        if (variate=='multivariate' or variate=='multi'):
            # multivariate, meaning there is one multidimensional Gaussian for all features
            n = len(self.sigma)
            det = np.linalg.det(self.sigma)
            x_mu = X-self.mu
            if (np.ndim(X)>1):
                # initialize array
                p = np.zeros(len(X))
                for i in range(len(X)):
                    p[i] = 1/((2*np.pi)**(n//2)*det) * np.exp(-(x_mu[i].T @ np.linalg.inv(self.sigma) @ x_mu[i])/2) 
            else:
                p = 1/((2*np.pi)**(n//2)*det) * np.exp(-(x_mu[:,None].T @ np.linalg.inv(self.sigma) @ x_mu[:,None])/2) 
            return p
        
    def suggest_epsilon(self,x,variate='univariate'):
        """
        Provided with a known anomaly (or multiple anomalies), return the lowest epsilon value
            that would detect them all.
            This can serve as a guideline for future detections
                user is advised to increase the given epsilon slightly, to be on the safe side
        """
        p = self.calculate_p(x,variate)
        return p.max()   

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
