'''
Principle component analysis, including linear algebra tools necessary to perform it
Specifically: calculating covariance, the covariance matrix, eigenvectors and eigenvalues
'''

import numpy as np
import math

class PCA():
    '''
    Principle component analysis (PCA) solver
    '''

    def __init__(self):
        self.vectors=None
        self.values=None
        self.means=None
        self.k=None

    def fit(self,data):
        '''
        finds the eigenvectors and eigenvalues
        '''
        # save means for later
        self.means = data.mean(axis=0)
        # move data to origin
        A = data-self.means
        # get covariance matrix
        cov = covariance_matrix(A)
        vectors = []
        values = []
        for i in range(len(cov)):
            v = power_iteration(cov,eigens=vectors)
            vectors.append(v)
            values.append(eigen_value(cov,v))
        self.values = np.asarray(values)
        self.vectors = np.asarray(vectors)

    def reduce_dimensionality(self,data,k):
        '''
        Transform given data, reducing it to k dimensions,
        using eigenvectors previously computed
        '''
        # move data to origin
        A = data-self.means
        W = self.vectors[:k]
        self.k=k
        return np.matmul(W,A.T).T

    def restore_dimensionality(self,data):
        '''
        Transform give data, increasing it back to n dimensions, reversing the reduction of the PCA
        '''
        return np.dot(data, self.vectors[:self.k]) + self.means

def covar(x,y):
    '''Calculate the covariance between two 1-d arrays'''
    mx = x.mean()
    my = y.mean()
    total = 0
    for i in range(len(x)):
        total+=(x[i]-mx)*(y[i]-my)
    return total/(len(x)-1)

def covariance_matrix(M):
    '''Calculate the covariance matrix of a dataset'''
    features = M.shape[1]
    output = np.zeros((features,features))
    for i in range(features):
        for j in range(features):
            output[i][j]=(covar(M[:,i],M[:,j]))
    return output

# inspited by reading http://madrury.github.io/jekyll/update/statistics/2017/10/04/qr-algorithm.html
def power_iteration(A, eigens=None, tol=0.000001):
    '''
    The power iteration method to find principle components of give matrix A,
    A list of previously found eigenvectors can be provided to find the orthogonal eigenvectors
    The method outputs one eigenvector at a time, in eigenvalue descending order
    The tolerance for convergence can be specified by the user

    This code was heavily inspired by the fine explanations provided in the following page:
    http://madrury.github.io/jekyll/update/statistics/2017/10/04/qr-algorithm.html
    '''
    # if previously found eigenvectors not specified, start with a random vector
    if (eigens is None or len(eigens)==0):
        v = np.random.normal(size=len(A))
    else:
        # start with an orthogonal vector
        v = find_ortho(eigens)
    # make it a unit vector
    v /= math.sqrt(np.square(v).sum())
    while True:
        # save previous result for comparison
        previous = v
        # multiply by A (cummulative)
        v = np.matmul(A,v)
        # normalize again
        v /= math.sqrt(np.square(v).sum())
        # orthogonalize if provides with previous eigenvectors
        if (eigens is not None and len(eigens)>0):
            v = find_ortho(eigens,v)
        if np.abs(v-previous).sum()<tol:
            break
    return v

def find_ortho(v, start=None):
    '''
    Find an arbitrary vector that is perpendicular to the give vector/s in v
    A starting vector can be give to orthoganlize it. 
    This is necessary in case some tranformation, due to machine precision, breaks orthogonality
    '''
    # determine how many vectors are in v
    if (isinstance(v,list)):
        orthos = len(v)
    else:
        orthos = 1
        # for consistency
        v = [v]
    # quality control
    if (orthos>=len(v[0])):
        raise ValueError("Too many vectors provided")
    if (start is None):
        #start with vector [1,1,1,1,1,u]
        u = np.ones(len(v[0]))
    else:
        u = start
    # randomly choose n elements to adjust
    elements = np.sort(np.random.choice(np.arange(len(v[0])),size=orthos,replace=False))
    # mask trick to get the negative of elements
    mask = np.ones(len(v[0]),dtype=bool)
    mask[elements]=0
    b_elements = np.arange(len(v[0]))[mask]
    # solve equations using a matrix where V is the matrix of vectors in orthos and b is the fixed values
    V = np.concatenate(v).reshape((orthos,len(v[0])))
    b = -(V[:,b_elements] * u[b_elements]).sum(axis=1)
    # find solution
    u_elements = np.matmul(np.linalg.inv(V[:,elements]),b)
    # update vector u and return in
    for i in range(len(elements)):
        e = elements[i]
        u[e] = u_elements[i]
    return u

def eigen_value(M,v):
    ''' Provides the eigenvalue of a given matrix and a previously found eigenvector '''
    return (np.matmul(M,v)/v).mean()

if (__name__ == '__main__'):
    print('This module is not intended to run by iself')
