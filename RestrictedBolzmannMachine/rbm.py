'''Restricted Bolzmann machine'''

import numpy as np

class RBM():
    
    def __init__(self,n_input,n_hidden):
        '''
        A restricted Bolzmann machine
            n_input is the number of input features
            n_hidden is the numer of hidden nodes
        Uses a Bernoulli distribution
        '''
        self.weights = np.random.normal(0,0.1,(n_hidden,n_input))
        self.bias_v = np.random.normal(0,0.1,(n_input))  
        self.bias_h = np.random.normal(0,0.1,(n_hidden))
        
    def sample_h(self,X,return_probabilities=False):
        '''
        Sample activations from the hidden layer
            can return the probabilities or the activations
        '''
        z = X @ self.weights.T + self.bias_h # shape of result: samples x n_hidden
        prob = sigmoid(z)
        if (return_probabilities):
            return prob
        else:
            return bernoulli(prob) 

    def sample_v(self,X,return_probabilities=False):
        '''
        Sample activations from the visible layer
            can return the probabilities or the activations
        '''
        z = X @ self.weights + self.bias_v # shape of result: samples x n_visible
        prob = sigmoid(z)
        if (return_probabilities):
            return prob
        else:
            return bernoulli(prob) 
    
    def update(self,x0,xk,ph0,phk,learning_rate=0.01):
        '''
        Update the weights and biases of the machine
            x0 and xk are the visible nodes values at cycle 0 (input) and k
            ph0 and phk are the probabilities of sampling the hidden nodes at 0 and k
            learning rate for the weights and bias gradient descent
        '''
        self.weights += learning_rate* (x0.T @ ph0 - xk.T @ phk).T
        self.bias_v += learning_rate* (x0 - xk).sum(axis=0)
        self.bias_h += learning_rate* (ph0 - phk).sum(axis=0)

    def fit(self,X,epochs=100,learning_rate=0.01,k=1,batch_size=10,ignore=None):
        '''
        Fit the machine with the data X
            User can specify:
                number of epochs 
                learning rate
                k- number of iterations through the machine (visible-hidden-visible) before updating 
                batch size
                ignore - optional: do not learn from input that is equal to ignore (e.g. -1 for missing)
        '''
        for epoch in range(epochs):
            # index for batches
            ind = 0
            for batch in range(len(X)//batch_size+1):
                # the batch data
                x0 = X[ind:ind+batch_size]
                if (len(x0)==0):
                    # last batch can sometimes be empty
                    break
                # save probabilities for the update later
                ph0 = self.sample_h(x0,True)
                # xk when k=0
                xk = x0
                # iterations visible-hidden
                for i in range(k):
                    # feed into the machine
                    hk = self.sample_h(xk)
                    # retrieve visible nodes
                    xk = self.sample_v(hk)
                    if (ignore is not None):
                        # do not learn (=do not update visible nodes)
                        xk[x0==ignore] = ignore
                # at the end of the iterations, calculate probabilities 
                phk = self.sample_h(xk,True)
                # update weights and biases
                self.update(x0,xk,ph0,phk,learning_rate=learning_rate)
                # update index for next batch
                ind+=batch_size

    def predict(self,X):
        '''Iterate the data through he machine once'''
        return self.sample_v(self.sample_h(X))
                 
def sigmoid(X):
    return 1/(np.exp(-X)+1)    

def bernoulli(X):
    '''Returns 0 or 1 using a uniform distribution, with cutoffs provided in X'''
    draws = np.random.random(X.shape)
    np.random.random_sample()
    return (X > draws).astype(int)

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
