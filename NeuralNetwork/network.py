'''Neural network builder, trainer and predictor (based on Layer class)'''

import numpy as np
from neural_network import Layer

class Network():
    
    def __init__(self,*layers,output_type='classification',inner_activation='ReLU'):
        '''
        Construct an atrificial neural network
            User specifies a list of number of neurons (sequential, including input and output layers)
            User can speciffy classification or regression, the number of output neurons will determine
                whether to use a sigmoid or softmax for the output layer
            Inner activation is ReLU by default, but user can specify other functions
                currently supported are 'none', 'sigmoid' and 'ReLU'
        '''
        self.labels = None
        self.encoded = False
        network = []
        prev = Layer(layers[0])
        network.append(prev)
        for l in layers[1:-1]:
            curr = Layer(l,prev,activation=inner_activation)
            network.append(curr)
            prev = curr
        if (output_type=='classification'):
            # label network for potentially label encoding
            self.labels = True
            if (layers[-1]==1):
                out = 'sigmoid'
            else:
                out = 'softmax'
            network.append(Layer(layers[-1],prev,activation=out))
        else:
            network.append(Layer(layers[-1],prev,activation='none'))
        self.network = network
        
        
    def train(self, X, y, epochs,learning_rate=0.001):
        '''Train the network with the provided data'''
        # check if y needs to be converted from labels to label encoded
            # i.e. from [1,2,3] to [1,0,0],[0,1,0],[0,0,1]
        if (self.labels is not None):
            self.labels = np.unique(y)
            if (len(self.labels)==2 and y.ndim==1):
                # binary classification
                new_y = np.zeros(len(y),dtype=int)
                new_y[y==self.labels[1]] = 1
                # sanity check
                if (self.network[-1].n != 1):
                    raise ValueError('Your output does not properly match your network architecture')
            else:
                # multiclass labeling
                # convert to label encoded, unless already encoded
                if (y.ndim == 1):
                    new_y = np.zeros((len(y),len(self.labels)),dtype=int)
                    for i in range(len(self.labels)):
                        new_y[y==self.labels[i],i] = 1
                    # sanity check
                    if (self.network[-1].n != len(self.labels)):
                        raise ValueError('Your output does not properly match your network architecture')
                else:
                    self.encoded = True
                    # for consistency
                    new_y = y
            # replace y with corrected array
            y = new_y
        for epoch in range(epochs):
            for i in range(len(X)):
                self.network[0].feed_forward(X[i])
                self.network[-1].back_propagate(y[i],learning_rate)  
    
    def predict(self, X):
        if (self.labels is None or self.encoded):
            n = self.network[-1].n
            pred = np.zeros((len(X),n))
        else:
            n = 1
            pred = np.zeros(len(X)).astype(self.labels.dtype)
        for i in range(len(X)):
            p = self.network[0].feed_forward(X[i])
            if (self.labels is None):
                pred[i] = p
            elif (len(self.labels)==2 and self.encoded==False):
                if (p<0.5):
                    pred[i] = self.labels[0]
                else:
                    pred[i] = self.labels[1]
            else:
                if (self.encoded):
                    arg = np.argmax(p)
                    pred[i,arg] = 1
                else:
                    pred[i] = self.labels[np.argmax(p)]
        if n==1:
            pred = pred.ravel()
        return pred
    
if (__name__ == '__main__'):
    print('This module is not intended to run by iself')
