'''
A layer class to construct an artificial neural network (ANN)
    the ANN can be used for regression or classification
'''

import numpy as np

class Layer():
    
    def __init__(self,no_of_neurons,previous_layer=None,activation='none',name='unnamed'):
        '''
        An ANN layer
            User specifies the number of neurons, the previous layer that sends input into this layer
                the activation function and optionally a name
            If no previous layer is specified, the layer is assumed to be the input layer
                Only when a layer is specified as a previous layer it is really initiated 
                    (except for an output layer)
        '''
        # number of neurons in current layer
        self.n = no_of_neurons
        # layer that sends input into this layer
        self.previous_layer = previous_layer
        # layer that receives this layer's output
        self.next_layer = None
        if (self.previous_layer is not None):
            # sort the hierarchy
            previous_layer.next_layer = self
            # initialize the weights between the layers
            previous_layer.initialize_weights()
        # set the activation function
        self.set_activation(activation)
        # save name
        self.name = name
    
    def initialize_weights(self):
        '''
        Initializes the weights between this layer and the next one
            weights are initialized to a value close to zero
        '''
        if (self.next_layer is not None):
            # weights
            self.weights = np.random.normal(0,0.1,(self.n,self.next_layer.n))
            # bias separately
            self.bias = np.random.normal(0,0.1,self.next_layer.n)
    
    def set_activation(self,func):
        '''
        Set the activation function
        '''
        if (func=='ReLU'):
            # set ReLU and the derivative
            self.func = ReLU
            self.deriv = dReLU
        if (func=='sigmoid'):
            self.func = sigmoid
            self.deriv = dSigmoid
        elif (func=='none'):
            self.func = lambda x:x
            self.deriv = lambda x:1

    def set_manual_activation_function(self,function,derivative):
        '''
        Set user provided custom functions (including the derivative)
        '''
        self.func = function
        self.deriv = derivative
            
    def feed_forward(self,z):
        '''
        Feed input into a layer (if this is the first layer, z is X)
            input should be provided as a single sample
        '''
        # remember z (input) and a (output after activation, but before weights) for later
        self.z = z
        self.a = self.func(z)
        if (self.next_layer is not None):
            # feed the weighted output (and bias) to the next layer
            # recursively return the output
            return self.next_layer.feed_forward(self.a @ self.weights + self.bias)
        else:
            # last layer, return the output backwards
            return self.a
          
    def back_propagate(self,y=None,learning_rate=0.0001):
        '''
        Backpropagate true y values into the network for training
            Learning rate can be specified
        '''
        if (self.next_layer is None):
            # last layer's error and delta are the derivative of the loss function
            if (self.func == sigmoid):
                # classification, using binary cross entropy
                self.error = -( (y/self.a) - ((1-y)/(1-self.a)) )
                self.delta = self.error * self.deriv(self.z)
            else:
                # regression, using MSE 
                self.error = self.a - y
                self.delta = self.error * self.deriv(self.z)
        else:
            # generic layer, propagate error from the next layer's error
            self.error = self.weights @ self.next_layer.delta
            self.delta = self.error * self.deriv(self.z)
            for i in range (self.weights.shape[1]):
                # update weights
                self.weights[:,i] -= learning_rate * (self.a * self.next_layer.delta[i])
            for i in range(len(self.bias)):
                # update bias
                self.bias[i] -= learning_rate * self.next_layer.delta[i]
        if (self.previous_layer is not None):
            # continue to backpropagate as long as there are layers
            self.previous_layer.back_propagate(learning_rate=learning_rate)
        
def ReLU(x):
    '''ReLU function'''
    return np.maximum(x,0)

def dReLU(x):
    '''Derivative of relu'''
    # make a copy so that input is not modified
    x = x.copy()
    x[x<=0] = 0
    x[x>0] = 1
    return x   

def sigmoid(x):
    '''Sigmoid function'''
    return 1/(np.exp(-x)+1)

def dSigmoid(x):
    '''Derivative of sigmoid'''
    s = sigmoid(x)
    return s*(1-s)

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
