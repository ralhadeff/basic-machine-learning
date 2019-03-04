'''Max pooling network layer for 2D images'''

import numpy as np
import neural_network

class MaxPool(neural_network.Layer):
    
    def __init__(self,pool_size,previous_layer=None,name='unnamed'):
        '''
        A max pooling layer for 2D images
            User specifies the pooling size
            Optional - user can specify a name
        '''
        self.size = pool_size
        
        self.previous_layer = previous_layer
        # sort hierarchy
        self.next_layer = None
        if (self.previous_layer is not None):
            previous_layer.next_layer = self
            previous_layer.initialize_weights()
        # no activation function
        self.func = lambda x:x
        self.deriv = lambda x:1
        # save name
        self.name = name

    def initialize_weights(self):
        # override Layer
        pass
        
    def feed_forward(self,z):
        # do max pooling then feed forward or return output
        self.z = z
        self.a = max_pool(z,self.size)
        if (self.next_layer is not None):
            return self.next_layer.feed_forward(self.a)
        else:
            return self.a  
        
    def back_propagate(self,y=None,learning_rate=0.0001):
        # do d_max pooling then feed backward
        self.delta = d_max_pool(y,self.z,self.size)
        if (self.previous_layer is not None):
            self.previous_layer.back_propagate(self.delta,learning_rate)
    
def max_pool(data,step_size):
    '''Max pool data and reduce the number of pixels'''
    # setup parameters
    x,y,n = data.shape
    # generate output array
    pool = np.zeros((x//step_size,y//step_size,n))
    # reduce size
    for k in range(n):
        # for each kernel
        for i in range(0,x-x%step_size,step_size):
            for j in range(0,y-y%step_size,step_size):
                pool[i//step_size,j//step_size,k] = np.max(data[i:i+step_size,j:j+step_size,k])
    return pool

def d_max_pool(delta, data, step_size):
    '''
    delta of the next layer (feeding backward into the max pool, weighted and after reshaping)
        original data used during the feed forward
    '''
    # output array
    x,y,n = data.shape
    error = np.zeros(data.shape)
    # loop through x,y and kernel
    for k in range(n):
        for i in range(0,x-x%step_size,step_size):
            for j in range(0,y-y%step_size,step_size): 
                # obtain index of largest value in input for current window
                current = data[i:i+step_size, j:j+step_size,k]
                (a, b) = np.unravel_index(np.argmax(current),current.shape)
                # update error to propagate backward (only the max value is propagated)
                error[i+a, j+b,k] = delta[i//step_size, j//step_size,k]        
                # no weights to update
    return error

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
