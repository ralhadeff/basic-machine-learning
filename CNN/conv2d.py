'''Convolutional layer for 2D images'''

import numpy as np
import neural_network

class Conv2D(neural_network.Layer):
    
    def __init__(self,no_of_kernels,color_channels,kernel_size=3,previous_layer=None,activation='relu',name='unnamed'):
        '''
        A convolutional layer for 2D images
            User specifies:
                the number of kernels and the kernel size
                    kernels are square, have a stride of 1 and no padding
                the previous layer that sends input into this layer
                the activation function
                optional: a name
        '''
        # save parameters for initialization
        self.n = no_of_kernels
        self.c = color_channels
        self.k = kernel_size
        
        # set hierarchy (same as Layer)
        self.previous_layer = previous_layer
        self.next_layer = None
        if (self.previous_layer is not None):
            previous_layer.next_layer = self
            previous_layer.initialize_weights()
        self.set_activation(activation)
        self.name = name
        # activation and backprop are explicit
        self.func = None
        self.deriv = None
           
    def initialize_weights(self):
        '''Initialize kernels and bias'''
        self.kernels = initialize_kernels(self.n,self.k,self.c)
        self.bias = np.zeros(self.n)    
    
    def set_activation(self,func):
        '''Conv2D has fixed activation (the convolution)'''
        pass
        
    def set_manual_activation_function(self,function,derivative):
        '''Conv2D has fixed activation (the convolution)'''
        pass
    
    def feed_forward(self,z):
        '''Propagate images forward into this layer'''
        self.z = z
        # do convolution and apply ReLU
        self.a = self.convolve(z)
        # apply ReLU
        a = neural_network.ReLU(self.a)
        if (self.next_layer is not None):
            return self.next_layer.feed_forward(a)
        else:
            return a
        
    def back_propagate(self,y=None,learning_rate=0.0001):
        '''Propagate errors backward into this layer'''        
        if (self.next_layer is None):
            # if the convolution layer is the last, TODO
            pass
        else:
            d = self.next_layer.delta.copy()
            # backprop ReLU
            d[self.z<=0] = 0
            # get the error, kernel error and bias error through backpropagation
            self.delta, k, b = self.d_convolve(d)
            # update kernels
            for i in range (self.weights.shape[1]):
                self.kernels -= learning_rate*k
            # update bias
            for i in range(len(self.bias)):
                self.bias -= learning_rate*b
        if (self.previous_layer is not None):
            self.previous_layer.back_propagate(learning_rate=learning_rate)

    def convolve(self,X):
        '''
        Performs convolution and return the resulting images (summed over all color channels)
            Image shape should be (width,height,channel)
        '''
        # get parameters
        n,k,_,ch = self.kernels.shape
        x,y,_ = X.shape
        # create new images array (one image for each kernel)
        convolved = np.zeros([n,x-k+1,y-k+1])
        # skip on edges (left and right)
        s_l = k//2
        s_r = k-s_l-1
        # apply convolution
        for kernel in range(n):
            # merge channels
            for i in range(s_l,x-s_r):
                for j in range(s_l,y-s_r):
                    # sum over all kernels
                    convolved[kernel,i-s_l,j-s_l] = (X[i-s_l:i+s_r+1,j-s_l:j+s_r+1,:] 
                                                     * self.kernels[kernel,:,:,:]).sum() + self.bias[kernel]
        return convolved

    def d_convolve(self,X):
        '''
        delta of the next layer (back propagating into this one)
        '''
        n,k_size,_,_ = self.kernels.shape
        x,y,_ = self.z.shape
        # initialize derivatives
        error = np.zeros(self.z.shape) 
        d_kernels = np.zeros(self.kernels.shape)
        d_bias = np.zeros(n)
        for k in range(n):
            for i in range(x-k_size):
                for j in range(y-k_size): 
                    # kernel weights delta
                    d_kernels[k] += delta[k,i,j] * self.z[i:i+k_size,j:j+k_size,:]
                    # error
                    error[i:i+k_size,j:j+k_size,:] += delta[k,i,j] * self.kernels[k]
            # loss gradient of the bias
            d_bias[k] = np.sum(delta[k])
        return error, d_kernels, d_bias
    
def initialize_kernels(n,k=3,c=3):
    '''
    Initialize kernels with weights and bias
        n - number of kernels/filters
        k - the size of the kernel (in pixels, square)
        c - the depth (channels)
    '''
    std = 1/np.sqrt(np.prod((n,k,k,c)))
    return np.random.normal(scale=std,size=(n,k,k,c))

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")