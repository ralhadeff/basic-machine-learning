'''
VAE that also takes a y features array and tries to steer some latent features to map to given features
Most of the code is identical to vae.py
'''

import numpy as np
import tensorflow as tf

from tensorflow.distributions import Normal
from tensorflow.distributions import Bernoulli

class SteeredVAE:
    
    def __init__(self, n_input, n_list, n_y=None,y_weight=100):
        '''
        n_input - number of input neurons
        n_list - list of numbers of neurons in the hidden layers
        n_y: optional - number of features that will be given as input y during training
        y_weight - relative weight of losses (VAE vs regression for y features). Trial-and-error
        '''
        # input data
        self.X = tf.placeholder(tf.float32, shape=(None, n_input))
        # input y features
        if (n_y is not None):
            self.y = tf.placeholder(tf.float32, shape=(None, n_y))
        
        # encoder
        self.encoder_layers = []
        # input layer
        previous = n_input
        # current is the output of each layer (skip last because there is nothing after it)
        for current in n_list[:-1]:
            h = DenseLayer(previous,current)
            self.encoder_layers.append(h)
            previous = current
        # latent features number
        latent = n_list[-1]
        encoder_output = DenseLayer(current,latent*2,activation='none')
        self.encoder_layers.append(encoder_output)
        
        # feed forward through encoder
        c_X = self.X
        for layer in self.encoder_layers:
            c_X = layer.feed_forward(c_X)
        # c_X now holds the output of the encoder
        # first half are the means
        self.means = c_X[:,:latent]
        # second half are the std; must be positive; +1e-6 for smoothing
        self.std = tf.nn.softplus(c_X[:,latent:]) + 1e-6
        
        # optional loss for steered latent features
        if (n_y is not None):
            self.yhat = self.means[:,:n_y]
            self.error = tf.losses.mean_squared_error(labels=self.y,predictions=self.yhat)
        
        # reparameterization trick
        normal = Normal(loc=self.means,scale=self.std)
        self.Z = normal.sample()

        # decoder
        self.decoder_layers = []
        previous = latent
        for current in reversed(n_list[:-1]):
            h = DenseLayer(previous,current)
            self.decoder_layers.append(h)
            previous = current
        # output is the reconstruction
        decoder_output = DenseLayer(previous,n_input,activation=lambda x:x)
        self.decoder_layers.append(decoder_output)

        #feed forward through decoder, using the sampled 'data'
        c_X = self.Z
        for layer in self.decoder_layers:
            c_X = layer.feed_forward(c_X)
        logits = c_X
        # use logits for cost function below
        neg_cross_entropy = -tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X,
                    logits=logits)
        neg_cross_entropy = tf.reduce_sum(neg_cross_entropy, 1)
        
        # output
        self.y_prob = Bernoulli(logits=logits)
        
        # sample from output
        self.post_pred = self.y_prob.sample()
        self.post_pred_probs = tf.nn.sigmoid(logits)
        
        # generate 'de-novo' output
        self.gen = tf.Variable(0)
        Z_std = Normal(0.0,1.0).sample([self.gen,latent])
        c_X = Z_std
        for layer in self.decoder_layers:
            c_X = layer.feed_forward(c_X)
        logits = c_X
        
        prior_pred_dist = Bernoulli(logits=logits)
        self.prior_pred = prior_pred_dist.sample()
        self.prior_pred_probs = tf.nn.sigmoid(logits)
        
        # manually input Z
        self.Z_input = tf.placeholder(np.float32, shape=(None, latent))
        c_X = self.Z_input
        for layer in self.decoder_layers:
            c_X = layer.feed_forward(c_X)
        logits = c_X
        self.manual_prior_prob = tf.nn.sigmoid(logits)
        
        # cost function
        # Kullback–Leibler divergence
        kl = -tf.log(self.std) + 0.5*(self.std**2 + self.means**2) - 0.5
        kl = tf.reduce_sum(kl, axis=1)
        # ELBO
        self.elbo = tf.reduce_sum(neg_cross_entropy - kl)
        
        if (n_y is None):
            # only ELBO
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(-self.elbo)
        else:
            # weighted regression loss and ELBO
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(
                tf.reduce_sum(y_weight*self.error-self.elbo))

        self.init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(self.init)
    
    def steer(self,X,y,epochs=10,batch=50):
        '''Replaces fit, user provides the y features for the latent steering'''
        n_batches = len(X) // batch
        for epoch in range(epochs):
            print('Epoch:',epoch+1)
            cost = 0
            e_cost = 0
            for b in range(n_batches):
                c_batch = X[b*batch:(b+1)*batch]
                y_batch = y[b*batch:(b+1)*batch]
                _,c,e, = self.session.run((self.optimizer, self.elbo,self.error),feed_dict={self.X: c_batch,self.y:y_batch})
                # accumulate cost
                cost+=c
                e_cost+=e
            print('Cost:', cost,e_cost)
    
    def fit(self,X,epochs=10,batch=50):
        n_batches = len(X) // batch
        for epoch in range(epochs):
            print('Epoch:',epoch+1)
            cost = 0
            for b in range(n_batches):
                c_batch = X[b*batch:(b+1)*batch]
                _,c, = self.session.run((self.optimizer, self.elbo),feed_dict={self.X: c_batch})
                # accumulate cost
                cost+=c
            print('Cost:', cost)
                       
    def predict(self,X,out='prob'):
        '''
        Pass data through encoder and decoder and retrieve reconstructed output
            by default the probabilities are returned, user can specify 'sample' or 'both'
        '''
        # correct shape if needed
        if (X.ndim==1):
            X = X.reshape([1,-1])
        pred,prob,mm = self.session.run((self.post_pred,self.post_pred_probs,self.means),feed_dict={self.X:X})
        if (out=='prob'):
            return prob,mm
        elif (out=='sample'):
            return pred
        else:
            return pred,prob

    def generate(self,n=1,out='prob'):
        '''
        Generate output
            by default the probabilities are returned, user can specify 'sample' or 'both'
            User specifies the number of points requested 
        '''
        pred,prob = self.session.run((self.prior_pred,self.prior_pred_probs),feed_dict={self.gen:n})
        if (out=='prob'):
            return prob
        elif (out=='sample'):
            return pred
        else:
            return pred,prob
    
    def feed(self,Z):
        '''Generate output using provided latent-space input Z'''
        # correct shape if needed
        if (Z.ndim==1):
            Z = Z.reshape([1,-1])
        return self.session.run(self.manual_prior_prob,feed_dict={self.Z_input:Z})
    
    def close(self):
        self.session.close()

class DenseLayer(object):
    '''A fully connected layer'''
    
    def __init__(self, n_in, n_out, activation=tf.nn.relu):
        '''number of input and output neurons; the activation function'''
        self.weights = tf.Variable(tf.random_normal(shape=(n_in, n_out), stddev=2/np.sqrt(n_in)))
        self.bias = tf.Variable(tf.constant(0.0,shape=[n_out]))
        if (activation=='none'):
            self.activation = lambda x: x
        else:
            self.activation = activation
            
    def feed_forward(self, X):
        '''Run input through layer and retrieve output'''
        return self.activation(tf.matmul(X, self.weights) + self.bias)

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
