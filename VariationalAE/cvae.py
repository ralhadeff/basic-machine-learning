import numpy as np
import tensorflow as tf

from tensorflow.distributions import Normal
from tensorflow.distributions import Bernoulli

'''A Convolution Variational Auto Encoder'''

class ConvVAE:
    
    def __init__(self, image_shape=(128,128,3), conv_param=(3,16,True), n_list=[256,32]):
        # input data
        self.X = tf.placeholder(tf.float32, shape=(None, *image_shape))
        
        # encoder
        self.encoder_layers = []
        # convolution layer
        h = Conv2DLayer(image_shape[2],conv_param[0],conv_param[1],conv_param[2])
        self.encoder_layers.append(h)
        # flatten layer
        self.encoder_layers.append(FlattenLayer())
        # calculate number of input neurons to the FC layer
        previous = image_shape[0]*image_shape[1]*conv_param[1]
        if conv_param[2]:
            previous=previous//4
        # save for later
        flat = previous
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
        decoder_output = DenseLayer(previous,flat,activation=lambda x:x)
        self.decoder_layers.append(decoder_output)       
        #feed forward through decoder, using the sampled 'data'
        c_X = self.Z
        for layer in self.decoder_layers:
            c_X = layer.feed_forward(c_X)
        # reshape
        if (conv_param[2]):
            shape = [-1,image_shape[0]//2,image_shape[0]//2,conv_param[1]]
        else:
            shape = [-1,image_shape[0],image_shape[0],conv_param[1]]
        c_X = tf.reshape(c_X,shape)
        # convolution transpose
        self.trans_k = tf.Variable(tf.truncated_normal(
            [conv_param[0],conv_param[0],image_shape[2],conv_param[1]],stddev=0.1))
        if (conv_param[2]):
            strides=(1,2,2,1)
        else:
            strides=(1,1,1,1)
        c_X = tf.nn.conv2d_transpose(c_X, self.trans_k,strides=strides,padding='SAME',
                                     output_shape=[50,*image_shape])

        # output logit
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
        c_X = tf.reshape(c_X,shape)
        c_X = tf.nn.conv2d_transpose(c_X, self.trans_k,strides=strides,padding='SAME',
                                     output_shape=[50,*image_shape])
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
        
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(-self.elbo)
            
        self.init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(self.init)
        
    def fit(self,X,epochs=10,batch=50):
        n_batches = len(X) // batch
        for epoch in range(epochs):
            print('Epoch:',epoch+1)
            np.random.shuffle(X)
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
        pred,prob = self.session.run((self.post_pred,self.post_pred_probs),feed_dict={self.X:X})
        if (out=='prob'):
            return prob
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

class Conv2DLayer(object):
    '''A Conv2D layer'''
    
    def __init__(self,channels,k_size,n_kernels,use_pooling=True):
        shape=[k_size,k_size,channels,n_kernels]
        self.kernels = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
        self.bias = tf.Variable(tf.constant(0.0,shape=[n_kernels]))
        self.use_pooling = use_pooling
    
    def feed_forward(self, X):
        layer = tf.nn.conv2d(input=X,filter=self.kernels,strides=[1,1,1,1],padding='SAME')
        layer+= self.bias
        if self.use_pooling:
            layer = tf.nn.max_pool(layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        layer = tf.nn.relu(layer)
        return layer

class FlattenLayer(object):
    '''A flattening layer'''
          
    def feed_forward(self,X):
        shape = X.shape
        num_features = np.array(shape[1:4],dtype=int).prod()
        layer = tf.reshape(X,[-1,num_features])
        return layer
        
if (__name__ == '__main__'):
    print('This module is not intended to run by iself')
