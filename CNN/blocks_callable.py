
# coding: utf-8

# In[1]:


from keras.layers import Convolution2D, MaxPooling2D, Activation, add, concatenate


# In[2]:


class VGG_block():
    
    def __init__(self, kernels, layers):
        l = []
        for i in range(layers):
            l.append(Convolution2D(kernels,kernel_size=(3,3), padding='same', activation='relu'))
        l.append(MaxPooling2D((2,2)))
        self.layers = l
            
    def __call__(self,x):
        for l in self.layers:
            x = l(x)
        return x    

class Residual_block():
    
    def __init__(self,kernels):
        self.kernels = kernels
        self.layers = []
        self.layers.append(Convolution2D(kernels,(1,1),padding='same',activation='relu'))
        self.layers.append(Convolution2D(kernels,(3,3),padding='same',activation='relu'))
        self.layers.append(Convolution2D(kernels,(3,3),padding='same',activation='linear'))
        self.act = Activation('relu')
        
    def __call__(self,x):
        res = x
        if (res.shape[-1] != self.kernels):
            x = self.layers[0](x)
        for l in self.layers[1:]:
            x = l(x)
        x = add([x,res])
        x = self.act(x)
        return x

class Inception_block():
    
    def __init__(self,kernels_1, k_3_in, kernels_3, k_5_in, kernels_5, kernels_pool):
        
        self.l1 = Convolution2D(kernels_1,(1,1),padding='same',activation='relu')
        self.l3 = []
        self.l3.append(Convolution2D(k_3_in,(1,1),padding='same',activation='relu'))
        self.l3.append(Convolution2D(kernels_3,(3,3),padding='same',activation='relu'))
        self.l5 = []
        self.l5.append(Convolution2D(k_5_in,(1,1),padding='same',activation='relu'))
        self.l5.append(Convolution2D(kernels_5,(5,5),padding='same',activation='relu'))
        self.p = []
        self.p.append(MaxPooling2D((3,3),strides=(1,1),padding='same'))
        self.p.append(Convolution2D(kernels_pool, (1,1), padding='same', activation='relu'))
     
    def __call__(self,x):
        x1 = self.l1(x)
        x3 = self.l3[1](self.l3[0](x))
        x5 = self.l5[1](self.l5[0](x))
        p = self.p[1](self.p[0](x))
        x = concatenate([x1,x3,x5,p],axis=-1)
        return x

