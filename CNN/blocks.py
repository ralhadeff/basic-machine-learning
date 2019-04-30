from keras.layers import Dense, Convolution2D, MaxPooling2D, concatenate, add, Activation

# VGG block
def vgg(X, kernels, layers):
    x = X
    # convolutions
    for i in range(layers):
        x = Convolution2D(kernels,kernel_size=(3,3), padding='same', activation='relu')(x)
    # pooling
    x = MaxPooling2D((2,2))(x)
    return x

# residual block (ResNet)
def residual(X, kernels):
    res = X
    x = X
    # fit the input shape if needed
    if (res.shape[-1] != kernels):
        res = Convolution2D(kernels,(1,1),padding='same',activation='relu')(x)
    # convolution
    x = Convolution2D(kernels,(3,3),padding='same',activation='relu')(x)
    x = Convolution2D(kernels,(3,3),padding='same',activation='linear')(x)
    # add convolution to the residual
    x = add([x,res])
    x = Activation('relu')(x)
    return x

# naive inception block
def inception_naive(X, kernels_1, kernels_3, kernels_5):
    x = X
    x1 = Convolution2D(kernels_1,(1,1),padding='same',activation='relu')(x)
    x3 = Convolution2D(kernels_3,(3,3),padding='same',activation='relu')(x)
    x5 = Convolution2D(kernels_5,(5,5),padding='same',activation='relu')(x)
    p = MaxPooling2D((3,3),strides=(1,1),padding='same')(x)
    x = concatenate([x1,x3,x5,p],axis=-1)
    return x

# proper inception block
def inception(X,kernels_1, k_3_in, kernels_3, k_5_in, kernels_5, kernels_pool):
    x = X
    
    x1 = Convolution2D(kernels_1,(1,1),padding='same',activation='relu')(x)
    
    x3 = Convolution2D(k_3_in,(1,1),padding='same',activation='relu')(x)
    x3 = Convolution2D(kernels_3,(3,3),padding='same',activation='relu')(x3)
    
    x5 = Convolution2D(k_5_in,(1,1),padding='same',activation='relu')(x)
    x5 = Convolution2D(kernels_5,(5,5),padding='same',activation='relu')(x5)
    
    p = MaxPooling2D((3,3),strides=(1,1),padding='same')(x)
    p = Convolution2D(kernels_pool, (1,1), padding='same', activation='relu')(p)
    
    x = concatenate([x1,x3,x5,p],axis=-1)
    return x
