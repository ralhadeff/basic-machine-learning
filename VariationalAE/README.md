# Variational Autoencoder

For practical reasons, the VAE is using TensorFlow rather than my implementation of a network.

`vae.py` file contains a class for a VAE that can process images. There are two examples, one on MNIST and one on randomly generated rectangles.

`streered_vae.ipynb` is an experimental hybrid, where the user provides images but also the labels for the desired latent features. The VAE learn to encode using the specified features. The application that I want to apply this to is inputing cell image planes, and teaching a network to reconstruct the cell, but also be able to generate in-between (Z-axis) images for a continuous reconstruction of the 3D image (currently the images produced by the group I am working with is discreet stacks). 
