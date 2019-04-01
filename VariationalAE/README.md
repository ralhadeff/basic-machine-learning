# Variational Autoencoder

For practical reasons, the VAE is using TensorFlow rather than my implementation of a network.

`vae.py` file contains a class for a VAE that can process images. There are two examples, one on MNIST and one on randomly generated rectangles.

`streered_vae.ipynb` is an experimental hybrid, where the user provides images but also the labels for the desired latent features. The VAE learn to encode using the specified features, and from those features.  
The application that I am trying to apply this to is inputing cell cross-section images, and teaching a network to reconstruct the images, but also be able to generate in-between (w.r.t. cross-axis) images for a continuous reconstruction of the 3D image (see progress [here](https://github.com/ralhadeff/computational-biology/tree/master/tomographies)).  

The trained latent features can encompass the whole latent feature space or only part of it (see `steered_vae_partial.ipynb` as well).
