# Generative adversarial network (GAN)

### DCGAN with Keras on MNIST dataset

DCGAN along with several improvement stategies are demonstrated for comparison and future reference.


* `DCGAN` - the standard DCGAN.
* `DCGAN_experience_replay` - variation with experience replay; results appear similar.
* `DCGAN_noise` - variation with random noise added to the images; results are inferior.
* `DCGAN_conditional` - variation with labels provided during training; failed to learn, needs to be reworked.
