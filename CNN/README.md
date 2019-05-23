# Convolutional neural network

Layers that are typical to CNNs. The standard layers were constructed from scratch, whereas more complex applications of convolutions were done using Keras (_e.g._ residual block) or other off-the-shelf libraries for speed (_e.g._ `VariationalAE`). See also `misc` folder for specific techniques (_e.g._ semantic segmentation).  

`blocks.py` and `blocks_callable.py` contain several CNN architecture blocks implemented with keras and demonstrated on MNIST in `_blocks_example.ipynb` (implemented as methods or callable classes, repsectively).  
