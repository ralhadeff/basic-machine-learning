## Glossary

This is an incomplete list of terms relating to Data Science and Machine Learning.  
  
Note: these descriptions are written in a way that helps me understand and remember them better, and I apologize if anyone finds them unhelpful.
<br>
<br>

| Term  | Brief description | Link |
| --- | --- | --- |
| AdaBoost | An ensemble learning method where several weak learners are trained sequentially, and the training samples are weighted based on whether they were correctly classified or not, in the previous iteration | - |
| AdaGrad | An algorithm for SGD where the learning rate is adapted based on the frequency of features. It is a good algorithm for sparse data (e.g. bag of words). Related: RMSprop. | [link](http://ruder.io/optimizing-gradient-descent/index.html#adagrad) |
| Adversarial learning | A technique used, mostly for malicious motivations, to fool machine learning tools (*e.g.* using special character to fool simple spam classifiers. | [rifle-turtle](https://www.theverge.com/2017/11/2/16597276/google-ai-image-attacks-adversarial-turtle-rifle-3d-printed) |
| Autodiff | A computational algorithm to find the derivative of a function. Note, that this is not a numerical ('trial and error') not a symbolic (hard coding the derivative by a user) solution. | [Intuition tutorial](https://www.youtube.com/watch?v=twTIGuVhKbQ) |
| Autoencoder | An ANN that attempts to encode data to a lower dimension, then reconstruct it as faithfully as possible. In practice, this is an ANN that has fewer neurons in the hidden layers than the input, and is trained by comparing the output to the input | - |
| Container technology | An application bundled together with all its dependencies, libraries, binaries, and configuration files needed to run it - to avoid problems when migrating between different computers (or similar). | - |
| DNA storage | A method for storing digital data encoded into DNA (using the standard nucleotides). The achieved compression is enormous, but reading from the DNA is a slow process | - |
| Document clustering | Clustering textual documents together in a way that makes sense to humans, *e.g.* by subject or type of document. | - |
| Dropout / Dropconnect | Regularization methods for training ANNs. In Dropout, randomly selected neurons have their activation output set to zero, in each training sample. In dropconnect, randomly selected individual weights are set to zero, in each training sample. The weights have to be adjusted at the end of the training to account for this process | [Stack Exchange](https://stats.stackexchange.com/questions/201569/difference-between-dropout-and-dropconnect/201891) |
| Gradient boosting | A machine learning technique where different estimators are joined together (typically in sequence) to improve the overall predictions. | - |
| Label powerset | A multilabel classification solution (not to be confused with multiclass label classificiation) where a classifier is generated for each label combination in the dataset (up to the total combinatorial possibilities in n-labels). The advantage is that it considers correlations between the labels| [link](https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff) |
| Multi-level modeling | A modeling scheme with hierarchy. Different models are put together, where each focuses on features at different hierarchies (e.g. nested features). For example, a model for predicting student's performance based on students features, then school features, then city features, then country features | - |
| Sentiment analysis | A method to decipher subjective human emotion from given data; typically to determine the sentiment (*e.g.* like/dislike) from a text. | - |
| Shallow learning | Simplistic definition: ML methods that rely on user input (such as feature selection and manipulation), as opposed to deep learning, where the user only defines the architecture, and the deep learner finds the patterns and importances in all the raw data | - |
| Survival analysis | A tool used to investigate the time it takes for an event of interest to occur (*e.g.* death of an organism).  | - |
| VGG | A pretrained CNN model for object detection (from Oxford). | [website](http://www.robots.ox.ac.uk/~vgg/) |
| Whitening | A preprocessing method where the data is transformed such that there is no correlation between the features and all features have a variance of 1. Whiteningn can be done using PCA, and can improve the learning of a ML algorithm | [link](http://mccormickml.com/2014/06/03/deep-learning-tutorial-pca-and-whitening/) | 
| Word vector | A numerical representation (in the form of a vector) for words, where each dimension implies a specific feature of the word. Simplified example: king could be [1,1,0] where the first element stands for royalty and the second element stands for maleness (the third is femaleness), now man would be [0,1,0] and female would be [0,0,1], thus, kind-man+female=[1,0,1] which should also correspond to queen. | [link](https://medium.com/@jayeshbahire/introduction-to-word-vectors-ea1d4e4b84bf) |
