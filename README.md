# basic-machine-learning
**A python implementation for machine learning tools**

*Written by Raphael Alhadeff, started Jan 2019*

This repository will include all my machine learning implementations in python. The purpose of this implementation is for my own training, and it is not particularly efficient.

Note: although many sections in the code could be improved for efficiency (e.g. more vectorization) one of my goals here was to make the code more instructive for a human reading it.

For the first few estimators (Linear and Logistic) I wrote a full support to pandas DataFrames and Series. I did not always implement this because there was no more new code to write and learn from by continuing this support. Generally speaking, using DF.values or Series.values should work for all numerical data.


General tools that are in specific folders (other folder names are self-explanatory):
 * Regularization is in LinearRegression (and has not be implemented in the LogisticRegression)
 * One-vs-rest classification scheme is in LogisticRegression, ovr
 * One-vs-one classification scheme is in SVM, ovo
 * `modules` contains linear algebra, scaling tools, a polynomial features generator, and a data splitting tool
