'''
A tool to generate cumulative accuracy profiles (CAP) for binary classifications
'''

import numpy as np

def cap(true_y,probabilities,perfect=False):
    '''
    Cumulative accuracy profile (CAP)
    input true labels and probability output from classifier
    perfect can be set to true to output the perfect results (for plotting) for this dataset
    '''
    # if perfect is requested, generate 3 point array and exit early
    if (perfect):
        # total True
        tt = (true_y==True).sum()
        cap = np.zeros((3,2))
        cap[0,:] = (0,0)
        cap[1,:] = (tt,tt)
        cap[2,:] = (len(true_y),tt)
        return cap
  
    # join array and sort by probability
    m = np.concatenate((true_y,probabilities)).reshape(2,len(true_y)).T
    m = np.flip(m[m[:,1].argsort()],axis=0)
    
    # build results matrix
    cap = np.zeros((len(true_y),2))
    for i in range(len(m)):
        cap[i,0] = i
        cap[i,1] = m[:i,0].sum()
    
    return cap

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
