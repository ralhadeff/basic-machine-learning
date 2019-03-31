'''
A simple scaler for pandas dataframes. Saves the fitted scaling factors, then can scale or unscale dataframes as needed
'''

import pandas as pd
import numpy as np

class Scaler(object):
    '''
    Scaler that scales all numeric data in a dataframe
    Scaler is first fit to a dataset, then it can be applied to scale or unscale datat
    '''
    
    def __init__(self):
        # factors for the scaling
        # key is the name of the column, and the factors (a,b) are for scaling:
        # x_scaled = (x-a)/b
        self.factors = {}
        
    def fit(self,df,scaler='standard',skip_boolean='True',skip_row=[]):
        '''
        Fits the scaling for all numeric data in dataframe
        default scaler is standard distribution
        other option for scaler is normalize (min=0 max=1)
        skip_boolean will skip all values that only have 0s and 1s from the scaling
        skip_row is a list of row names to skip
        '''
        for i in df:
            # add default factor
            self.factors[i]=(0,0)
            if (i not in skip_row):
                if (df.dtypes[i]==np.int64 or df.dtypes[i]==np.float64):
                    if (skip_boolean):
                        list = df[i].unique()
                        if (len(list)<3):
                            if (0 in list and 1 in list):
                                #self.factors[i]=(a,b)
                                continue
                    # set factors based on method
                    if (scaler=='standard'):
                        a = df[i].mean()
                        b = df[i].std()
                    elif (scaler=='normalize'):
                        a = df[i].min()
                        # (max - min)
                        b = df[i].max() - a
                    # save factors for future use
                    self.factors[i]=(a,b)
        
    def scale(self,df):
        '''
        Scales the dataframe based on the fitted factors
        returns a new dataframe
        '''
        new_df = pd.DataFrame()
        
        for i in df:
            # scale, except columns that should be skipped
            a,b = self.factors[i]
            if (b!=0):
                new_df[i] = df[i].map(lambda x: (x-a)/b)
            else:
                new_df[i] = df[i]
        return new_df
    
    def unscale(self,df):
        '''
        Unscales a dataframe based on the fitted factors (reverses the scaling)
        returns a new dataframe
        '''
        new_df = pd.DataFrame()
        
        for i in df:
            # unscale, except columns that should be skipped
            a,b = self.factors[i]
            if (b!=0):
                new_df[i] = df[i].map(lambda x: x*b+a)
            else:
                new_df[i] = df[i]
        return new_df        

if (__name__ == '__main__'):
    print('This module is not intended to run by iself')
