'''
pandas DataFrame scaler that scales all numeric values and returns a scaled DataFrame
returned DataFrame includes the untouched non-numeric values
'''

import pandas as pd
import numpy as np
import random

def scale(df,scaler='standard',skip_boolean='True',skip_row=[]):
    '''
    Scales all numeric data in dataframe
    default scaler is standard distribution
    other option for scaler is normalize (min=0 max=1)
    skip_boolean will skip all values that only have 0s and 1s from the scaling
    skip_row is a list of row names to skip
    '''
    new_df = pd.DataFrame()
    
    for i in df:
        new_df[i] = df[i]
        if (i not in skip_row):
            if (df.dtypes[i]==np.int64 or df.dtypes[i]==np.float64):
                if (skip_boolean):
                    list = df[i].unique()
                    if (len(list)<3):
                        if (0 in list and 1 in list):
                            continue
                if (scaler=='standard'):
                    mean = df[i].mean()
                    std = df[i].std()
                    if (std!=0):
                        new_df[i] = df[i].map(lambda x: (x-mean)/std)
                elif (scaler=='normalize'):
                    min = df[i].min()
                    max = df[i].max()
                    if (min-max!=0):
                        new_df[i] = df[i].map(lambda x: (x-min)/(max-min))
    return new_df

if (__name__ == '__main__'):
    print('This module is not intended to run by iself')
