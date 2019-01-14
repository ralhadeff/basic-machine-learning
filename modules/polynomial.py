"""
This tool takes a DataFrame or a numpy array and returns an expanded array with higher polynomial features
"""

import numpy as np
import pandas as pd

def add_polynimials(df,power=2,cross=2,skip_row=[]):
    """
    Adds polynomial features for all features in the provided input
    The order is: [all original features (power 1)][all features squared][all features **3]...[all features **n]
    where the order is maintained within each bracket
    power is same feature to the power, cross is cross multiplications between features
    if using cross, all cross multiplications will be added to the end
    """
    # build skip list
    mask = []
    if (len(skip_row)>0):
        if isinstance(skip_row[0],str):
            for i in skip_row:
                mask.append(df.columns.get_loc(i))
        elif isinstance(skip_row[0],int):
            mask = skip_row
            # convert skip_row to names if using pandas DataFrame
            names = []
            if (type(df)==pd.DataFrame):
                for i in skip_row:
                    names.append(df.columns[i])
                skip_row = names

    
    # convert DataFrame to array
    pandas = False
    if (type(df)==pd.DataFrame):
        columns = df.columns.values
        new_columns = df.columns.values
        df = df.values
        pandas = True
    
    # reverse the mask
    rev = []
    for i in range(df.shape[1]):
        if (i not in mask):
            rev.append(i)
    mask = rev
        
    # add power polynomials
    new_df = df.copy()
    # skip 1 (is the original df), include power
    for i in range(2,power+1):
        df_power = np.vectorize(lambda x: x**i)(df[:,mask])
        new_df = np.hstack((new_df,df_power))
        if (pandas):
            new_columns = np.concatenate((new_columns,[name + "^" + str(i) for name in columns if name not in skip_row]))
    
    # add cross multiplications
    if (pandas):
        #convert names to list
        new_columns = list(new_columns)    
    # cross will be done pair-wise
    for colA in range(df.shape[1]):
        for colB in range(colA+1,df.shape[1]):
            if (colA in mask and colB in mask):
                # skip 1 (original df), include cross
                for i in range(2,cross+1):
                    # power of colA and colB
                    b = i-1
                    a = cross-b
                    df_cross = np.power(df[:,colA],a) * np.power(df[:,colB],b)
                    df_cross.shape = (new_df.shape[0],1)
                    new_df = np.hstack((new_df,df_cross))
                    if (pandas):
                        string = columns[colA]
                        if (a!=1):
                            string+= '^' + str(a)
                        string+='_x_' + columns[colB]
                        if (b!=1):
                            string+= '^' + str(b)
                        new_columns.append(string)
    
    if (pandas):
        new_df = pd.DataFrame(new_df,columns=new_columns)
    return new_df

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
