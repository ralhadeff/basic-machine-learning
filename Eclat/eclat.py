'''Eclat association groups finder'''

import numpy as np
from itertools import permutations

class Eclat():
    
    def fit(self,data,min_support=0.2,min_group_size=2,max_group_size=6):
        '''
        Find and record all combinations (groups) in the data with a minimum support as provided
            User can specify the minimum and maximum size of groups desired
        '''
        self.groups = []
        # minimum group size - only for purposes of export
        min_count = int(min_support*len(data))
        
        # get list of unique items in data
        items = []
        for i in data:
            items.extend(i)
        items = list(set(items))
        
        # generate groups and calculate support
        # only groups that passed the threshold will be carried over to increase the size
        passed = items.copy()
        # start with 2 to minimize combinations down the path
        for s in range(2,max_group_size+1):
            # add items to passed
            groups = []
            for p in passed:
                if (type(p) is str):
                    p = (p,)
                # keep groups as sets, to avoid order permutations
                to_add = [{*p,i} for i in items if i not in p]
                for a in to_add:
                    # only add unique combinations
                    if (a not in groups):
                        groups.append(a)
            # save only groups that passed the threshold
            passed = []
            for group in groups:
                count = 0
                # for each group, check that it is a subset of sample
                g = set(group)
                for i in data:
                    if g <= set(i):
                        count+=1
                if (count>=min_count):
                    passed.append(group)
                    if (s>=min_group_size):
                        self.groups.append(group)

if (__name__ == '__main__'):
    print('This module is not intended to run by iself')
