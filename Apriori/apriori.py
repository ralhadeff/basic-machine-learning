'''A tool to calculate A-priori association rules'''

import numpy as np
import pandas as pd

class Apriori():
    
    def __init__(self):
        '''An apriori association rules finder'''
        pass
    
    def fit(self,data):
        '''
        Find all the association rules in the dataset
            Only does rules for pairs
        '''
        # total number of samples
        users = len(data)
        # generate list of possible items
        items = []
        for i in data:
            items.extend(i)
        self.items = list(set(items))
        # number of counts for each item (assume each person only buys one copy)
        item_counts = np.array([np.sum([n in i for i in data]) for n in self.items])
        # all the supports
        self.supports = item_counts/users   
        
        # co-support array
        self.co_support = np.zeros((len(self.items),len(self.items)))

        # generate associated counting array
        for i in range(len(self.items)):
            for j in range(len(self.items)):
                for p in data:
                    if self.items[i] in p and self.items[j] in p:
                        # co_support is initially the counts
                        self.co_support[i,j]+=1
        # calculate confidence (transpose to divide column-wise)
        self.confidence = (self.co_support / item_counts).T
        # co support corrected
        self.co_support/= users

        # calculate lift
        self.lift = self.confidence / self.supports
    
    def get_association(self,item_1, item_2):
        '''
        Fetch association rule for 2 specified items
            return the co-support, the confidence of the rule and the lift
            
        The rule applies for 'if item_1 then what is the rule for item_2'
        '''
        a = self.items.index(item_1)
        b = self.items.index(item_2)
        return self.co_support[a,b],self.confidence[a,b],self.lift[a,b]
    
    def get_rules(self,min_support,min_confidence,min_lift):
        '''
        Return a list of all the rules that comply with the given conditions
        '''
        # collect indices of rules 
        rules = []
        for i in range(len(self.co_support)):
            for j in range(i+1,len(self.co_support)):
                    if (self.co_support[i,j]>=min_support):
                        if (self.lift[i,j]>=min_lift):
                            # confidence is not symmetrical
                            left = self.confidence[i,j]
                            right = self.confidence[j,i]
                            if (left>=min_confidence or right>=min_confidence):
                                if (left>=right):
                                    rules.append((i,j))
                                else:
                                    rules.append((j,i))
        # generate DataFrame of output, placing the highest confidence on the left
        names = [(self.items[i],self.items[j]) for i,j in rules]
        numbers = [self.get_association(i,j) for i,j in names]
        return pd.DataFrame(np.hstack((names,numbers)),
                            columns=['item_a','item_b','support','confidence(left)','lift'])

