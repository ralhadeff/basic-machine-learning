'''A tool to perform TF-IDF and related analyses'''

import numpy as np
import string

class TFIDF():
    
    def fit(self,data,remove_punctuation='default',skip_words=None,ignore_case=False,return_processed=False):
        '''
        Fit the data and build a tf-idf matrix
            Data can be:
                a list of strings, where each string is a full document for the purpose of tf-idf
                a list of lists, where each inner list contains all the words in a full document
            
            By default, all punctuations will be remove (using string.punctuation)
                user can provide a custom string to remove specific punctuations
                Note: remove punctuations only works if strings are provided
            
            User can specify a list of words that should be skipped from the analysis (e.g. 'and','if')
            
            By default, case is ignored, user can specify that case should be ignored
                Note: ignore_case only works if strings are provided

            User can request for the modified data list of lists, for controls
        '''
        # convert to list of lists if necessary
        if (type(data[0]) is str):
            # strings
            new_data = []
            if remove_punctuation=='default':
                remove_punctuation = string.punctuation
            for document in data:
                if (ignore_case):
                    document = document.lower()
                if not (remove_punctuation=='' or
                    remove_punctuation=='none' or
                   remove_punctuation=='no'):
                    document = document.translate(str.maketrans('','',remove_punctuation))
                    words = document.split()
                    new_data.append(words)
            data = new_data
        # number of documents
        docs = len(data)
        # get a list of all unique words
        unique_words = set()
        for doc in data:
            unique_words.update(doc)
        # optionally remove words
        if (skip_words is not None):
            skip_words = set(skip_words)
            unique_words = unique_words - skip_words
        # convert to list to ensure order retention
        unique_words = list(unique_words)
        words = len(unique_words)
        # tf-idf arrays
        tf = np.zeros((docs,words))
        idf = np.zeros(words)
        # count occurrences
        for i in range(docs):
            # doc length
            d_length = len(data[i])
            if (d_length>0):
                for w in range(words):
                    # word count in current sample
                    w_count = data[i].count(unique_words[w])
                    # word frequency in current sample
                    tf[i,w] =  w_count / d_length
                    # add to idf, if word appeared in this sample
                    if (w_count>0):
                        idf[w]+=1
        # calculate idf correctly
        idf = np.log(docs/idf)

        # save information 
        self.unique_words = unique_words
        self.tf = tf
        self.tf_idf = tf*idf
        
        if (return_processed):
            # return the processed document
            return data
    
    def search_word(self,word,prune=False,return_tf_idf=False):
        '''
        Return a list with the indices of documents, sorted by relevance of the word provided
        User can prune out documents that do not contain the word at all
        User can specify the return of TF-IDF values as well
        '''
        # get index of word
        idx = self.unique_words.index(word)
        # get TD-IDF for word
        occ = self.tf_idf[:,idx]
        # get the sorted document indices, descending
        sort = np.argsort(occ)[::-1]
        # sort the TD-IDF
        sorted_occ = occ[sort]
        if (prune):
            # save only non zero TD-IDF
            pruned = sorted_occ[sorted_occ>0]
            # prune indices
            sort = sort[sorted_occ>0]
            # overwrite for returns
            sorted_occ = pruned
        if (return_tf_idf):
            # return indices and TD-IDF's
            return sort, occ[sort]
        else:
            # return only indices
            return sort
    
    def get_important_list(self,no_of_words=1):
        '''Return the n most relevant words for each document in corpus'''
        relevant = []
        for doc in self.tf_idf:
            # for each document, find the highest TF-IDF values, descending
            # return list of words
            relevant.append(
                [self.unique_words[i] for i in np.argsort(doc)[::-1][:no_of_words]])
        return relevant

if (__name__ == '__main__'):
    print('This module is not intended to run by iself')
