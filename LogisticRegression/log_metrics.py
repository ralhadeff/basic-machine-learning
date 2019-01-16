"""
Logistic regression metrics tools
"""

import numpy as np

def accuracy(real_y,predictions):
    """
    Total accuracy of classification (correct labels)
    """
    return (real_y==predictions).mean()

def cross_entropy_loss(real_y,probabilities):
    """
    Cross entropy loss using : (yt log(yp) + (1 - yt) log(1 - yp))
        yt = y true
        yp = probability of y being 1
    Note that predictions here must be provided as probabilities
    """
    # check that predictions are correctly provided as probabilities (won't work for multiclass):
    if (np.unique(probabilities).shape == (2,)):
        raise ValueError("Input predictions should be probabilities, not labels")
    logyp = np.log(probabilities)
    log1yp = np.log(1-probabilities)
    return -(real_y*logyp + (1-real_y)*log1yp).mean()

def roc(real_y,probabilities,points=None,area=False):
    """
    Receiver operating characteristic (ROC) curve
    plots the true-positive rate as a function of the false-positive rate
    Number of points to calculate the ROC curve can be provided (to increase precision)
    Area under the graph can be calculated if requested
    """
    # check that predictions are correctly provided as probabilities:
    if (np.unique(probabilities).shape == (2,)):
        raise ValueError("Input predictions should be probabilities, not labels")
    # set number of points equal to the dataset size
    if (points==None):
        t = len(np.unique(probabilities))
    else:
        t=points
    # +1 because start and end are included
    tpr = np.zeros((t+1,))
    fpr = np.zeros((t+1,))
    cutoffs = np.linspace(0,1,t+1)
    for i in range(t+1):
        c = cutoffs[i]
        # nearly same code as confusion matrix, only works for binary classification
        matrix = np.zeros((2, 2),dtype=int)
        for actual, predicted in zip(real_y, (probabilities>c)):
            matrix[actual][int(predicted)]+= 1
        tpr[i] = matrix[1,1]/(matrix[1,:].sum())
        fpr[i] = matrix[0,1]/(matrix[0,:].sum())
    # prune duplicates out
    tpr,fpr = np.split(np.unique(np.concatenate((tpr,fpr)).reshape(2,t+1).T,axis=0),2,axis=1)
    # make 1-D array
    tpr = tpr.ravel()
    fpr = fpr.ravel()
    if (area):
        # calculate area under graph, simplified to linear lines between points
        previous_x = 0.0
        previous_y = 0.0
        area = 0.0
        for i in range(len(tpr)):
            x = fpr[i]
            y = tpr[i]
            # square under the triangle
            area+= (x-previous_x)*previous_y
            # triangle between previous line and current line
            area+= (x-previous_x)*(y-previous_y)/2
            # update
            previous_x = x
            previous_y = y
        return fpr, tpr, area
    else:
        return fpr,tpr 

def confusion_matrix(real_y,predictions,show=False):
    """
    Calculate and print the confusion matrix
    default return is matrix, print flag will instead print the output in a readable way
    """
    # determine the unique labels
    labels = len(np.unique(real_y))
    # construct confusion matrix
    matrix = np.zeros((labels, labels),dtype=int)
    for actual, predicted in zip(real_y, predictions):
        matrix[actual][predicted]+= 1
    if (show):
        lines=[]
        labels = np.sort(np.unique(real_y))
        lines.append('Prediction >\t')
        for label in labels:
            lines[0]+=str(label)+'\t'
        lines.append('V Label V')
        lines.append('')
        for i in range(len(labels)):
            line = str(labels[i])+'\t'
            for j in matrix[i,:]:
                line+='\t'+str(j)
            lines.append(line)
        for line in lines:
            print(line)
    else:
        return matrix

def classification_report(real_y,predictions, decimal=2,show=False):
    """
    Calculate and return the classification report
    last row is the weighted average
    print flag will instead print the output in a readable way
    """
    labels = np.unique(real_y)
    matrix = confusion_matrix(real_y,predictions)
       
    report = np.zeros((len(labels)+1,5))
    
    for label in range(len(labels)):
        # label
        report[label,0]=labels[label]
        # precision
        report[label,1]=np.round(matrix[label,label]/(matrix[:,label].sum()),decimals=decimal)
        # recall
        report[label,2]=np.round(matrix[label,label]/(matrix[label,:].sum()),decimals=decimal)
        # f1-score
        report[label,3]=np.round(2*(report[label,1]*report[label,2])/
                                 (report[label,1]+report[label,2]),decimals=decimal)
        # support
        report[label,4]=(real_y==label).sum()
    # weighted average
    for i in range(1,4):
        report[len(labels),i] = np.round((report[:-1,i]*report[:-1,4]).sum()/len(real_y),decimals=decimal)
    report[len(labels),4] = report[:-1,4].sum()
    
    if (show):
        print('\tprecisn\trecall\tf1-sc\tsupport')
        print('\n')
        for label in range(len(labels)):
            print(int(report[label,0]),'\t',report[label,1],'\t',report[label,2],'\t',
                 report[label,3],'\t',int(report[label,4]))
        print('\n')
        print('w-ave\t',report[len(labels),1],'\t',report[len(labels),2],'\t',
              report[len(labels),3],'\t',int(report[len(labels),4]))
    else:
        return report    

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
