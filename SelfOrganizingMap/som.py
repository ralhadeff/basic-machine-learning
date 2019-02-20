'''
A self organizing map calculator
'''

import numpy as np
import math

class SOM():
    
    def __init__(self, rows, columns):
        '''
        Self organizing map calculator.  
            Grid's number of rows and columns
        '''
        self.rows = rows
        self.cols = columns

    def fit(self, X, iterations, learning_rate=0.5):
        '''
        Calculate a 2D map with a square topology
            User input is the number of iterations to perform and the starting learning
        '''
        
        # initialize grid and node weights
        nodes = np.random.normal(0,0.1,(self.rows,self.cols,X.ndim))

        # setup starting values
        rows,cols,_ = nodes.shape
        # smallest radius that will still include all nodes
        start_radius = (max(rows,cols))/2
        time_constant = iterations / math.log(start_radius)

        for i in range(iterations):
            # pick one sample at random
            sample = X[np.random.randint(len(X))]
            # fine BMU
            closest = find_clostest(nodes,sample)
            node = nodes[closest]
            # update current radius and learning rate
            l = learning_rate * math.exp ( -i/iterations )
            r_squared = (start_radius * math.exp(-i/time_constant))**2
            # update BMU
            nodes[closest] = updated_weights(node,sample,learning_rate)
            # update neighbors
            ii,jj,_ = nodes.shape
            for i in range(ii):
                for j in range(jj):
                    neighbor = nodes[i,j]
                    if ((i,j) != closest):
                        # distance to neighbor
                        dist_squared = (np.array((closest[0]-i,closest[1]-j))**2).sum()
                        # update neighbor if within radius (squared)
                        if (dist_squared < r_squared):
                            influence = math.exp(-dist_squared/(2*r_squared) )
                            nodes[(i,j)] = updated_weights(nodes[(i,j)],sample,influence*l)
        self.nodes = nodes
    
    def get_map(self):
        '''Currently using all features'''
        x,y,_ = self.nodes.shape
        distance_map = np.zeros((x,y))
        # accumulate distances
        for i in range(x):
            for j in range(y):
                # calculate the distance of all new pairs and accumulate in the map
                # distances are calculated once and added to both nodes on the map
                for a,b,c,d in get_new_pairs(i,j,x,y):
                    dist = get_inter_node_distance(self.nodes[a,b],self.nodes[c,d])
                    distance_map[a,b]+=dist
                    distance_map[c,d]+=dist
        # divide by number of neighbors (using adjacents including diagonals)
        # corners have 3 neighbors, edges have 5 and all others have 8
        distance_map/=8
        distance_map[[0,-1],:]*=1.6
        distance_map[:,[0,-1]]*=1.6
        distance_map[[0,-1],[0,-1]]/=0.96
        distance_map[[-1,0],[0,-1]]/=0.96
        # return averaged distance map
        return distance_map
        
def get_new_pairs(i,j,x,y):
    '''
    Return a list of tuples of all the pairs that
        have not been already calculated for the current node
        i,j - current indices
        x,y - dimention of grid
    '''
    pairs = []
    if (i<x-1):
        if (j>0):
            # top right
            pairs.append((i,j,i+1,j-1))
        # front
        pairs.append((i,j,i+1,j))
        if (j<y-1):
            # bottom right
            pairs.append((i,j,i+1,j+1))
    if (j<y-1):
        # bottom
        pairs.append((i,j,i,j+1))
    return pairs
                             
def get_inter_node_distance(i,j):
    '''calculate the distance (in grid space) between two nodes'''
    return math.sqrt( ((i-j)**2).sum() )
        
def find_clostest(nodes,sample):
    '''find the closest node to this sample'''
    min_dist = float('inf')
    n_rows,n_cols,_ = nodes.shape
    for i in range(n_rows):
        for j in range(n_cols):
            # distance to current node
            dist = ((nodes[i][j]-sample)**2).sum()
            if (dist<min_dist):
                # update if closer than previously closest
                min_dist = dist
                closest = (i,j)
    return closest

def updated_weights(current,sample,rate):
    '''calculate the current updated weights (does not assign)'''
    delta = sample-current
    return current + delta * rate

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
