"""
This module is a simple and inefficient implementation of the basic linear algebra required for machine learning tools
It is designed for educational purposes only, and is by no means intended to be actually used (numpy is much better if you actually need to use linear algebra)
The code was written by Raphael Alhadeff, Jan 2019, as part of my self-training in data science skills
For simplicity, I will use lists for vectors, and then lists of vectors for matrices

Note: more linear algebra tools specific for PCA are present in the pca.py file in the PCA folder
"""

import math
from collections import Iterable

class Vector(object):
    """
    This is a simple implementation of a vector
    constructor takes an arbitrary number of numbers, and converts them to floats
    elements are stored in a list
    """  
    
    def __init__(self, *args):
        """
        *args takes an arbitrary number of numbers
        can include vectors, lists and tuples, and will create a vector list of all the collections in order
        """
        self.values = []
        if isinstance(args,Iterable):
            for i in args:
                if (isinstance(i,Iterable)):
                    for j in i:
                        self.values.append(float(j))
                else:
                    self.values.append(float(i))
    
    def present(self):
        """
        Print the vector's values
        """
        print(self.values)
    
    def size(self):
        """
        Returns the vector's size in space
        """
        return math.sqrt(sum([x**2 for x in self.values]))
    
    def length(self):
        """
        Returns the number of elements in the vector
        """
        return len(self.values)
    
    def copy(self):
        """
        Returns a copy of the current vector
        """
        return Vector(self.values)
    
    def add(self,vector):
        """
        Adds two vectors together, returns the sum (doesn't modify original vector)
        """
        if (self.length()!=vector.length()):
            raise ValueError("Vectors must be of the same size to be added together")
        return Vector([x+y for (x,y) in zip(self.values,vector.values)])
    
    def multiply(self,scalar):
        """
        Multiplies all elements in a vector by scalar, returns the result (doesn't modify original vector)
        """
        return Vector([x*scalar for x in self.values])
    
    def dot(self,vector):
        """
        Returns the dot product of the two vectors
        """
        if (self.length()!=vector.length()):
            raise ValueError("Vectors must be of the same size to be added together")
        return sum([x*y for (x,y) in zip(self.values,vector.values)])
    
    def normalize(self):
        """
        Returns a 1-unit size vector the is parallel to current vector
        """
        l = self.size()
        return Vector([x/l for x in self.values])
    
    def orthogonal(self):
        """
        Returns the orthogonal vector
        """
        # set all values to 1 except last one
        last = -1*sum(self.values[:-1])/self.values[-1]
        vector = [1 for i in range(self.length()-1)]
        vector.append(last)
        return Vector(vector).normalize()
       
    def __iter__(self):
        # iterable of vector returns the list of values
        return self.values.__iter__()
    
    def __next__(self):
        # iterable of vector returns the list of values
        return self.values.__next__()
       
    def __len__(self):
        # len returns the length of values
        return self.length()
    
    def __getitem__(self, key):
        # getitem treats the values list directly
        return self.values[key]
    
    def __neg__(self):
        return Vector([-x for x in self])
    
    def __add__(self,other):
        # adds scalar to all values or adds vector (of same size) element-wise
        if (isinstance(other,Iterable)):
            if (len(other)!=self.length()):
                raise ValueError("Added collections must be of the same size")
            else:
                return Vector([x+y for (x,y) in zip(self.values,other)])
        else:
            return Vector([x+other for x in self])
    
    def __sub__(self,other):
        # subtracts scalar from all values or subtracts vector (of same size) element-wise
        if (isinstance(other,Iterable)):
            if (len(other)!=self.length()):
                raise ValueError("Added collections must be of the same size")
            else:
                return Vector([x-y for (x,y) in zip(self.values,other)])
        else:
            return Vector([x-other for x in self])
    
    def __mul__(self,other):
        # multiplies all values by scalar, or returns the dot product of two vectors
        if (isinstance(other,Iterable)):
            if (len(other)!=self.length()):
                raise ValueError("Added collections must be of the same size")
            else:
                return self.dot(Vector(other))
        else:
            return Vector([x*other for x in self])
    
    def __truediv__(self,other):
        # divides all values by scalar
        return Vector([x/float(other) for x in self])
    
    def to_matrix(self):
        """
        Converts vector to a matrix with 1 row
        """
        return Matrix(self)
    
    def transpose(self):
        """
        Converts vector to a matrix with 1 column = transpose
        """
        trans = []
        for i in self:
            trans.append((i,))
        return Matrix(*trans)
    
    def delete(self, i):
        """
        Return vector with element i removed
        """
        return Vector(self.values[:i]+self.values[i+1:])

class Matrix(object):
    """
    This is a simple implementation of a matrix, using a list of Vectors from this same module
    """   
   
    def __init__(self, *args):
        """
        *args can be a list of lists, tuples or Vectors
        """
        self.matrix = []
        # initinalize number of columns variable
        columns = len(args[0])
        for i in args:
            if (len(i)!=columns):
                raise ValueError("Matrix must have the same number of elements in each row")
            self.matrix.append(Vector(i))
        
    def present(self):
        """
        Print all the values of the matrix
        """
        for i in self.matrix:
            Vector(i).present()
            
    def shape(self):
        """
        Returns a tuple with the matrix's dimensions
        """
        return (len(self.matrix),len(self.matrix[0]))
    
    
    def transpose(self):
        """
        Returns the transposed matrix
        """
        trans = []
        for i in range(len(self.matrix[0])):
            trans.append([v[i] for v in self.matrix])
        return Matrix(*trans)
    
    def __iter__(self):
        # iterable of vector returns the list of values
        return self.matrix.__iter__()
    
    def __next__(self):
        # iterable of vector returns the list of values
        return self.matrix.__next__()
          
    def __getitem__(self, key):
        # getitem treats the values list directly
        return self.matrix[key]
    
    def __neg__(self):
        m = []
        for v in self.matrix:
            m.append(-v)
        return Matrix(*m)
    
    def __add__(self,other):
        """
        Adds a scalar to all values of the matrix
        or adds a vector row-wise, vector must be of length equal to number of columns
        or adds a matrix element wise, matrix must have the same size
        """
        m = []
        if (isinstance(other,Matrix)):
            for i in range(self.shape()[0]):
                m.append(self[i]+other[i])
        elif (isinstance(other,Iterable)):
            if (len(other)!=self.shape()[1]):
                raise ValueError("Added collection must have the size equal to the number of columns")
            else:
                for v in self.matrix:
                    m.append(v+Vector(other))
        else:
            for v in self.matrix:
                m.append(v+other)
        return Matrix(*m)
    
    def __sub__(self,other):
        # same as add but with a -
        return self+(-other)
    
    def __mul__(self,other):
        """
        Multiplies all elements by scalar
        or multiplies matrix by vector or matrix, where shapes have to be appropriate to linear algebra rules
        """
        # using methods to make code more readable
        if (isinstance(other,Matrix)):
            return self.by_matrix(other)
        elif (isinstance(other,Iterable)):
            return self.by_vector(Vector(other))
        else:
            m = []
            for v in self.matrix:
                m.append(v*other)
        return Matrix(*m)
 
    def __truediv__(self,other):
        # divides all elements by scalar
        m = []
        for v in self.matrix:
            m.append(v/other)
        return Matrix(*m)
    
    def by_vector(self,other):
        """
        Multiplies matrix by vector
        vector lengths must match row vector's length
        """
        if (len(self.matrix[0])!=len(other)):
            raise ValueError("Dimensions don't match")
        else:
            return Vector([v*other for v in self])
    
    def by_matrix(self,other):
        """
        Multiplies two matrix
        the first matrix's column number must match the second matrix's row number
        """
        m = []
        for u in other.transpose():
            m.append(self*u)
        return Matrix(*m).transpose()
    
    def det(self):
        return det(self)
    
    def minor(self, i,j):
        """
        Returns the minor of the matrix, remove row i and column j
        """
        m = []
        for v in range(len(self.matrix)):
            if (v!=i):
                m.append(self.matrix[v].delete(j))
        return Matrix(*m)
    def cofactor(self):
        return cofactor(self)
    
    def inverse(self):
        return invert(self)
    
def det(matrix):
    """
    Calculates the determinant of a matrix
    """
    # no determinant for non square matrices
    if (not isinstance(matrix,Matrix) or len(matrix.matrix[0])!=len(matrix.matrix)):
        raise ValueError("Only square matrices have a determinant")
    # 1x1 matrix's determinant is trivial
    if (len(matrix[0])==1):
        return matrix[0][0]
    #recursively, calculate the determinant until reaching a 2x2 matrix
    if (len(matrix[0])==2):
        return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
    else:
        sum = 0
        # alternating sign
        p = +1
        # go over first line
        for i in range(len(matrix[0])):
            sum+= p * matrix[0][i] * det(matrix.minor(0,i))    
            p*=-1
    return sum

def cofactor(matrix):
    """
    Returns the co-factor matrix
    """
    if (not isinstance(matrix,Matrix) or len(matrix.matrix[0])!=len(matrix.matrix) or len(matrix[0])<2):
        raise ValueError("Only square matrices of at least 2x2 elements have a co-factor matrix")
    size= len(matrix[0])
    # construct new matrix
    m=[]
    for i in range(size):
        a = []
        for j in range(size):
            p = (-1)**(i+j)
            a.append(p*det(matrix.minor(i,j)))
        m.append(a)
    return Matrix(*m)

def adjoint(matrix):
    """
    Returns the adjoint matrix
    """
    return cofactor(matrix).transpose()

def invert(matrix):
    """
    Returns the inverted matrix
    """
    return adjoint(matrix)/det(matrix)

if (__name__ == '__main__'):
    print("This module is not intended to run by iself")
