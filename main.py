# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:36:03 2020

@author: bakan
"""
#from neuralnetwork import *
from neuralnetwork_kai import *
#from neuralnetwork_final import *
import time

if __name__ == '__main__':

    X = numpy.array([[0, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1]])
    T = numpy.array([[0], [0], [1], [0], [1], [1]])
    N = X.shape[0] # number of data
    inputdata = numpy.array([[0, 1, 0], [0, 1, 1]])

    input_size = X.shape[1]
    hidden_size = 4
    noml = 3
    output_size = 1
    epsilon = 0.1
    epoch = 10000
    
    starttime = time.time()
    
    nn = Neural(input_size, hidden_size, output_size)
    #nn = Neural(input_size, hidden_size, output_size,noml)
    nn.train(X, T, epsilon, epoch)
    n#n.train(X, T, epsilon, epoch,noml)
    nn.error_graph()

    Y = nn.predict(X)
    #Y = nn.predict(X,noml)
    
    
    endtime = time.time()
    
    print(Y)
    print(endtime-starttime)
