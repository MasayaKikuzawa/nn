# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:36:03 2020

@author: bakan
"""

from neuralnetwork_kai import *

if __name__ == '__main__':

    X = numpy.array([[0, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1]])
    T = numpy.array([[0], [0], [1], [0], [1], [1]])
    NOML = 3
    N = X.shape[0] # number of data
    inputdata = numpy.array([[0, 1, 0], [0, 1, 1]])

    input_size = X.shape[1]
    hidden_size = 8
    output_size = 1
    epsilon = 0.1
    epoch = 50000

    nn = Neural(input_size, hidden_size, output_size)
    nn.train(X, T, epsilon, epoch)
    nn.error_graph()

    Y = nn.predict(X)
    print(Y)

