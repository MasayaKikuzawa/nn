# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 12:36:03 2017

@author: bakan
"""

"""
neuralnetwork：オリジナル、入力層、中間層、出力層の3層のモデル
neuralnetwork_kai：入力層、中間層、中間層、出力層の4層のモデル
neuralnetwork_final:中間層の層数を変更できるモデル
使いたいモデルのコメント文を外してください
"""
#from neuralnetwork import *
#from neuralnetwork_kai import *
from neuralnetwork_final import *
import time

if __name__ == '__main__':
    #学習データ
    X = numpy.array([[0, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1]])
    #教師データ
    T = numpy.array([[0], [0], [1], [0], [1], [1]])
    #データ数
    N = X.shape[0]
    #入力データ
    inputdata = numpy.array([[0, 1, 0], [0, 1, 1]])
    #入力ノード数
    input_size = X.shape[1]
    #中間層ノード数
    hidden_size = 4
    #中間層の層数
    noml = 3
    #出力ノード数
    output_size = 1
    #学習率
    epsilon = 0.1
    #エポック数
    epoch = 100000
    
    starttime = time.time()
    """
    finalを使う場合はnomlが引数に入ってるものを利用、それ以外はnomlが引数に入っていないものを使用
    """
    #nn = Neural(input_size, hidden_size, output_size)
    nn = Neural(input_size, hidden_size, output_size,noml)
    #nn.train(X, T, epsilon, epoch)
    nn.train(X, T, epsilon, epoch,noml)
    nn.error_graph()

    #Y = nn.predict(X)
    Y = nn.predict(X,noml)
    
    endtime = time.time()
    
    print(Y)
    print(endtime-starttime)
    
    """
    大まかな関数の説明はneuralnetwork.pyに書いておきます
    """
    
