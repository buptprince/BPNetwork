#coding=utf-8

import numpy as np
from BPNBase import *

class BPNetwork:
    def __init__(self):
        self.theta = None
    def train(self, X, y, layers, alpha = 0.1, iter_time = 100):
        # 单输出模型
        '''
            for example, 
                X: m x 400
                Theta1: 25 x 401
                Theta2: 10 x 26
                y 10 x 1
            layers = [400, 25, 10]
        '''
        # Theta 个数
        tlen = len(layers) - 1
        ts = [GetRandWeights(layers[i], layers[i+1]) for i in range(tlen)] 
        D = [np.matrix(np.zeros(ts[i].shape)) for i in range(tlen)]
        m, n = X.shape
        for i in range(iter_time):
            #Predict
            z = [None for _ in range(tlen + 1)]
            a = [None for _ in range(tlen + 1)]
            a[0] = X.T
            #改为每列一个样本
            for i in range(len(ts)):
                t = ts[i]
                z[i + 1] = t * np.r_[np.matrix(np.ones((1, a[i].shape[1]))), a[i]]
                a[i + 1] = sigmoid(z[i + 1])
            err = a[len(ts)] - y.T # 每列一个样本
            for j in range(tlen-1,-1,-1):
                for k in range(m):
                    u = np.r_[np.ones((1,1)), a[j][:,k]]
                    d = err[:,k] * u.T
                    D[j] = D[j] + d
                if j == 0:
                    break
                err = np.multiply(ts[j][:,1:].T * err, sigmoidGradient(z[j]))
            #D/m 为正梯度
            for j in range(tlen):
                ts[j] = ts[j] - (alpha / m) * D[j]
        self.theta = ts
        hx = Predict(X, ts)
        print ("J: %lf" % Cost(hx, y))

    def predict(self, X):
        return Predict(X, self.theta)
