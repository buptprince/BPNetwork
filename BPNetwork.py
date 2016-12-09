#coding=utf-8

import numpy as np
from BPNBase import *
import pickle
import copy

class BPNetwork:
    def __init__(self):
        self.theta = None
    def train(self, X, y, layers, alpha = 0.1, iter_time = 100, lam = 0, NewTheta = True):
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
        if NewTheta:
            ts = [GetRandWeights(layers[i], layers[i+1]) for i in range(tlen)] 
        else:
            ts = copy.copy(self.theta)
        D = [np.matrix(np.zeros(ts[i].shape)) for i in range(tlen)]
        m, n = X.shape
        xt = np.matrix(X.T).astype(np.float)
        bestJ = 1000
        bestT = None
        for it in range(iter_time):
            #Predict
            z = [None for _ in range(tlen + 1)]
            a = [None for _ in range(tlen + 1)]
            a[0] = xt
            #改为每列一个样本
            for i in range(len(ts)):
                z[i + 1] = ts[i] * np.r_[np.matrix(np.ones((1, a[i].shape[1]))), a[i]].astype(np.float)
                #a[i + 1] = sigmoid(z[i + 1])
                a[i + 1] = 1.0 / (1.0 + np.exp(-z[i + 1])).astype(np.float)
            err = a[len(ts)] - y.T # 每列一个样本
            for j in range(tlen-1,-1,-1):
                for k in range(m):
                    u = np.r_[np.ones((1, 1)), a[j][:,k]]
                    d = err[:,k] * u.T
                    D[j] = D[j] + d
                b = ts[j]
                b[:, 0] = 0
                D[j] += lam * b
                if j == 0:
                    break
                sz = 1.0 / (1 + np.exp(-z[j]))
                sgz = np.multiply(sz, (1 - sz))
                err = np.multiply(ts[j][:,1:].T * err, sgz)
            #D/m 为正梯度
            for j in range(tlen):
                ts[j] = ts[j] - (alpha / m) * D[j]
            #Get Cost

            hx = Predict(X, ts)
            J = Cost(hx, y)
            if J < bestJ:
                bestT = copy.copy(ts)
                bestJ = J
            print ("Iter: %d,  J: %lf" % (it, J))
            if J <= 5e-7:
                break

        self.theta = copy.copy(bestT)
        hx = Predict(X, bestT)
        print ("J: %lf" % Cost(hx, y, self.theta, lam))

    def predict(self, X):
        return Predict(X, self.theta)
    def save(self, filename):
        output = open(filename, "wb")
        pickle.dump(self.theta, output)
        output.close()
    def load(self, filename):
        fin = open(filename, "rb")
        self.theta = pickle.load(fin)
        fin.close()
