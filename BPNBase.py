#coding=utf-8

import numpy as np
import math

sigmoid = np.frompyfunc(lambda z:1.0 / (1+ math.exp(-z)),1,1)
sigmoidGradient = np.frompyfunc(lambda z : sigmoid(z) * (1-sigmoid(z)),1,1)
def multip_func(a, b):
    if a == 0:
        return 0
    return a * b
multip = np.frompyfunc(multip_func, 2, 1)
def logp_func(a):
    if a == 0:
        return -np.inf
    return np.log(a)
logp = np.frompyfunc(logp_func, 1, 1)

def GetRandWeights(Lin, Lout):
    einit = math.sqrt(6.0 / (Lin + Lout))
    w = np.matrix(np.random.rand(Lout, 1 + Lin) * 2 * einit - einit)
    return w

def Predict(X, ts):
    tmp = X
    for t in ts:
        z = np.c_[np.ones([tmp.shape[0], 1]), tmp] * t.T
        tmp = sigmoid(z) # a
    return tmp.astype(np.float)


def Cost(hx, y):
    m,_ = y.shape
    a = multip(-y, logp(hx))
    b = multip(1.0 - y, logp(1.0 - hx))
    J = np.sum(a - b) / m
    return J
