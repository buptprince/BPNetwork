#coding=utf-8

from BPNetwork import *
import numpy as np

X = np.matrix("0,0;0,1;1,0;1,1")
y = np.matrix("0;1;1;1")

net = BPNetwork()
net.train(X, y, [2,1], alpha = 0.1, iter_time = 1000)
