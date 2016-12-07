#coding=utf-8

from BPNetwork import *
import numpy as np
import scipy.io as sio

#data = sio.loadmat('ex4data1.mat')
#X = data['X']
#y = data['y']
X = np.fromfile("train.bin").reshape(100, 64)
y = np.matrix([i / 10 for i in range(100)]).reshape(100,1)
m, n = X.shape

#notice: in y, 0 -> 10

net = BPNetwork()

ty = np.matrix(np.zeros([m, 10]))
for j in range(m):
    k = y[j]
    if k == 10:
        k = 0
        y[j] = 0
    ty[j, k] = 1

net.train(X, ty, [64, 25, 10], alpha = 0.001, iter_time = 10000, lam = 0.1)
p = net.predict(X)
py = np.argmax(p, axis = 1)
right = np.sum(py == y)
print ("%lf %%" % (right * 1.0 / m * 100))
net.save("mynetgood.bin")
