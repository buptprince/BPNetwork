#coding=utf-8

from BPNetwork import *
import numpy as np
import scipy.io as sio

GRID = 10
fin = open("tdata.txt", "r")
Xs = []
ys = []
while 1:
    line = fin.readline()
    if len(line) == 0:
        break
    flag = int(line)
    ys.append(flag)
    x = []
    for r in range(GRID):
        line = fin.readline().strip()
        for i in line.split(' '):
            x.append(int(i.strip()))
    Xs.append(x)

n = GRID * GRID
m = len(ys)
X = np.matrix(Xs)
y = np.matrix(ys).reshape(m, 1)

#notice: in y, 0 -> 10

net = BPNetwork()

ty = np.matrix(np.zeros([m, 10]))
for j in range(m):
    k = y[j]
    if k == 10:
        k = 0
        y[j] = 0
    ty[j, k] = 1

net.load("mynetgood.bin")
net.train(X, ty, [GRID * GRID, 100, 10], alpha = 0.05, iter_time = 100, lam = 0.1, NewTheta = False)
p = net.predict(X)
py = np.argmax(p, axis = 1)
right = np.sum(py == y)
print ("%lf %%" % (right * 1.0 / m * 100))
net.save("mynetgood.bin")
