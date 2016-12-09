#coding=utf-8

from BPNetwork import *
import numpy as np
import scipy.io as sio
import random

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
ratio = 0.8
tm = int(m * ratio)
cvm = m - tm

r = range(m)
ti = random.sample(r, tm)
tXs = []
tys = []
cvXs = []
cvys = []
for i in range(m):
    if i in ti:
        tXs.append(Xs[i])
        tys.append(ys[i])
    else:
        cvXs.append(Xs[i])
        cvys.append(ys[i])

X = np.matrix(tXs)
y = np.matrix(tys).reshape(tm, 1)
cvX = np.matrix(cvXs)
cvy = np.matrix(cvys).reshape(cvm, 1)

#notice: in y, 0 -> 10

net = BPNetwork()

def GetTy(y):
    m = len(y)
    ty = np.matrix(np.zeros([m, 10]))
    for j in range(m):
        k = y[j]
        if k == 10:
            k = 0
            y[j] = 0
        ty[j, k] = 1
    return ty

ty = GetTy(y)

net.load("mynetgood.bin")
#net.train(X, ty, [GRID * GRID, 100, 10], alpha = 0.05, iter_time = 100, lam = 0.1, NewTheta = False)
net.train(X, ty, [GRID * GRID,20,10], alpha = 0.001, iter_time = 100, lam = 0.1, NewTheta = False)
p = net.predict(cvX)
py = np.argmax(p, axis = 1)
right = np.sum(py == cvy)
print ("%lf %%" % (right * 1.0 / cvm * 100))
net.save("mynetgood.bin")
