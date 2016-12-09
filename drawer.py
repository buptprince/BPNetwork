#coding=utf-8
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import sys
import numpy as np
from BPNetwork import *
 
LEARNING_MODE = True
LEARNING_SINGLE = 10

class Drawer(QWidget):
    GRID_SIZE = 8
    GRID_W = 80
    GRID_H = 80
    def __init__(self):
        QWidget.__init__(self)
        self.initUI()
        self.mat = np.matrix(np.zeros((Drawer.GRID_W, Drawer.GRID_H)))
        self.mouseButton = 0
    def initUI(self):
        self.setGeometry(200, 200, Drawer.GRID_SIZE * Drawer.GRID_W + 300, Drawer.GRID_SIZE * Drawer.GRID_H)
        self.setWindowTitle('Drawer')
        self.label = QLabel(self)
        self.label.setText("Predict: ?")
        self.btnclear = QPushButton(self)
        self.btnclear.setText("clear")
        w = Drawer.GRID_SIZE * Drawer.GRID_W
        self.editbox = QLineEdit(self)
        self.editbox.resize(20, 20)
        self.label.move(w + 10, 10)
        self.btnclear.move(w + 10, 50)
        self.editbox.move(w + 10, 80)
        self.connect(self.btnclear, SIGNAL("clicked()"), self.ClearMat)
        #self.connect(self.editbox, SIGNAL(""), self.
        self.net = BPNetwork()
        self.net.load("mynetgood.bin")
        self.learningS = 0 # 学习次数
        self.learningM = 0 # 正在学习的数字
        self.X = None
        self.y = None
        self.lastP = None
        self.show()
        
    def ClearMat(self):

        if LEARNING_MODE:
            if self.learningM == 10:
                print "finished"
                return
            print "%d: %d / 10" % (self.learningM, self.learningS + 1)
            if self.X == None:
                self.X = self.mat.copy()
                self.y = np.matrix([[self.learningM]])
            else:
                self.X = np.r_[self.X, self.mat]
                self.y = np.r_[self.y, np.matrix([[self.learningM]])]
            '''
            print self.X
            print "==="
            print self.y
            '''
            self.learningS += 1
            if self.learningS == LEARNING_SINGLE:
                if self.learningM == 10:
                    print "save"
                    self.X.tofile("X.bin")
                    self.y.tofile("y.bin")
                else:
                    self.learningS = 0
                self.learningM += 1


        self.mat = np.matrix(np.zeros((Drawer.GRID_W, Drawer.GRID_H)))
        self.label.setText("Predict: ?")
        self.update()
    def mousePressEvent(self, event):
        p = event.pos()
        self.mouseButton = event.button()
        self.drawp(p.x(), p.y())
        self.lastp = p
    def mouseMoveEvent(self, event): 
        p = event.pos()
        self.dline(p)
        self.drawp(p.x(), p.y())
    def dline(self, p):
        #self.lastp -> p
        d = self.lastp - p
        print d
        self.lastp = p
    def mouseReleaseEvent(self, event):
        self.lastp = None
        #Predict
        x = self.mat.reshape(1, self.GRID_W * self.GRID_H)
        p = self.net.predict(x)
        y = np.argmax(p, axis = 1)
        self.label.setText("Predict: %d" % y)
    def keyPressEvent(self, event):
        print event.key()
    def drawp(self, x, y):
        c = x / self.GRID_SIZE
        r = y / self.GRID_SIZE
        if 0 <= c < self.GRID_W and 0 <= r <= self.GRID_H:
            if self.mouseButton == Qt.LeftButton:
                self.mat[r, c] = 1
            else:
                self.mat[r, c] = 0
            self.update()
    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
            

        for r in range(Drawer.GRID_H):
            for c in range(Drawer.GRID_W):
                x = c * Drawer.GRID_SIZE
                y = r * Drawer.GRID_SIZE
                color = Qt.white
                if self.mat[r, c] == 1:
                    color = Qt.black
                qp.setPen(QPen(color, 1, Qt.SolidLine))
                qp.setBrush(QBrush(color, Qt.SolidPattern))
                qp.drawRect(x, y, Drawer.GRID_SIZE, Drawer.GRID_SIZE)

        pen = QPen(Qt.gray, 1 , Qt.SolidLine)
        pen.setStyle(Qt.CustomDashLine)
        pen.setDashPattern([1, 2])
        qp.setPen(pen)
        for i in range(Drawer.GRID_W):
            qp.drawLine(i * Drawer.GRID_SIZE, 0, i * Drawer.GRID_SIZE, Drawer.GRID_H * Drawer.GRID_SIZE)
        for i in range(Drawer.GRID_H):
            qp.drawLine(0, i * Drawer.GRID_SIZE, Drawer.GRID_W * Drawer.GRID_SIZE, i * Drawer.GRID_SIZE)




        qp.end()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Drawer()
    app.exec_()
    
