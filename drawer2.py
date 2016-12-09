#coding=utf-8
import pygame
import math
from pygame.locals import *
import numpy as np
from pygame.font import *
import threading
from BPNetwork import *
 
LEARNING_NUM = -1 
class Thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        while 1:
            q = raw_input("Please Input Learing Num: (-1 doesn't not learning): ")
            try:
                i = int(q)
                if 0 <= i <= 9:
                    print "Learning: %d" % i
                else:
                    print "Doen't Learn"
                    i = -1
            except:
                i = -1
                print "Doesn't Learn"
            global LEARNING_NUM
            LEARNING_NUM = i

class Brush():
    def __init__(self, screen):
        self.screen = screen
        self.color = (0, 0, 0)
        self.size  = 1
        self.drawing = False
        self.last_pos = None
        self.space = 1
        # if style is True, normal solid brush
        # if style is False, png brush
        self.style = False
        # load brush style png
        #self.brush = pygame.image.load("brush.png").convert_alpha()
        # set the current brush depends on size
        #self.brush_now = self.brush.subsurface((0,0), (1, 1))
        self.npoints = []

    def start_draw(self, pos):
        self.drawing = True
        self.last_pos = pos
    def end_draw(self):
        self.drawing = False

    def set_brush_style(self, style):
        print "* set brush style to", style
        self.style = style
    def get_brush_style(self):
        return self.style

    def set_size(self, size):
        if size < 0.5: size = 0.5
        elif size > 50: size = 50
        print "* set brush size to", size
        self.size = size
        #self.brush_now = self.brush.subsurface((0,0), (size*2, size*2))
    def get_size(self):
        return self.size

    def draw(self, pos):
        if self.drawing:
            for p in self._get_points(pos):
                # draw eveypoint between them
                self.npoints.append(p)
                if self.style == False:
                    pygame.draw.circle(self.screen,
                            self.color, p, self.size)
                else:
                    self.screen.blit(self.brush_now, p)

            self.last_pos = pos

    def _get_points(self, pos):
        """ Get all points between last_point ~ now_point. """
        points = [ (self.last_pos[0], self.last_pos[1]) ]
        len_x = pos[0] - self.last_pos[0]
        len_y = pos[1] - self.last_pos[1]
        length = math.sqrt(len_x ** 2 + len_y ** 2)
        step_x = len_x / length
        step_y = len_y / length
        for i in xrange(int(length)):
            points.append(
                    (points[-1][0] + step_x, points[-1][1] + step_y))
        points = map(lambda x:(int(0.5+x[0]), int(0.5+x[1])), points)
        # return light-weight, uniq list
        return list(set(points))
 


def show_text(surface_handle, pos, text, color, font_bold = False, font_size = 13, font_italic = False): 
    '''
    Function:文字处理函数
    Input：surface_handle：surface句柄
           pos：文字显示位置
           color:文字颜色
           font_bold:是否加粗
           font_size:字体大小
           font_italic:是否斜体
    Output: NONE
    author: socrates
    blog:http://blog.csdn.net/dyx1024
    date:2012-04-15
    '''       
    #获取系统字体，并设置文字大小
    cur_font = pygame.font.SysFont("宋体", font_size)
    
    #设置是否加粗属性
    cur_font.set_bold(font_bold)
    
    #设置是否斜体属性
    cur_font.set_italic(font_italic)
    
    #设置文字内容
    text_fmt = cur_font.render(text, 1, color)
    
    #绘制文字
    surface_handle.blit(text_fmt, pos)  

class Painter():
    WIDTH = 200
    HEIGHT = 200
    GRID = 10
    def __init__(self):
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.init()
        pygame.display.set_caption("Drawer")
        self.clock = pygame.time.Clock()
        self.brush = Brush(self.screen)
        self.net = BPNetwork()
        #self.net.load("net.bin")
 
    def run(self):
        self.screen.fill((255, 255, 255))
        while True:
            # max fps limit
            self.clock.tick(30)
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
                elif event.type == KEYDOWN:
                    # press esc to clear screen
                    if event.key == K_ESCAPE:
                        self.screen.fill((255, 255, 255))
                elif event.type == MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.brush.start_draw(event.pos)
                    else:
                        z = np.zeros((10, 10))
                        for p in self.brush.npoints:
                            x = int(round(p[0] * 1.0 * self.GRID / self.WIDTH))
                            y = int(round(p[1] * 1.0 * self.GRID / self.HEIGHT))
                            try:
                                z[y, x] = 1
                            except:
                                pass
                        if LEARNING_NUM != -1:
                            fout = open("tdata.txt", 'a')
                            m = self.GRID * self.GRID
                            buf = "%d\n" % LEARNING_NUM
                            for r in range(self.GRID):
                                for c in range(self.GRID):
                                    buf += "%d " % z[r, c]
                                buf += "\n"
                            fout.write(buf)
                            print buf
                            fout.close()
                            self.screen.fill((255, 255, 255))
                            self.brush.npoints = []
                elif event.type == MOUSEMOTION:
                    self.brush.draw(event.pos)
                elif event.type == MOUSEBUTTONUP:
                    z = np.zeros((10, 10))
                    for p in self.brush.npoints:
                        x = int(round(p[0] * 1.0 * self.GRID / self.WIDTH))
                        y = int(round(p[1] * 1.0 * self.GRID / self.HEIGHT))
                        try:
                            z[y, x] = 1
                        except:
                            pass
                    try:
                        x = z.reshape(1, self.GRID * self.GRID)
                        p = self.net.predict(x)
                        y = np.argmax(p, axis = 1)
                        show_text(self.screen, (0, 0), "%d" % y, (255,0,0), True, 40)
                    except:
                        pass
                    self.brush.end_draw()
 
            pygame.display.update()
 
if __name__ == '__main__':
    thread1 = Thread()
    thread1.setDaemon(True) #主线程退出时, 子线程退出
    thread1.start()

    app = Painter()
    app.run()

