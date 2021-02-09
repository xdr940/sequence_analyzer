
from matplotlib.animation import FuncAnimation  # 动图的核心函数
import numpy as np
import matplotlib.pyplot as plt
class Drawer():
    def __init__(self):
        self.fig, self.ax = plt.subplots()

        self.timer = self.fig.canvas.new_timer(interval=100)
        self.ax.set_ylim([0, 2000])
        self.POINTS = 100
        self.npys = np.load('/home/roit/aws/aprojects/sequence_analyzer/scripts/sequences/error_maps_hist/vsd_uav0000121_00516_s_hist.npy')
        self.index= 0
        self.line = self.npys[0,0]
        self.length = self.npys.shape[0]
        pass
    def real_time_draw(self):
        self.ax.grid(True)
        self.hanlder, = self.ax.plot(range(self.POINTS), self.line, label='Sin() output', color='cornflowerblue')




        self.timer.add_callback(self.single_draw, self.ax)
        self.timer.start()
        plt.show()


    def single_draw(self,ax):
        if self.index == self.length-1:
            self.index = 0
        self.index += 1
        self.line = self.npys[self.index][0]
        self.hanlder.set_ydata(self.line)
        ax.draw_artist(self.hanlder)
        ax.set(xlabel='time (s)', ylabel='voltage (mV)',
               title='frame {}'.format(self.index))
        ax.figure.canvas.draw()



if __name__ == '__main__':
    drawer = Drawer()
    drawer.real_time_draw()