import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import mpl_toolkits.mplot3d

class AnimatedScatter(object):
    def __init__(self, numpoints, box_len, pos, mom, acc, dt, n_t, r_all, do_pressure, updfunc):
        self.numpoints = numpoints
        self.pos = pos
        self.mom = mom
        self.acc = acc
        self.box_len = box_len
        self.dt = dt
        self.n_t = n_t
        self.L = box_len
        self.r_all = r_all
        self.do_pressure = do_pressure
        self.updfunc = updfunc
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ani = ani.FuncAnimation(self.fig, self.update, interval=1, init_func=self.setup, blit=False)

    def setup(self):
        self.pos, self.mom, self.acc = self.updfunc(self.numpoints, self.box_len, self.pos, self.mom, self.acc, self.dt, self.n_t, self.r_all, self.do_pressure)[0:3]
        self.scat = self.ax.scatter(self.pos[0,:],self.pos[1,:],self.pos[2,:])
        self.scat._offsets3d = self.pos
        self.ax.set_xlim(-self.L/2, self.L/2)
        self.ax.set_ylim(-self.L/2, self.L/2)
        self.ax.set_zlim(-self.L/2, self.L/2)
        return self.scat,

    def update(self, i):
        self.pos, self.mom, self.acc = self.updfunc(self.numpoints, self.box_len, self.pos, self.mom, self.acc, self.dt, self.n_t, self.r_all, self.do_pressure)[0:3]
        plt.draw()
        return self.scat,

    def show(self):
        plt.show()

