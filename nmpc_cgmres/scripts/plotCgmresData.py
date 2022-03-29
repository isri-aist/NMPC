#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class CgmresDataPlotter(object):
    def __init__(self):
        self.loadData()
        self.setupPlot()

    def loadData(self, file_dir="/tmp"):
        print("load temporary files in " + file_dir + " directory")

        # set traj
        self.x_traj = np.genfromtxt(file_dir+"/cgmres_x.dat", delimiter=",")
        self.u_traj = np.genfromtxt(file_dir+"/cgmres_u.dat", delimiter=",")
        self.err_traj = np.genfromtxt(file_dir+"/cgmres_err.dat", delimiter=",")

    def setupPlot(self):
        self.fig = plt.figure(figsize=plt.figaspect(1))
        self.fig.canvas.set_window_title("CgmresDataPlotter")

    def plot(self):
        ax_num = np.max([self.x_traj.shape[1], self.u_traj.shape[1]]) - 1
        first_ax = None

        # x
        for i in range(self.x_traj.shape[1] - 1):
            ax = self.fig.add_subplot(3, ax_num, i + 1,
                                      xlabel="time", ylabel="x[{}]".format(i+1),
                                      **{} if first_ax is None else {"sharex": first_ax})
            ax.plot(self.x_traj[:,0], self.x_traj[:,i+1])
            if first_ax is None:
                first_ax = ax

        # u
        for i in range(self.u_traj.shape[1] - 1):
            ax = self.fig.add_subplot(3, ax_num, ax_num + i + 1, sharex=first_ax,
                                      xlabel="time", ylabel="u[{}]".format(i+1))
            ax.plot(self.u_traj[:,0], self.u_traj[:,i+1])

        # err
        for i in range(self.err_traj.shape[1] - 1):
            ax = self.fig.add_subplot(3, ax_num, 2 * ax_num + i + 1, sharex=first_ax,
                                      xlabel="time", ylabel="err[{}]".format(i+1))
            ax.plot(self.err_traj[:,0], self.err_traj[:,i+1])

        plt.pause(0.01)


if __name__ == "__main__":
    plotter = CgmresDataPlotter()
    plotter.plot()
    plt.show()
