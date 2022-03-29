#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class PlotTestDDPBipedal(object):
    def __init__(self, result_file_path):
        self.result_data_list = np.genfromtxt(result_file_path, dtype=None, delimiter=None, names=True)
        print("[PlotTestDDPBipedal] Load {}".format(result_file_path))

        fig = plt.figure()
        plt.rcParams["font.size"] = 16

        ax = fig.add_subplot(211)
        ax.plot(self.result_data_list["time"], self.result_data_list["planned_zmp"],
                color="red", label="planned ZMP")
        ax.plot(self.result_data_list["time"], self.result_data_list["ref_zmp"],
                color="blue", linestyle="dashed", label="ref ZMP")
        ax.plot(self.result_data_list["time"], self.result_data_list["com_pos"],
                color="green", label="CoM")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("pos [m]")
        ax.grid()
        ax.legend(loc="upper right")

        ax = fig.add_subplot(212)
        ax.plot(self.result_data_list["time"], self.result_data_list["omega2"], color="green", label="omega^2")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("omega^2 [1/s^2]")
        ax.grid()
        ax.legend(loc="upper right")

        plt.show()


if __name__ == "__main__":
    result_file_path = "/tmp/TestDDPBipedalResult.txt"

    import sys
    if len(sys.argv) >= 2:
        result_file_path = sys.argv[1]

    plot = PlotTestDDPBipedal(result_file_path)
