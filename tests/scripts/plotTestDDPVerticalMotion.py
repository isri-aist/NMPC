#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class PlotTestDDPVerticalMotion(object):
    def __init__(self, result_file_path):
        self.result_data_list = np.genfromtxt(result_file_path, dtype=None, delimiter=None, names=True)
        print("[PlotTestDDPVerticalMotion] Load {}".format(result_file_path))

        fig = plt.figure()
        plt.rcParams["font.size"] = 16

        ax = fig.add_subplot(211)
        ax.plot(self.result_data_list["time"], self.result_data_list["pos"],
                color="red", label="planned pos")
        ax.plot(self.result_data_list["time"], self.result_data_list["ref_pos"],
                color="blue", linestyle="dashed", label="ref pos")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("pos [m]")
        ax.grid()
        ax.legend(loc="upper left")

        ax = fig.add_subplot(212)
        ax.plot(self.result_data_list["time"], self.result_data_list["force"],
                color="green", label="force")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("force [N]")
        ax.grid()
        ax.legend(loc="upper left")

        plt.show()


if __name__ == "__main__":
    result_file_path = "/tmp/TestDDPVerticalMotionResult.txt"

    import sys
    if len(sys.argv) >= 2:
        result_file_path = sys.argv[1]

    plot = PlotTestDDPVerticalMotion(result_file_path)
