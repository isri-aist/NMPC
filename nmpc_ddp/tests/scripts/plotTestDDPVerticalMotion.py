#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class PlotTestDDPVerticalMotion(object):
    def __init__(self, result_file_path):
        self.result_data_list = np.genfromtxt(result_file_path, dtype=None, delimiter=None, names=True)
        print("[PlotTestDDPVerticalMotion] Load {}".format(result_file_path))

        fig = plt.figure()
        plt.rcParams["font.size"] = 16

        # Preprocess the number of contacts for setting background color
        num_contact_list = self.result_data_list["num_contact"]
        num_contact_diff_list = (num_contact_list[1:] - num_contact_list[:-1]).astype(bool)
        num_contact_diff_list = np.insert(num_contact_diff_list, 0, False)
        num_contact_switch_idx_list = np.where(num_contact_diff_list)[0]
        num_contact_switch_idx_list = np.insert(num_contact_switch_idx_list, 0, 0)
        num_contact_switch_idx_list = np.insert(num_contact_switch_idx_list, len(num_contact_switch_idx_list), -1)
        num_contact_colors = ["blue", "green", "yellow"]

        ax = fig.add_subplot(211)
        ax.plot(self.result_data_list["time"], self.result_data_list["pos"],
                color="red", label="planned pos")
        ax.plot(self.result_data_list["time"], self.result_data_list["ref_pos"],
                color="blue", linestyle="dashed", label="ref pos")
        for start_idx, end_idx in zip(num_contact_switch_idx_list[:-1], num_contact_switch_idx_list[1:]):
            num_contact = num_contact_list[start_idx]
            if start_idx != np.where(num_contact_list==num_contact)[0][0]:
                label = None
            elif num_contact == 0:
                label = "no contact"
            elif num_contact == 1:
                label = "1 contact"
            elif num_contact >= 2:
                label = "{} contacts".format(num_contact)
            ax.axvspan(self.result_data_list["time"][start_idx], self.result_data_list["time"][end_idx],
                       facecolor=num_contact_colors[num_contact], alpha=0.2, label=label)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("pos [m]")
        ax.grid()
        ax.legend(loc="lower left")

        ax = fig.add_subplot(212)
        ax.plot(self.result_data_list["time"], self.result_data_list["force"],
                color="green", label="force")
        for start_idx, end_idx in zip(num_contact_switch_idx_list[:-1], num_contact_switch_idx_list[1:]):
            ax.axvspan(self.result_data_list["time"][start_idx], self.result_data_list["time"][end_idx],
                       facecolor=num_contact_colors[num_contact_list[start_idx]], alpha=0.2)
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
