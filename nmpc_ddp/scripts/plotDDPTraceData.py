#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class PlotDDPTraceData(object):
    def __init__(self, trace_file_path):
        self.trace_data_list = np.genfromtxt(trace_file_path, dtype=None, delimiter=' ', names=True)
        print("[PlotDDPTraceData] Load {}".format(trace_file_path))

    def __call__(self, key_list, x_axis_iter=True):
        if isinstance(key_list, str):
            key_list = [key_list]

        # Check key
        for key in key_list:
            if key not in self.trace_data_list.dtype.names:
                print("[PlotDDPTraceData] {} is not contained in key list. Key list is {}".format(key, self.trace_data_list.dtype.names))
                return

        # Plot
        for key in key_list:
          if x_axis_iter:
              plt.plot(self.trace_data_list["iter"], self.trace_data_list[key], marker='o', label=key)
          else:
              plt.plot(self.trace_data_list[key], marker='o', label=key)

        # Show
        if x_axis_iter:
            plt.xlabel("iter")
        else:
            plt.xlabel("index")
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: {} <trace_file_path>".format(sys.argv[0]))
        sys.exit(1)

    plot = PlotDDPTraceData(sys.argv[1])
    print("[PlotDDPTraceData] Run \"plot(<key>)\" or \"plot([<key>, <key>, ...])\". Select the key from the following:\n{}".format(
        plot.trace_data_list.dtype.names))

    import code
    code.interact(local=locals())
