#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    # By default, read from 'epileptor_output.dat' if no filename is provided
    filename = sys.argv[1] if len(sys.argv) > 1 else 'epileptor_output.txt'

    # Load the 3-column data: time, var1, var2
    data = np.loadtxt(filename)
    t    = data[:, 0]
    var1 = data[:, 1]
    var2 = data[:, 2]

    # Create the plot
    plt.plot(t, var1, label='Variable 1')
    plt.plot(t, var2, label='Variable 2')
    plt.xlabel('Time')
    plt.ylabel('Variables')
    plt.title('Epileptor Simulation')
    plt.legend()

    # Save plot to file
    plt.savefig('epileptor_plot.png')
    print(f"Plot saved to epileptor_plot.png")

if __name__ == "__main__":
    main()
