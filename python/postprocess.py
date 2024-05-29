import numpy as np
import matplotlib.pyplot as plt
import loadDataFile

def plot_vesicle_data(fileName):
    # Load data
    vesx, vesy, time, N, nv, xinit, yinit = loadDataFile(fileName)

    # Iterate through each time step
    for it in range(len(time)):
        plt.figure(1)
        plt.clf()
        
        # Concatenate the first point to the end to close the loop
        x = np.vstack((vesx[:, :, it], vesx[0, :, it]))
        y = np.vstack((vesy[:, :, it], vesy[0, :, it]))
        
        plt.plot(x, y, 'r', linewidth=2)
        plt.plot(x[0], y[0], 'ko', markersize=10, markerfacecolor='k')
        plt.plot(np.mean(x), np.mean(y), 'ko', markerfacecolor='k')
        plt.axis('equal')
        
        plt.pause(0.1)

