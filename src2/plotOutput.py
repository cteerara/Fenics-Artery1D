import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rcParams.update({'font.size': 12})

for i in range(0,1000):
    A = np.load("output/Q_"+str(i)+".npy")
    plt.plot(A)
    plt.ylim([0,40])
    plt.pause(0.001)
    plt.cla()
