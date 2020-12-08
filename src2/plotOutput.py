import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rcParams.update({'font.size': 12})

for i in range(0,8000,100):
    A = np.load("output/ThetaNewton_out/A_"+str(i)+".npy")
    plt.plot(A)
    plt.ylim([0.6,1])
    plt.pause(0.0001)
    plt.cla()
