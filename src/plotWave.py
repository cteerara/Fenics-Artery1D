import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
plt.rcParams.update({'font.size': 14})

# -- Load area and flowrate (A,Q) over time
# A and Q is an array where A.shape = ( numTimeStep, numPoints, numSegments )
tag = sys.argv[1]
A = np.load( tag+tag[:-1]+'_A.npy' )
Q = np.load( tag+tag[:-1]+'_Q.npy' )
print(A.shape)

# -- Load vessel data
# vessel data is a dictionary of vessel properties. Each entry is a list of length numSegments
# these are the same values as specified in the input file
vesselData = pickle.load( open(tag+tag[:-1]+"_properties.pkl", "rb") )
print(vesselData)
numVessels = len(vesselData['L'])


# -- Recreate inlet pressure condition for plotting
# This must be exactly the same as the one specified in demo.py
r0 = vesselData['r0'][0]
E = vesselData['E'][0]
h0 = vesselData['h0'][0]
T = 2*0.165
A0 = np.pi*r0**2
nt = A.shape[0]
time = np.linspace(0,(T/2+(0.25-0.165)),int(nt))
dt = time[1]-time[0]
beta = E*h0*np.sqrt(np.pi)
freq = 2
Pin = 2e4*np.sin(2*np.pi*time/T*freq/2) * np.heaviside(T/freq-time,1)

# -- Convert area array to pressure array
P = np.zeros( A.shape )
for i, Ei, h0i in zip(range(numVessels), vesselData['E'], vesselData['h0']):
    beta = Ei*h0i*np.sqrt(np.pi)
    P[:,:,i] = beta* ( np.sqrt(A[:,:,i]) - np.sqrt(A0) )/A0

# Plot pressure propagation over time
L = np.cumsum(vesselData['L'])
x0 = np.linspace(0,L[0],A.shape[1])
x1 = np.linspace(L[0],L[1],A.shape[1])
x2 = np.linspace(L[1],L[2],A.shape[1])
for i in range(0,P.shape[0],10):
    if i%1 !=0: continue
    plt.plot(x0, P[i,:,0], color='r', label='$\\Omega_1$')
    plt.plot(x1, P[i,:,1], color='b', label='$\\Omega_2$')
    plt.plot(x2, P[i,:,2], color='k', label='$\\Omega_3$')
    plt.title("Time = %f seconds" % (time[i]))
    plt.legend()
    plt.ylabel('Pressure $dyn \\cdot cm^{-2}$')
    plt.ylim([-2e4, 2e4])
    plt.pause(0.01)
    plt.cla()


# Plot pressure profile over time at the middle of the domain
plt.figure(figsize=[10,6])
plt.plot(time, P[:,P.shape[1]//2,0], 'r', label='$\\Omega_1$')
plt.plot(time, P[:,P.shape[1]//2,1], 'b', label='$\\Omega_2$')
plt.plot(time, P[:,P.shape[1]//2,2], 'g', label='$\\Omega_3$')
plt.plot(time, Pin)
plt.title("Pressure at vessel center over time")
plt.xlabel('Time (s)')
plt.ylabel('Pressure $dyn \\cdot cm^{-2}$')
plt.legend()
plt.show()

