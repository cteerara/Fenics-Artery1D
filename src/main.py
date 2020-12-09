import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rcParams.update({'font.size': 12})
from Global import *
from Artery_class import *

fe.set_log_level(40)

# -- Constants
L = 15
ne = 2**7
r0 = 0.5
E = 3e6
h0 = 0.05
Q0 = 0

# -- Time values
theta = 0.5
nt = 1000
T = 2*0.165
time = np.linspace(0,(T/2+(0.25-0.165)),int(nt))
dt = time[1]-time[0]
Pin = 2e4*np.sin(2*np.pi*time/T) * np.heaviside(T/2-time,1)
A0 = np.pi*r0**2
beta = E*h0*np.sqrt(np.pi)
Ain = (Pin*A0/beta+np.sqrt(A0))**2;


A = Artery(L, ne, r0, Q0,    E, h0, theta, dt, degA=1,degQ=1)
# Arty1 = Artery(L, ne, r0, Q0, 10*E, h0, theta, dt, degA=1,degQ=1)

tid = 0
for t in time:


    (ARBC,QRBC) = A.getNoReflectionBC()

    A.solve(Ain=Ain[tid],Aout=ARBC,Qout=QRBC)
    A.plotSol("A")
    plt.ylim([0.6,1.1])
    plt.pause(0.01)
    plt.cla()
    print("Timestep: %d out of %d completed" % (tid,nt))
    tid += 1


