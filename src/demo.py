import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
from Global import *
from Artery_Network_class import *

fe.set_log_level(40)

# -- Set up constants
theta = 0.5
nt = 1000
r0 = 0.5
E = 3e6
h0 = 0.05
T = 2*0.165
A0 = np.pi*r0**2
time = np.linspace(0,(T/2+(0.25-0.165)),int(nt))
dt = time[1]-time[0]

# -- Create inlet boundary conditions
freq = 1
Pin = 2e4*np.sin(2*np.pi*time/T*freq) * np.heaviside(T/freq/2-time,1)
beta = E*h0*np.sqrt(np.pi)
Ainlet = (Pin*A0/beta+np.sqrt(A0))**2;

# -- Create arterial network
inputFile = sys.argv[1]
ArteryNetwork = Artery_Network(inputFile,dt,theta, nt)

# -- Solve
tid = 0
for t in time:
    print("Solving at timestep %d" % (tid))
    (Ain,Qin,Aout,Qout) = ArteryNetwork.getBoundaryConditions(Ainlet[tid])
    ArteryNetwork.solve( Ain, Aout, Qin, Qout, tid )
    tid +=1
ArteryNetwork.saveOutput_matlab('YBifurcation')




