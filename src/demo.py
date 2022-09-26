import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
from arteryNetworkClass import *
import os

fe.set_log_level(40)

# -- Set up time parameters
theta = 0.5
nt = 5000
T = 2*0.165
time = np.linspace(0,(T/2+(0.25-0.165)),int(nt))
dt = time[1]-time[0]

# -- Create pressure/area inlet boundary conditions
# here I am creating a pulse sine wave which stops at T/freq (implemented via heaviside theta) 
# since I am defining my BC as pressure first, I need vessel properties of the inlet vessel (E/r0/h0)
# these are currently hard coded
r0 = 0.5
E = 3e6
h0 = 0.05
A0 = np.pi*r0**2
freq = 2
Pin = 2e4*np.sin(2*np.pi*time/T*freq/2) * np.heaviside(T/freq-time,1)
beta = E*h0*np.sqrt(np.pi)
Ainlet = (Pin*A0/beta+np.sqrt(A0))**2
plt.plot(time, Ainlet)
plt.title('Boundary condition at the inlet')
plt.xlabel('time')
plt.ylabel('Cross-sectional (cm^2)')
plt.show()

# -- Define fluid properties
rho = 1
nu = 0.04
alpha = 1

# -- Create arterial network
inputFile = sys.argv[1]
ArteryNetwork = arteryNetwork(inputFile,dt,theta,nt,rho,nu,alpha)

# -- Solve
tid = 0
for t in time:
    print("Solving at timestep %d" % (tid))
    (Ain,Qin,Aout,Qout) = ArteryNetwork.getBoundaryConditions(Ainlet[tid])
    ArteryNetwork.solve( Ain, Aout, Qin, Qout, tid )
    tid +=1

# -- Save output
tag = inputFile.split("/")[-1].split(".")[0]
os.system( "mkdir -p %s" % tag )
tag = tag+'/'+tag
ArteryNetwork.saveOutput(tag)




