import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rcParams.update({'font.size': 12})
from Global import *
from Artery_Network_class import *

fe.set_log_level(40)

# -- Time values
theta = 0.5
nt = 1000
r0 = 0.5
E = 3e6
h0 = 0.05
T = 2*0.165
A0 = np.pi*r0**2
time = np.linspace(0,(T/2+(0.25-0.165)),int(nt))
# time = np.linspace(0,(T/2),int(nt))
dt = time[1]-time[0]

freq = 1
Pin = 2e4*np.sin(2*np.pi*time/T*freq) * np.heaviside(T/freq/2-time,1)
beta = E*h0*np.sqrt(np.pi)
Ainlet = (Pin*A0/beta+np.sqrt(A0))**2;
# plt.plot(time,Pin)
# plt.show()


inputFile = sys.argv[1]
ArteryNetwork = Artery_Network(inputFile,dt,theta, nt)

tid = 0
for t in time:
    print("Solving at timestep %d" % (tid))
    (Ain,Qin,Aout,Qout) = ArteryNetwork.getBoundaryConditions(Ainlet[tid])
    ArteryNetwork.solve( Ain, Aout, Qin, Qout, tid )

    if tid % 10 == 0:
        plt.subplot(3,1,1)
        Asol_0 = ArteryNetwork.Arteries[0].getSol("A").compute_vertex_values()
        plt.plot(Asol_0)
        plt.title("Vessel 0 tid="+str(tid))
        plt.ylim([0.6,1.1])
        plt.subplot(3,1,2)
        Asol_1 = ArteryNetwork.Arteries[1].getSol("A").compute_vertex_values()
        plt.plot(Asol_1)
        plt.title("Vessel 1 tid="+str(tid) )
        plt.ylim([0.6,1.1])
        plt.subplot(3,1,3)
        Asol_2 = ArteryNetwork.Arteries[2].getSol("A").compute_vertex_values()
        plt.plot(Asol_2)
        plt.title("Vessel 2 tid="+str(tid))
        plt.ylim([0.6,1.1])
        # plt.subplot(4,1,4)
        # Asol_3 = ArteryNetwork.Arteries[3].getSol("A").compute_vertex_values()
        # plt.plot(Asol_3)
        # plt.title("Vessel 3 tid="+str(tid))
        # plt.ylim([0.6,1.1])
        plt.pause(1e-6)
    plt.clf()

    tid +=1
# ArteryNetwork.saveOutput_matlab('Prosthesis',saveMatlab=True)
ArteryNetwork.saveOutput_matlab('YBifurcation',saveMatlab=True)




