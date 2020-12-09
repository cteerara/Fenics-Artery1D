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


A = Artery(L,ne,r0,Q0,E,h0,theta,dt,degA=1,degQ=1)

tid = 0
for t in time:
    (AR0,QR0) = A.getBoundaryAQ("right" )
    c = A.getWavespeed(AR0)
    (lamR0,tmp) = A.getEigenvalues(AR0,QR0)
    xW1R = fe.Point(A.L - lamR0*dt,0,0)
    (AR,QR) = A.U0.split()
    AR = AR(xW1R)
    QR = QR(xW1R)
    (W1R,tmp) = A.getCharacteristics(AR,QR)
    W2R = A.W2_initial
    (ARBC,QRBC) = A.getAQfromChar(W1R,W2R)
    A.solve(Ain=Ain[tid],Aout=ARBC,Qout=QRBC)

    A.plotSol("A")
    plt.ylim([0.6,1.1])
    plt.pause(0.01)
    plt.cla()

    tid += 1




        # tid = 0
        # for t in self.time:
        #     # -- Get nonreflecting bc
        #     AR0 = self.U0.compute_vertex_values()[self.ne]
        #     QR0 = self.U0.compute_vertex_values()[2*self.ne+1]
        #     c = self.getWavespeed(AR0)
        #     (lamR0,tmp) = self.getEigenvalues(AR0,QR0)
        #     xW1R = fe.Point(self.L-lamR0*self.dt,0,0)
        #     (AR,QR) = self.U0.split()
        #     AR = AR(xW1R)
        #     QR = QR(xW1R)
        #     (W1R,tmp) = self.getCharacteristics(AR,QR)
        #     (ARBC, QRBC) = self.getAQfromChar(W1R,W2R)
        #     # -- Apply boundary conditions
        #     self.AinBC.Ain   = self.Ain[tid]
        #     self.AoutBC.Aout = ARBC
        #     self.QoutBC.Qout = QRBC
        #     self.solver.solve()
        #     self.U0.assign(self.Un)
        #     (Asol,Qsol) = self.Un.split()
        #     print('Timestep %d out of %d completed' % (tid,self.nt))
        #     tid += 1
        #     # -- Plot for diagnostics
        #     plt.plot(Asol.compute_vertex_values())
        #     plt.ylim([0.6,1.1])
        #     plt.pause(0.01)
        #     plt.cla()

