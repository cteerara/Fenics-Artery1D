import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rcParams.update({'font.size': 14})

# A = np.load('Prosthesis_A.npy')
# A = np.load('NoProsthesis_A.npy')
A = np.load('YBifurcation_A.npy')
print(A.shape)

nt = A.shape[0]
Amiddle = np.zeros(( 3,nt) )
for i in range(0,nt):
    Amiddle[0,i] = A[i,int(A.shape[1]/2),0]
    Amiddle[1,i] = A[i,int(A.shape[1]/2),1]
    Amiddle[2,i] = A[i,int(A.shape[1]/2),2]

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


A1 = beta* ( np.sqrt(Amiddle[0,:]) - np.sqrt(A0))/A0
A2 = beta* ( np.sqrt(Amiddle[1,:]) - np.sqrt(A0))/A0
A3 = beta* ( np.sqrt(Amiddle[2,:]) - np.sqrt(A0))/A0

plt.figure(figsize=[10,6])
# plt.plot(time,A1,'r',label='$\\Omega_1$')
# plt.plot(time,A2,'b',label='$\\Omega_2$')
# plt.plot(time,A3,'g',label='$\\Omega_3$')
plt.plot(time,Pin)
plt.xlabel('Time (s)')
plt.ylabel('Pressure $dyn \\cdot cm^{-2}$')
# plt.legend()
plt.show()

