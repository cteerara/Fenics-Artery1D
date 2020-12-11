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
plt.plot(Ain)
plt.show()

degA = 1
degQ = 1

Artyp = Artery(L, ne, r0, Q0,   E, h0, theta, dt, degA=degA,degQ=degQ)
Arty1 = Artery(L, ne, r0, Q0, E, h0, theta, dt, degA=degA,degQ=degQ)

# -- Constants
beta_p = Artyp.beta
beta_1 = Arty1.beta
A0_p   = Artyp.A0
A0_1   = Arty1.A0
gamma_p= beta_p/A0_p
gamma_1= beta_1/A0_1
sigma_p= 4*np.sqrt(beta_p/(2*rho*A0_p))
sigma_1= 4*np.sqrt(beta_1/(2*rho*A0_1))

# print("gamma",[gamma_p,gamma_1])
# print("sigma",[sigma_p,sigma_1])

# -- Stiffness matrix
K = np.zeros((4,4))
R = np.zeros(4)

tid = 0
# for t in time:
for i in range(0,1):
    # -- Get boundary conditions for the problem.
    # Here we are using a segmented domain. The right segment has higher young's modulus
    # -- Initial guess
    (A_p,Q_p) = Artyp.getBoundaryAQ("right")
    (A_1,Q_1) = Arty1.getBoundaryAQ("left")

    # Q_p = 2
    # Q_1 = 4

    # Here I am getting the characteristics at the previous timestep 
    # from the compatibility condition via characteristic extrapolation
    # Positive characteristic on the right for parent vessel
    # Negative characteristic on the left for daughter vessel
    (lam1_p, lam2_p) = Artyp.getEigenvalues(A_p,Q_p)
    (lam1_1, lam2_1) = Arty1.getEigenvalues(A_1,Q_1)
    (W1_p, W2_p) = Artyp.getCharacteristics( Artyp.getAatPoint( Artyp.L - lam1_p*Artyp.dt ) , Artyp.getQatPoint( Artyp.L - lam1_p*Artyp.dt ) )
    (W1_1, W2_1) = Arty1.getCharacteristics( Arty1.getAatPoint(         - lam2_1*Arty1.dt ) , Arty1.getQatPoint(         - lam2_1*Arty1.dt ) )
    # -- Inside NR loop
    NR_itmax = 1000
    tol = 1e-5
    for NR_it in range(0,NR_itmax):
        ptp = gamma_p*( np.sqrt(A_p) - np.sqrt(A0_p) ) + 1/2*(Q_p/A_p)**2
        pt1 = gamma_1*( np.sqrt(A_1) - np.sqrt(A0_1) ) + 1/2*(Q_1/A_1)**2
        K[0,0]=1
        K[0,1]=-1
        K[1,0]=Q_p/A_p**2
        K[1,1]=-Q_1/A_1**2
        K[1,2]=-Q_p**2/A_p**3+gamma_p/(2*np.sqrt(A_p))
        K[1,3]= Q_1**2/A_1**3-gamma_1/(2*np.sqrt(A_1))
        K[2,0]=alpha/A_p
        K[2,2]=-alpha*Q_p/A_p**2+sigma_p/(4*A_p**(3/4))
        K[3,1]=alpha/A_1
        K[3,3]=-alpha*Q_1/A_1**2-sigma_1/(4*A_1**(3/4))
        R[0] = Q_p - Q_1
        R[1] = ptp - pt1 
        R[2] = alpha*Q_p/A_p + sigma_p*A_p**(1/4) - W1_p
        R[3] = alpha*Q_1/A_1 - sigma_1*A_1**(1/4) - W2_1
        print("Time step: %d. NR iteration: %d. Residue = %f" % (tid, NR_it, np.linalg.norm(R)))
        if np.linalg.norm(R) < tol:
            break
        dU = np.linalg.solve(K,-R )
        Q_p += dU[0] ; Q_1 += dU[1]
        A_p += dU[2] ; A_1 += dU[3]


    (ANoReflect,QNoReflect) = Arty1.getNoReflectionBC()

    print("Ain ",[Ain[tid],A_1])
    print("Aout",[A_p,ANoReflect])
    print("Qin ",[None,Q_1])
    print("Qout",[Q_p,QNoReflect])
    print('')

    Artyp.solve(Ain=Ain[tid],Aout=A_p,Qout=Q_p)
    Arty1.solve(Ain=A_1, Qin=Q_1, Aout = ANoReflect, Qout = QNoReflect)


    Asol_p = Artyp.getSol("A").compute_vertex_values()
    Asol_1 = Arty1.getSol("A").compute_vertex_values()
    Asol = np.hstack( (Asol_p, Asol_1[1:])  )
    Qsol_p = Artyp.getSol("Q").compute_vertex_values()
    Qsol_1 = Arty1.getSol("Q").compute_vertex_values()
    # Qsol = np.hstack( ( Qsol_p, Qsol_1[1:]  ) )
    Qsol = np.hstack( ( Qsol_p, Qsol_1  ) )
    


    plt.plot(Asol_p)
    plt.show()
    # plt.ylim([0.6,1.1])
    # plt.pause(1e-6)
    # plt.cla()

    # plt.plot(Qsol)
    # plt.ylim([-60,60])
    # plt.pause(1e-6)
    # plt.cla()

    # print("Timestep: %d out of %d completed" % (tid,nt))

    tid += 1


