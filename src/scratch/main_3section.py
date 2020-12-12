import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rcParams.update({'font.size': 12})
from Global import *
from Artery_class import *

fe.set_log_level(40)

# -- Constants
L = 5
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

Artyp = Artery(L, ne, r0, Q0, E, h0, theta, dt, degA=degA,degQ=degQ)
Arty1 = Artery(L, ne, r0, Q0, 100*E, h0, theta, dt, degA=degA,degQ=degQ)
Arty2 = Artery(L, ne, r0, Q0, E, h0, theta, dt, degA=degA,degQ=degQ)


def getBCs( P, D  ):
    gamma_p = P.beta/P.A0
    gamma_d = D.beta/D.A0
    sigma_p = 4*np.sqrt(P.beta/(2*rho*P.A0))
    sigma_d = 4*np.sqrt(D.beta/(2*rho*D.A0))
    K = np.zeros( (4,4) )
    R = np.zeros(4)
    (A_p,Q_p) = P.getBoundaryAQ("right")
    (A_d,Q_d) = P.getBoundaryAQ("left")
    (lam1_p, lam2_p) = P.getEigenvalues(A_p,Q_p)
    (lam1_d, lam2_d) = D.getEigenvalues(A_d,Q_d)
    (W1_p, W2_p) = P.getCharacteristics( P.getAatPoint( P.L - lam1_p*P.dt ) , P.getQatPoint( P.L - lam1_p*P.dt ) )
    (W1_d, W2_d) = D.getCharacteristics( D.getAatPoint(     - lam2_d*D.dt ) , D.getQatPoint(     - lam2_d*D.dt ) )
    NR_itmax = 1000
    tol = 1e-5
    for NR_it in range(0,NR_itmax):
        ptp = gamma_p*( np.sqrt(A_p) - np.sqrt(P.A0) ) + 1/2*(Q_p/A_p)**2
        ptd = gamma_d*( np.sqrt(A_d) - np.sqrt(D.A0) ) + 1/2*(Q_d/A_d)**2
        R[0] = Q_p - Q_d
        R[1] = ptp - ptd
        R[2] = alpha*Q_p/A_p + sigma_p*A_p**(1./4) - W1_p
        R[3] = alpha*Q_d/A_d - sigma_d*A_d**(1./4) - W2_d
        K[0,0] =  1 
        K[0,1] = -1

        K[1,0] =  Q_p/A_p**2
        K[1,1] = -Q_d/A_d**2
        K[1,2] = -Q_p**2/A_p**3 + gamma_p/(2*np.sqrt(A_p))
        K[1,3] =  Q_d**2/A_d**3 - gamma_d/(2*np.sqrt(A_d))

        K[2,0] = alpha/A_p
        K[2,2] = -Q_p*alpha/A_p**2 + sigma_p/(4*A_p**(3./4))

        K[3,1] = alpha/A_d
        K[3,3] = -Q_d*alpha/A_d**2 - sigma_d/(4*A_d**(3./4))

        if np.linalg.norm(R) < tol:
            break
        dU = np.linalg.solve(K,-R )
        Q_p += dU[0] ; Q_d += dU[1]
        A_p += dU[2] ; A_d += dU[3]

    Aout_p = A_p
    Qout_p = Q_p
    Ain_d  = A_d
    Qin_d  = Q_d
    (Aout_d, Qout_d) = D.getNoReflectionBC()

    return ( Aout_p, Qout_p, Ain_d, Qin_d, Aout_d, Qout_d  )


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
for t in time:

    (Aout_p, Qout_p, Ain_1, Qin_1, tmp1, tmp2) = getBCs( Artyp, Arty1)
    (Aout_1, Qout_1, Ain_2, Qin_2, Aout_2, Qout_2) = getBCs( Arty1, Arty2)


    Artyp.solve( Ain=Ain[tid] , Qin=None  , Aout=Aout_p , Qout=Qout_p )
    Arty1.solve( Ain=Ain_1    , Qin=Qin_1 , Aout=Aout_1 , Qout=Qout_1 )
    Arty2.solve( Ain=Ain_2    , Qin=Qin_2 , Aout=Aout_2 , Qout=Qout_2 )


    Asol_p = Artyp.getSol("A").compute_vertex_values()
    Asol_1 = Arty1.getSol("A").compute_vertex_values()
    Asol_2 = Arty2.getSol("A").compute_vertex_values()
    Asol = np.hstack( (Asol_p, Asol_1[1:], Asol_2[1:])  )
    


    plt.plot(Asol)
    plt.ylim([0.6,1.1])
    plt.pause(1e-6)
    plt.cla()

    # plt.plot(Qsol)
    # plt.ylim([-60,60])
    # plt.pause(1e-6)
    # plt.cla()

    # print("Timestep: %d out of %d completed" % (tid,nt))

    tid += 1


