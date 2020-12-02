import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
from Artery_Class import *
import Global
from utils import *
plt.rcParams.update({'font.size': 12})


# -- Global variables
t = Global.t
T = Global.T
pi = Global.pi


# -- Initialize the artery
xStart = 0
xEnd = 15
L = xEnd - xStart
nx = 4
beta = fe.Expression("E*h*pow(pi,0.5)",E=3e6,h=0.05,degree=1)
A0 = fe.Expression("pi*r*r",r=0.5,degree=1)
Arty1 = Artery(xStart, xEnd, nx, beta, A0)

# -- Initial condition
# Initial pressure pulse
Pin = 2e4*np.sin(2*pi*t/T) * np.heaviside(T/2-t,1)
# FEniCS expressions
Ain  = fe.Expression("Ain" , Ain=0, degree=1)
Aout = fe.Expression("Aout", Aout=0, degree=1)
Qout = fe.Expression("Qout", Qout=0, degree=1)

def bcL(x, on_boundary):
    return on_boundary and x[0]-xStart < fe.DOLFIN_EPS
def bcR(x, on_boundary):
    return on_boundary and xEnd-x[0]   < fe.DOLFIN_EPS

BC_AL = fe.DirichletBC( Arty1.W.sub(1), Ain , bcL )
BC_AR = fe.DirichletBC( Arty1.W.sub(1), Aout, bcR )
BC_QR = fe.DirichletBC( Arty1.W.sub(0), Qout, bcR )

# W2R = -4 * (betaR/(2*rho*A0R))**(1/2) * A0R**(1/4)










# -- Test
# print(np.shape(  Arty1.beta.vector() ))
# print("Left: nx",Arty1.beta.vector()[nx])
# print("Right: 0",Arty1.beta.vector()[0])

(AL,AR) = Arty1.getBoundaryA()
(QL,QR) = Arty1.getBoundaryQ()
print("AL: %f   AR: %f" % (AL,AR) )
print("QL: %f   QR: %f" % (QL,QR) )



# fe.plot(Arty1.beta)
# plt.show()





# sol0 = Arty1.previousSol
# (Q0,A0) = fe.split(sol0)
# fe.plot(A0)
# plt.show()



# # # -- Test 
# x = np.linspace(xStart, xEnd, nx)
# fe.plot(Arty1.W2R)
# # fe.plot(Arty1.A0)
# # fe.plot(Arty1.dA0)
# plt.show()



