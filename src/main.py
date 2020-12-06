import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
from Artery_Class import *
import Global
from utils import *
plt.rcParams.update({'font.size': 12})


# -- Global variables
time = Global.time
T = Global.T
pi = Global.pi
alpha = Global.alpha
dt = Global.dt

# -- Initialize the artery
xStart = 0
xEnd = 15
L = xEnd - xStart
nx = 2**10
nx = int(nx)
beta = fe.Expression("E*h*pow(pi,0.5)",E=3e6,h=0.05,degree=1)
A0 = fe.Expression("pi*r*r",r=0.5,degree=1)
# beta = fe.Expression("x[0]*x[0]",degree=1)
# A0 = fe.Expression("x[0]+1",degree=1)
Arty1 = Artery(xStart, xEnd, nx, beta, A0, degQ=1, degA=1)

# # -- Test
# Q = fe.interpolate(fe.Expression("sin(x[0])",degree=1), Arty1.W.sub(1).collapse())
# A = fe.interpolate(fe.Expression("x[0]*x[0]+2",degree=1), Arty1.W.sub(0).collapse())
# F = Arty1.getF(Q,A)
# B = Arty1.getB(Q,A)
# H = Arty1.getH(Q,A)
# BU = Arty1.getBU(Q,A)
# fe.plot(BU[1][0])
# # fe.plot(BU[1][1])
# plt.show()


# -- Initial condition
# Initial pressure pulse
Pin = 2e4*np.sin(2*pi*time/T) * np.heaviside(T/2-time,1)
# FEniCS expressions
Ain  = fe.Expression("Ain" , Ain=0, degree=1)
Aout = fe.Expression("Aout", Aout=0, degree=1)
Qout = fe.Expression("Qout", Qout=0, degree=1)

def bcL(x, on_boundary):
    return on_boundary and x[0]-xStart < fe.DOLFIN_EPS
def bcR(x, on_boundary):
    return on_boundary and xEnd-x[0]   < fe.DOLFIN_EPS

BC_AL = fe.DirichletBC( Arty1.W.sub(0), Ain , bcL )
BC_AR = fe.DirichletBC( Arty1.W.sub(0), Aout, bcR )
BC_QR = fe.DirichletBC( Arty1.W.sub(1), Qout, bcR )
bcs = [BC_AL, BC_AR, BC_QR]

(W1R_ini  , W2R_ini)   = Arty1.getBoundaryCharacteristic("right")
(A0L  , A0R)   = Arty1.getBoundaryValues(Arty1.A0)
(betaL, betaR) = Arty1.getBoundaryValues(Arty1.beta)

wf = Arty1.wf
lhs = fe.lhs(wf)
rhs = fe.rhs(wf)
problem = fe.LinearVariationalProblem(lhs, rhs, Arty1.nextSol, bcs)
solver = fe.LinearVariationalSolver(problem)

# for i in range(0,len(time)):
for i in range(0,10):
# for i in range(0,1):
    # Plot solution
    (Asol, Qsol) = Arty1.currentSol.split(deepcopy=True)
    plt.plot(Asol.compute_vertex_values())
    plt.pause(0.5)
    print("Asol at i=%d" % i)
    print(Asol.compute_vertex_values())
    print("")
    t = time[i]
    # Apply inlet boundary
    Ain.Ain = Global.P2A(betaL, A0L, Pin[i]) 
    (lambdaRight_pos, lambdaRight_neg) = Arty1.getBoundaryLambda("right")
    xW1R = xEnd - lambdaRight_pos*dt
    # Get non reflecting characteristics at x=xEnd-lambdaRight_pos*dt
    (W1BC, W2BC) = Arty1.getCharacteristicAtPt(xW1R)
    W2R = W2R_ini
    W1R = W1BC
    # Get Q and A from characteristics at W1R and W2R
    betaXW1R = Arty1.interpolate( Arty1.beta , xW1R )
    A0XW1R   = Arty1.interpolate( Arty1.A0   , xW1R )
    
    # # Non reflecting BC
    # (Qout.Qout, Aout.Aout) = Arty1.getAQFromChar( betaXW1R, A0XW1R, W1R, W2R  )
    # # Fixed BC
    Aout.Aout = np.pi*0.5**2
    solver.solve()
    Arty1.updateSol()


    

# plt.plot(time,np.array(AinArr))
# plt.show()







# -- Test
# print(np.shape(  Arty1.beta.vector() ))
# print("Left: nx",Arty1.beta.vector()[nx])
# print("Right: 0",Arty1.beta.vector()[0])

# (AL,AR) = Arty1.getBoundaryA()
# (QL,QR) = Arty1.getBoundaryQ()
# print("AL: %f   AR: %f" % (AL,AR) )
# print("QL: %f   QR: %f" % (QL,QR) )



# fe.plot(Arty1.beta)
# plt.show()





# sol0 = Arty1.currentSol
# (Q0,A0) = fe.split(sol0)
# fe.plot(A0)
# plt.show()



# # # -- Test 
# x = np.linspace(xStart, xEnd, nx)
# fe.plot(Arty1.W2R)
# # fe.plot(Arty1.A0)
# # fe.plot(Arty1.dA0)
# plt.show()


