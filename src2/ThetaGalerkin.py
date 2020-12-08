import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rcParams.update({'font.size': 12})


# -- Constants
rho = 1
r0 = 0.5
alpha = 1
E = 3e6
h0 = 0.05
beta = E*h0*np.sqrt(np.pi)
nu = 0.046
KR = 8*np.pi*nu
A0 = np.pi*r0**2
Q0 = 0

def matMult(A,x):
    return [ A[0][0]*x[0]+A[0][1]*x[1] , A[1][0]*x[0] + A[1][1]*x[1]  ]
def getF(A,Q):
    F = [Q , alpha*Q**2/A + beta*A**(1.5)/(3*rho*A0) ]
    return F
def getH(A,Q):
    H11 = 0
    H12 = 1
    H21 = -alpha*(Q/A)**2 + beta/(2*A0*rho)*A**(0.5)
    H22 = 2*Q/A*alpha
    return [ [H11,H12], [H21,H22] ]
def getB(A,Q):
    return [0,KR*Q/A ]#+ A/(A0*rho)]
def getBU(A,Q):
    BU11 = 0
    BU12 = 0
    BU21 = -KR*Q/A**2 #+ 1/(A0*rho)
    BU22 = KR/A
    return [ [BU11,BU12],[BU21,BU22] ]
def getFLW(F,H,B):
    HB = matMult(H,B)
    FLW1 = F[0] + dt/2*HB[0]
    FLW2 = F[1] + dt/2*HB[1]
    return [FLW1, FLW2]
def getBLW(B,BU):
    BUB = matMult(BU,B)
    BLW1 = B[0] + dt/2*BUB[0]
    BLW2 = B[1] + dt/2*BUB[1]
    return [BLW1, BLW2]
def getBUdFdz(BU,H,dA,dQ):
    dFdz = matMult(H,[dA,dQ])
    BU_dFdz = matMult(BU,dFdz)
    return BU_dFdz
def getHdFdz(H,dA,dQ):
    dFdz = matMult(H,[dA,dQ])
    Hu_dFdz = matMult(H,dFdz)
    return Hu_dFdz

# -- Time domain
T = 2*0.165
nt = 1000
time = np.linspace(0,T/2+(0.25-0.165),int(nt))
dt = time[1]-time[0]

ne = 2**7
L = 15
mesh = fe.IntervalMesh(int(ne),0,L)
degQ = 1 ; degA = 1;
QE     = fe.FiniteElement("Lagrange", cell=mesh.ufl_cell(), degree=degQ)
AE     = fe.FiniteElement("Lagrange", cell=mesh.ufl_cell(), degree=degA)
ME     = fe.MixedElement([AE,QE])
V      = fe.FunctionSpace(mesh,ME)
V_A = V.sub(0)
V_Q = V.sub(1)
(v1,v2) = fe.TestFunctions(V)
dv1 = fe.grad(v1)[0]
dv2 = fe.grad(v2)[0]
(u1,u2) = fe.TrialFunctions(V)
U0 = fe.Function(V)
U0.assign( fe.Expression( ( 'A0', 'Q0'  ), A0=A0, Q0=Q0, degree=1 ) )
(u01,u02) = fe.split(U0)

F       = getF(u01,u02)
B       = getB(u01,u02)
H       = getH(u01,u02)
BU      = getBU(u01,u02)
FLW     = getFLW(F,H,B)
BLW     = getBLW(B,BU)
Bu_dFdz = getBUdFdz(BU, H, fe.grad(u01)[0], fe.grad(u02)[0])
Hu_dFdz = getHdFdz (    H, fe.grad(u01)[0], fe.grad(u02)[0])

FLW1     = FLW    [0] ; FLW2     = FLW    [1]
BLW1     = BLW    [0] ; BLW2     = BLW    [1]
Bu_dFdz1 = Bu_dFdz[0] ; Bu_dFdz2 = Bu_dFdz[1] 
Hu_dFdz1 = Hu_dFdz[0] ; Hu_dFdz2 = Hu_dFdz[1] 

lhs = u1*v1 + u2*v2 
lhs = lhs*fe.dx
rhs  = u01*v1 + u02*v2 
rhs += dt*FLW1*dv1 + dt*FLW2*dv2
rhs += -dt**2/2*( Bu_dFdz1*v1  + Bu_dFdz2*v2  )
rhs += -dt**2/2*( Hu_dFdz1*dv1 + Hu_dFdz2*dv2 )
rhs += dt*( BLW1*v1 + BLW2*v2 )
rhs  = rhs*fe.dx

# lhs = fe.lhs(wf)
# rhs = fe.rhs(wf)

Pin = 2e4*np.sin(2*np.pi*time/T) * np.heaviside(T/2-time,1)
Ain = (Pin*A0/beta + np.sqrt(A0))**2
# plt.plot(time,Ain)
# plt.show()

def bcL(x, on_boundary):
    return on_boundary and x[0] < fe.DOLFIN_EPS
def bcR(x, on_boundary):
    return on_boundary and L-x[0]   < fe.DOLFIN_EPS
AinBC  = fe.Expression("Ain"  , Ain=A0  , degree=1)
AoutBC = fe.Expression("Aout" , Aout=A0 , degree=1)
QoutBC = fe.Expression("Qout" , Qout=0  , degree=1) 
bc1 = fe.DirichletBC(V_A, AinBC  , bcL)
bc2 = fe.DirichletBC(V_A, AoutBC , bcR)
bc3 = fe.DirichletBC(V_Q, QoutBC , bcR)
bcs = [bc1, bc2, bc3]
W2R = -4*np.sqrt(beta/(2*rho*A0))*A0**(1./4.)


U_ = fe.Function(V)
tid = 0
for t in time:
    
    SR0 = U0.compute_vertex_values()[ne]
    QR0 = U0.compute_vertex_values()[2*ne+1]
    c = np.sqrt(beta/(2*rho*A0))*SR0**(1./4.)
    lamR0 = alpha*QR0/SR0 + np.sqrt(c**2+alpha*(alpha-1)*(QR0/SR0)**2)
    xW1R = fe.Point(L-lamR0*dt,0,0)
    (AR,QR) = U0.split()
    AR = AR(xW1R)
    QR = QR(xW1R)
    W1R = QR/AR + 4*np.sqrt(beta/(2*rho*A0))*AR**(1./4.)
    SRBC = (2*rho*A0/beta)**2 * ( (W1R-W2R)/8 )**4
    QRBC = SRBC + (W1R+W2R)/2
    AinBC.Ain = Ain[tid]
    AoutBC.Aout = SRBC
    QoutBC.Qout = QRBC
    fe.solve(lhs==rhs, U_, bcs)
    U0.assign(U_)
    (A,Q) = U0.split()
    fe.plot(A)
    plt.pause(1)

    tid += 1
