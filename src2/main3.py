import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rcParams.update({'font.size': 12})

# -- MPI variables
comm = fe.MPI.comm_world
rank = fe.MPI.rank(comm)
nPE  = fe.MPI.size(comm)

# -- Physical parameters
pi = np.pi
r0 = 0.5
A0 = np.pi*r0**2
E = 3e6
h0 = 0.05
beta = 4*pi*h0*E/(3*A0) 
rho = 1
nu = 0.035
KR = 8*pi*nu
alpha = 1

# -- Domain
L = 5
ne = 2**7
mesh = fe.IntervalMesh(int(ne),0,L)
T = 2*0.165
# nt = 1e3
# time = np.linspace(0,T/2+(0.25-0.165),int(nt))
# dt = time[1]-time[0]
dt = 2e-6
time = []
ti = 0
while ti < T/2+(0.25-0.165):
    time.append(ti)
    ti += dt
time = np.array(time)
Pin = 2e4*np.sin(2*np.pi*time/T) * np.heaviside(T/2-time,1)
Ain = ( Pin/beta + np.sqrt(A0) )**2
# plt.plot(time,Pin)
# plt.show()

# -- Function space
degQ = 1 ; degA = 1;
QE     = fe.FiniteElement("Lagrange", cell=mesh.ufl_cell(), degree=degQ)
AE     = fe.FiniteElement("Lagrange", cell=mesh.ufl_cell(), degree=degA)
ME     = fe.MixedElement([AE,QE])
V      = fe.FunctionSpace(mesh,ME)
(v1,v2) = fe.TestFunctions(V)
dv1 = fe.grad(v1)[0]
dv2 = fe.grad(v2)[0]
( ATrial , QTrial ) = fe.TrialFunctions(V)
U = fe.Function(V)
U.assign(fe.Expression( ('A0','q0'), degree=1, A0=A0, q0=0 ))
A = U[0]
Q = U[1]
dA = fe.grad(A)[0]
dQ = fe.grad(Q)[0]
c = np.sqrt(beta/(2*rho))*A**(0.25)
c2 = c**2
ubar = Q/A

# -- Terms
def matMult(A,x):
    return [ A[0][0]*x[0] + A[0][1]*x[1] , A[1][0]*x[0] + A[1][1]*x[1] ]
F = [ Q**2/A , beta/(3*rho)*A**(3./2)  ]
B = [ 0      , -KR*Q/A ]
FU = [ [ 0, 1 ] , [c2-ubar**2, 2*ubar] ]
BU = [ [0,0],[KR*Q/A**2, -KR/A] ]
dFdz = matMult(FU,[dA,dQ])
FLW1 = F[0] + dt/2*matMult(FU,B)[0]
FLW2 = F[1] + dt/2*matMult(FU,B)[1]
BLW1 = B[0] + dt/2*matMult(BU,B)[0]
BLW2 = B[1] + dt/2*matMult(BU,B)[1]
FUdFdz1 = matMult(FU,dFdz)[0]
FUdFdz2 = matMult(FU,dFdz)[1]
BUdFdz1 = matMult(BU,dFdz)[0]
BUdFdz2 = matMult(BU,dFdz)[1]


# FLW1 = Q - KR*dt/2*ubar
# FLW2 = beta/(3*rho) * A**(1.5) - KR*dt*ubar**2 + Q**2/A
# BLW2 = KR**2*dt/2 * Q/A**2 - KR*ubar
# BUdFdz2 = KR*ubar/A*dQ - KR*c2/A*dA + KR*ubar**2/A*dA - KR*2*ubar/A*dQ
# FUdFdz1 = c2*dA - ubar**2*dA + 2*ubar*dQ
# FUdFdz2 = (c2-ubar**2)*dQ + 2*ubar*(c2*dA - ubar**2*dA + 2*ubar*dQ)

# -- Weakform
wf  = (ATrial*v1 + QTrial*v2)
# wf1 = A*v1 + dt*FLW1*dv1 - dt**2/2*FUdFdz1*dv1 
# wf2 = Q*v2 + dt*FLW2*dv2 - dt**2/2*BUdFdz2*v2 - dt**2/2*FUdFdz2*dv2 + dt*BLW2*v2
wf1 = A*v1 + dt*FLW1*dv1 - dt**2/2*BUdFdz1*v1 - dt**2/2*FUdFdz1*dv1 + dt*BLW1*v1
wf2 = Q*v2 + dt*FLW2*dv2 - dt**2/2*BUdFdz2*v2 - dt**2/2*FUdFdz2*dv2 + dt*BLW2*v2
wf -= wf1 + wf2
wf  = wf * fe.dx
lhs = fe.lhs(wf)
rhs = fe.rhs(wf)

# -- Boundary condition
def bcL(x, on_boundary):
    return on_boundary and x[0] < fe.DOLFIN_EPS
def bcR(x, on_boundary):
    return on_boundary and L-x[0]   < fe.DOLFIN_EPS

AL = fe.Expression("AL",AL=0,degree=1)
AR = fe.Expression("AR",AR=A0,degree=1)
QR = fe.Expression("QR",QR=0,degree=1)
bc1 = fe.DirichletBC(V.sub(0), AL, bcL)
bc2 = fe.DirichletBC(V.sub(0), AR, bcR)
bc3 = fe.DirichletBC(V.sub(1), QR, bcR)
# bcs = [bc1,bc2]
bcs = [bc1, bc2,bc3]
USol = fe.Function(V)
W2R = -2*np.sqrt(2*beta/rho)*A0**(0.25) 

tid = 0
nt = 2
for t in time:
# t = 0
# for i in range(0,nt):
    (AA,QQ) = U.split(deepcopy=True)
    Aend  = AA.compute_vertex_values()[-1]
    Qend  = QQ.compute_vertex_values()[-1]
    cend  = np.sqrt( beta/(2*rho)  ) * Aend**(0.25) 
    lam1R = Qend/Aend + cend 
    xW1R = fe.Point(L - lam1R*dt, 0, 0)
    W1R = QQ(xW1R)/AA(xW1R) + 2*np.sqrt(2*beta/rho)* AA(xW1R)**(0.25)
    ARBC = (rho/beta)**2 * (W1R-W2R)**4/4**5
    QRBC = ARBC*(W1R+W2R)/2
    AL.AL = Ain[tid]
    AR.AR = ARBC
    QR.QR = QRBC
    print("t:",t,"xW1R:",xW1R.x(),"W1R:",W1R,"W2R:",W2R,"ARBC:",ARBC,"QRBC:",QRBC)
    fe.solve(lhs == rhs, USol, bcs) 
    U.assign(USol)
    fe.plot(AA)
    plt.pause(0.5)
    # plt.cla()
    # print(QQ.compute_vertex_values())
    tid += 1



