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
r0 = 0.5
A0 = np.pi*r0**2
E = 3e6
h0 = 0.05
beta = E*h0*np.sqrt(np.pi)
rho = 1
nu = 0.035
KR = 8*np.pi*nu
alpha = 1

# -- Domain
L = 1
ne = 1
mesh = fe.IntervalMesh(int(ne),0,L)
T = 2*0.165
nt = 1e3
time = np.linspace(0,T/2+(0.25-0.165),int(nt))
dt = time[1]-time[0]

# -- Function space
degQ = 1 ; degA = 1;
QE     = fe.FiniteElement("Lagrange", cell=mesh.ufl_cell(), degree=degQ)
AE     = fe.FiniteElement("Lagrange", cell=mesh.ufl_cell(), degree=degA)
ME     = fe.MixedElement([AE,QE])
V      = fe.FunctionSpace(mesh,ME)
( v1     , v2     ) = fe.TestFunctions(V)
dv1 = fe.grad(v1)[0]
dv2 = fe.grad(v2)[0]
( ATrial , QTrial ) = fe.TrialFunctions(V)
U = fe.Function(V)

# -- Initial conditions
def interpolate(e):
    return fe.interpolate(e, V.sub(0).collapse())
fe.assign( U.sub(0) , interpolate(fe.Expression("A0",A0=A0,degree=1)) )
W2R = -4*np.sqrt(beta/(2*rho*A0))*A0**(0.25)
Pin = 2e4*np.sin(2*np.pi*time/T) * np.heaviside(T/2-time,1)


# -- Weakform
zero = interpolate(fe.Constant(0))
one  = interpolate(fe.Constant(1))
(A,Q) = fe.split(U) 
# F
F1 = Q
F2 = alpha*Q**2/A + beta/(3*rho*A0) * A**(3./2.) 
# B
B1 = zero 
B2 = KR*Q/A
# H
c = (beta/(2*rho*A0))**(0.5) * A**(0.25)
H11 = zero                   ; H12 = one
H21 = -alpha*(Q/A)**2 + c**2 ; H22 = 2*alpha*Q/A
# BU
BU11 = zero       ; BU12 = zero
BU21 = -KR*Q/A**2 ; BU22 = KR/A
# FLW
FLW1 = F1 + dt/2. * ( H11  * B1  +  H12  * B2 ) 
FLW2 = F2 + dt/2. * ( H21  * B1  +  H22  * B2 )
# BLW
BLW1 = B1 + dt/2. * ( BU11 * B1  +  BU12 * B2 )
BLW2 = B2 + dt/2. * ( BU21 * B1  +  BU22 * B2 )
# dFdz
dFdz1 = fe.grad(F1)[0] 
dFdz2 = fe.grad(F2)[0]
# BU_dFdz
BU_dFdz1 = BU11*dFdz1 + BU12*dFdz2
BU_dFdz2 = BU21*dFdz1 + BU22*dFdz2
# H_dFdz
H_dFdz1  = H11 *dFdz1 + H12 *dFdz2
H_dFdz2  = H21 *dFdz1 + H22 *dFdz2
# Weakform
lhs   = (v1*ATrial + v2*QTrial); lhs*=fe.dx
# rhs  = (dv1*A + dv2*Q)
# rhs +=  dt       * (FLW1    *dv1    + FLW2 *dv2)
# rhs += -dt**2/2. * (BU_dFdz1*v1  + BU_dFdz2*v2 )
# rhs += -dt**2/2. * (H_dFdz1 *dv1 + H_dFdz2 *dv2) 
# rhs +=  dt       * (BLW1    *v1  + BLW2    *v2 )
rhs = v1+v2
rhs *=  fe.dx

# lhs = v1*ATrial*fe.dx
print(fe.assemble(lhs).array())
print(fe.assemble(rhs).get_local())

# # -- Boundary conditions
# def bcL(x, on_boundary):
#     return on_boundary and x[0] < fe.DOLFIN_EPS
# def bcR(x, on_boundary):
#     return on_boundary and L-x[0]   < fe.DOLFIN_EPS
# Ain  = fe.Expression("Ain" ,Ain=0  , degree=1)
# Aout = fe.Expression("Aout",Aout=0 , degree=1)
# Qout = fe.Expression("Qout",Qout=0 , degree=1)
# BC_AL = fe.DirichletBC( V.sub(0), Ain , bcL )
# BC_AR = fe.DirichletBC( V.sub(0), Aout, bcR )
# BC_QR = fe.DirichletBC( V.sub(1), Qout, bcR )
# bcs = [BC_AL, BC_AR, BC_QR]
# # bcs = [BC_AL, BC_AR]#], BC_QR]

# # -- Setup problem and solver
# u_ = fe.Function(V)
# problem = fe.LinearVariationalProblem(lhs, rhs, u_, bcs)
# solver = fe.LinearVariationalSolver(problem)

# # -- Solve
# tid = 0
# for t in time:
# # for t in range(0,2):
#     p = Pin[tid]
#     Ain.Ain = ( p*A0/beta + np.sqrt(A0) )**2
#     # current A and Q
#     (Acur,Qcur) = U.split(deepcopy=True)
#     Aend  = Acur.compute_vertex_values()[-1]
#     Qend  = Qcur.compute_vertex_values()[-1]
#     cend  = np.sqrt( beta/(2*rho*A0)  ) * Aend**(0.25) 
#     lam1R = alpha*Qend/Aend + np.sqrt( cend * alpha*(alpha-1)*(Qend/Aend)**2  )
#     print(lam1R)
#     xW1R  = fe.Point(L - lam1R*dt,0,0)
#     print(L - lam1R*dt)
#     AR = Acur(xW1R)
#     QR = Qcur(xW1R)
#     W1R = QR/AR + 4*np.sqrt(beta/(2*rho*A0))*AR**(0.25)
#     ARBC = (2*rho*A0/beta)**2 * ((W1R-W2R)/8)**4
#     QRBC = ARBC*(W1R+W2R)/2
#     # print(ARBC)
#     # print(QRBC)
#     Aout.Aout = np.pi*r0**2 
#     Qout.Qout = QRBC
#     solver.solve()
#     tid += 1
#     U.assign(u_)
#     fe.plot(A)
#     plt.pause(0.5)



# plt.show()









