import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rcParams.update({'font.size': 12})

pi = np.pi
theta = 0.5
dx = fe.dx
ds = fe.ds
# -- Domain
L = 15
ne = 2**7
mesh = fe.IntervalMesh(int(ne),0,L)
T = 2*0.165
nt = 1e3
time = np.linspace(0,T/2+(0.25-0.165),int(nt))
dt = time[1]-time[0]
# -- Initial condition
r0 = 0.5
A0 = pi*r0**2
q0 = 0
k1 = 2e7
k2 = -22.53
k3 = 8.65e5
f = fe.Expression('4.0/3.0*(k1*exp(k2*r0) + k3)', degree=1, k1=k1, k2=k2, k3=k3, r0=r0)

# -- Function space
degQ = 1 ; degA = 1;
QE     = fe.FiniteElement("Lagrange", cell=mesh.ufl_cell(), degree=degQ)
AE     = fe.FiniteElement("Lagrange", cell=mesh.ufl_cell(), degree=degA)
ME     = fe.MixedElement([AE,QE])
V      = fe.FunctionSpace(mesh,ME)
(v1,v2) = fe.TestFunctions(V)
U = fe.Function(V)
A,q = fe.split(U)
Un = fe.Function(V)
Un.assign(fe.Expression( ('A0','q0'), degree=1, A0=A0, q0=q0 ))

# Terms for variational form
U_v_dx = A*v1*dx + q*v2*dx
Un_v_dx = Un[0]*v1*dx + Un[1]*v2*dx
F2_v2_ds = (pow(q, 2)/(A+fe.DOLFIN_EPS)\
           +f*fe.sqrt(A0*(A+fe.DOLFIN_EPS)))*v2*ds
F2_dv2_dx = (pow(q, 2)/(A+fe.DOLFIN_EPS)\
            +f*fe.sqrt(A0*(A+fe.DOLFIN_EPS)))*fe.grad(v2)[0]*dx
dF_v_dx = fe.grad(q)[0]*v1*dx + F2_v2_ds - F2_dv2_dx
Fn_v_ds = (pow(Un[1], 2)/(Un[0])\
          +f*fe.sqrt(A0*(Un[0])))*v2*ds
Fn_dv_dx = (pow(Un[1], 2)/(Un[0])\
           +f*fe.sqrt(A0*(Un[0])))*fe.grad(v2)[0]*dx
dFn_v_dx = fe.grad(Un[1])[0]*v1*dx + Fn_v_ds - Fn_dv_dx

S_v_dx = - 2*sqrt(pi)/db/Re*q/sqrt(A+DOLFIN_EPS)*v2*dx\
       + (2*sqrt(A+DOLFIN_EPS)*(sqrt(pi)*f
                               +sqrt(A0)*dfdr)\
         -(A+DOLFIN_EPS)*dfdr)*drdx*v2*dx
# Sn_v_dx = -2*sqrt(pi)/db/Re*Un[1]/sqrt(Un[0])*v2*dx\
#         + (2*sqrt(Un[0])*(sqrt(pi)*f+sqrt(A0)*dfdr)\
#           -(Un[0])*dfdr)*drdx*v2*dx

# # # Variational form
# # variational_form = U_v_dx\
# #     - Un_v_dx\
# #     + dt*theta*dF_v_dx\
# #     + dt*(1-theta)*dFn_v_dx\
# #     - dt*theta*S_v_dx\
# #     - dt*(1-theta)*Sn_v_dx
