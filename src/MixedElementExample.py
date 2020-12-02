import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rcParams.update({'font.size': 12})

n = 20 # Number of interval
a = 0 # Starting point
b = 1 # End Point
mesh = fe.IntervalMesh(n,a,b)



QE = fe.FiniteElement("Lagrange", cell=mesh.ufl_cell(), degree=2)
PE = fe.FiniteElement("Lagrange", cell=mesh.ufl_cell(), degree=1)
ME = fe.MixedElement([QE,PE])
W = fe.FunctionSpace(mesh,ME)
(w,q) = fe.TestFunction(W)
(u,a) = fe.TrialFunction(W)
x = fe.Expression("x[0]",degree=1)
x2 = fe.Expression("x[0]*x[0]",degree=1)

# wf = fe.grad(u)[0]*fe.grad(w)[0]*fe.dx - w*x*fe.dx + fe.grad(a)[0]*fe.grad(q)[0]*fe.dx - q*x2*fe.dx
wf = fe.grad(u)[0]*w*fe.dx - w*x*fe.dx + fe.grad(a)[0]*q*fe.dx - q*x2*fe.dx
lhs = fe.lhs(wf)
rhs = fe.rhs(wf)


def bcL(x, on_boundary):
    return on_boundary and x[0] < fe.DOLFIN_EPS

def bcR(x, on_boundary):
    return on_boundary and 1-x[0] < fe.DOLFIN_EPS

zero = fe.Constant(0)
one = fe.Constant(1)
BC1 = fe.DirichletBC(W.sub(0), zero, bcL)
# BC2 = fe.DirichletBC(W.sub(0), one, bcR)
BC3 = fe.DirichletBC(W.sub(1), zero, bcL)
# BC4 = fe.DirichletBC(W.sub(1), one, bcR)
# bcs = [BC1, BC2, BC3, BC4]
bcs = [BC1,BC3]

sol = fe.Function(W)
fe.solve(lhs == rhs, sol, bcs)
xarr = np.linspace(0,1,10)
fsol = xarr**3/3
# fsol = 1/6*(5*xarr + xarr**3)
fe.plot(sol[1])
plt.plot(xarr,fsol,'.')
plt.show()

