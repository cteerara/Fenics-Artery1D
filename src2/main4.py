import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rcParams.update({'font.size': 12})
# -- MPI variables
fe.set_log_level(40)
comm = fe.MPI.comm_world
rank = fe.MPI.rank(comm)
nPE  = fe.MPI.size(comm)

# -- Physical parameters
pi = np.pi
r0 = 0.5
Sini = np.pi*r0**2
E = 3e6
h0 = 0.05
beta = E*h0*np.sqrt(pi) 
rho = 1
nu = 0.046
KR = 8*pi*nu
alpha = 1


# -- Domain
L = 15
ne = 2**7
mesh = fe.IntervalMesh(int(ne),0,L)
T = 2*0.165
nt = 1e3
time = np.linspace(0,T/2+(0.25-0.165),int(nt))
dt = time[1]-time[0]
time = np.array(time)
Pin = 2e4*np.sin(2*np.pi*time/T) * np.heaviside(T/2-time,1)
Ain = (Pin*Sini/beta+np.sqrt(Sini))**2;
# plt.plot(time,Ain)

# -- Function space
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
(ATrial,QTrial) = fe.TrialFunctions(V)
U0 = fe.Function(V)
U0.assign(fe.Expression( ('A0','q0'), degree=1, A0=Sini, q0=0 ))
Un = fe.Function(V)
Un.assign(fe.Expression( ('A0','0'), degree=1, A0=Sini, q0=0 ))
ndof = len(np.array(Un.vector()))
Ao = U0[0]
Qo = U0[1]
An = Un[0]
Qn = Un[1]


# -- Boundary conditions

def bcL(x, on_boundary):
    return on_boundary and x[0] < fe.DOLFIN_EPS
def bcR(x, on_boundary):
    return on_boundary and L-x[0]   < fe.DOLFIN_EPS
AinBC  = fe.Expression("Ain",Ain=Sini,degree=1)
AoutBC = fe.Expression("Aout",Aout=0, degree=1)
QoutBC = fe.Expression("Qout",Qout=0, degree=1)
bc1 = fe.DirichletBC(V_A, AinBC  , bcL)
bc2 = fe.DirichletBC(V_A, AoutBC , bcR)
bc3 = fe.DirichletBC(V_Q, QoutBC , bcR)
bcs = []
bcs.append(bc1)
bcs.append(bc2)
bcs.append(bc3)

W2R = -4*np.sqrt(beta/(2*rho*Sini))*Sini**(1./4.)


# # -- Solve
# problem = fe.NonlinearVariationalProblem(wf, Un, bcs, J=J)
# solver = fe.NonlinearVariationalSolver(problem)

# -- Test

H11 = fe.Constant(0) 
H12 = fe.Constant(1)
H21 = -alpha*Qn**2/An**2 + beta/(2*rho*Sini)*fe.sqrt(An) 
H22 = 2*alpha*Qn/An
G1 = fe.Constant(0)
G2 = KR*Qn/An + An/(Sini*rho)
C11 = fe.Constant(0) ; C12 = fe.Constant(0)
C21 = KR*Qn/An**2 ; C22 = fe.Constant(0)

# H11 = fe.Constant(0) 
# H12 = fe.Constant(1)
# H21 = fe.Constant(0) 
# H22 = fe.Constant(0) 
# G1 = fe.Constant(0)
# G2 = fe.Constant(0) 
# C11 = fe.Constant(0) ; C12 = fe.Constant(0)
# C21 = fe.Constant(0) ; C22 = fe.Constant(0)

# Residue
R  = dt * v1 * ( H11*fe.grad(An)[0] + H12*fe.grad(Qn)[0] )
R += dt * v2 * ( H21*fe.grad(An)[0] + H22*fe.grad(Qn)[0] )
R += -dt*v1*G1 - dt*v2*G2
R += v1*(An-Ao) + v2*(Qn-Qo)

# Tangent matrix
K = v1*ATrial + v2*QTrial 
K += dt*v1* ( H11*fe.grad(ATrial)[0] + H12*fe.grad(QTrial)[0] ) 
K += dt*v2* ( H21*fe.grad(ATrial)[0] + H22*fe.grad(QTrial)[0] )
K -= dt*v1* ( C11*ATrial + C12*QTrial)
K -= dt*v2* ( C21*ATrial + C22*QTrial)
R *= fe.dx
K *= fe.dx

# Diagnostic
def prettyPrint(A):
    import pandas as pd
    pd.set_option("display.precision",8)
    Table = pd.DataFrame(A)
    Table.columns = ['']*Table.shape[1]
    print(Table.to_string(index=False))

print( "dt : %f\nbeta : %f\nSini : %f\nW2R : %f\n" % (dt, beta, Sini, W2R) )
# RR = fe.assemble(R)
# KK = fe.assemble(K)
# prettyPrint(RR.get_local())
# prettyPrint(np.array(KK.array()))

dU = fe.Function(V)
tid = 0
itMax = 100
tol = 1e-4
print("beta",beta)
print("rho",rho)
print("A0",Sini)
for t in time: # Time loop
    for NR_it in range(0,itMax): # Newton-Raphson loop

        if NR_it == 0:
            Residue0 = np.linalg.norm(fe.assemble(R).get_local())

        (SR0, QR0) = U0.split(deepcopy=True)
        QR0 = QR0.compute_vertex_values()[-1]
        SR0 = SR0.compute_vertex_values()[-1]

        c = np.sqrt(beta/(2*rho*Sini))*SR0**(1./4.)
        lamR0 = alpha*QR0/SR0 + np.sqrt(c**2+alpha*(alpha-1)*QR0**2/SR0**2)
        xW1R = fe.Point(L-lamR0*dt,0,0)

        (AR,QR) = Un.split(deepcopy=True)
        AR = AR(xW1R)
        QR = QR(xW1R)
        W1R = QR/AR + 4*np.sqrt(beta/(2*rho*Sini))*AR**(1./4.)
        SRBC = (2*rho*Sini/beta)**2 * ((W1R-W2R)/8)**4
        QRBC = SRBC*(W1R+W2R)/2

        # print("AR",AR)
        # print("QR",QR)
        # print("xW1R",xW1R.x())
        # print("W1R",W1R)
        # print("W2R",W2R)
        # print("SRBC",SRBC)
        # print("QRBC",QRBC)
        # sys.exit()

        # Apply boundary conditions
        (AR,QR) = Un.split(deepcopy=True)
        Un.vector().vec().setValueLocal( ndof-2 , Ain[tid]  )
        Un.vector().vec().setValueLocal( 0 , SRBC )
        Un.vector().vec().setValueLocal( 1 , QRBC )
        AinBC.Ain   = 0#-AR.compute_vertex_values()[0] + Ain[tid]
        AoutBC.Aout = 0 #-AR.compute_vertex_values()[-1] + SRBC
        QoutBC.Qout = 0 #-QR.compute_vertex_values()[-1] + QRBC

        # print("Ain BC")
        # print("A: ",AR.compute_vertex_values()[1])

        # prettyPrint(fe.assemble(R).get_local())
        # prettyPrint(np.array(fe.assemble(K).array()) )
        # print("AinBC:",AinBC.Ain)
        # print("SRBC:",AoutBC.Aout)
        # print(QR.compute_vertex_values()[-1])
        # print(QRBC)
        # print("QRBC:",QoutBC.Qout)


        fe.solve(K==-R, dU, [bc1,bc2,bc3])
        # print(AoutBC.Aout)
        # print(QoutBC.Qout)
        # print(c)
        # print(lamR0)
        # print(xW1R.x())
        # print(SRBC)
        # print(QRBC)
        (dA,dQ) = dU.split(deepcopy=True)

        # prettyPrint(dA.compute_vertex_values())
        # prettyPrint(dQ.compute_vertex_values())

        Un.assign(Un+dU)
        # (A_Updated,Q_Updated) = Un.split(deepcopy=True)
        # prettyPrint(A_Updated.compute_vertex_values())
        # prettyPrint(Q_Updated.compute_vertex_values())
        Residuen = np.linalg.norm(fe.assemble(R).get_local())
        print("Timestep:",tid,"Iteration:", NR_it,"Relative residue:",Residuen/Residue0)
        if (Residuen/Residue0) < tol:
            break
        # prettyPrint(A_Updated.compute_vertex_values())

    U0.assign(Un)
    (A_Output, Q_Output) = Un.split(deepcopy=True)
    np.save("output/A_"+str(tid)+".npy", A_Output.compute_vertex_values())
    np.save("output/Q_"+str(tid)+".npy", Q_Output.compute_vertex_values())
    if tid%20 == 0:
        plt.cla()
        fe.plot(A_Output)
        plt.ylim([0.5,1])
        plt.title("t="+str(t))
        plt.pause(0.5)
    tid += 1
    
