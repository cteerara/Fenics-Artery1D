import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import *
from Global import *

class Artery:
    '''
    This class contains the artery's properties 
    '''
    def __init__(self, L, ne, r0, Q0, E, h0, degA=1, degQ=1):

        # -- Setup domain
        self.L = L
        self.ne = ne
        self.mesh = fe.IntervalMesh( ne, 0, L )
        QE = fe.FiniteElement("Lagrange", cell=self.mesh.ufl_cell(), degree=degQ)
        AE = fe.FiniteElement("Lagrange", cell=self.mesh.ufl_cell(), degree=degA)
        ME = fe.MixedElement([AE,QE])

        # -- Setup functionspaces
        self.V  = fe.FunctionSpace(self.mesh,ME)
        self.V_A = self.V.sub(0)
        self.V_Q = self.V.sub(1)
        (self.v1,self.v2) = fe.TestFunctions(self.V)
        self.dv1 = fe.grad(self.v1)[0]
        self.dv2 = fe.grad(self.v2)[0]
        self.E  = E
        self.h0 = h0
        self.beta = E*h0*np.sqrt(np.pi)
        self.A0 = np.pi*r0**2
        self.Q0 = Q0
        self.U0 = fe.Function(self.V)
        self.Un = fe.Function(self.V)

        # -- Setup initial conditions
        self.U0.assign( fe.Expression( ( 'A0', 'Q0' ) , A0=self.A0, Q0=Q0, degree=1 ) )
        self.Un.assign( fe.Expression( ( 'A0', 'Q0' ) , A0=self.A0, Q0=Q0, degree=1 ) )
        (self.u01, self.u02) = fe.split(self.U0)
        (self.un1, self.un2) = fe.split(self.Un)
        self.du01 = fe.grad(self.u01)[0] 
        self.du02 = fe.grad(self.u02)[0]
        self.dun1 = fe.grad(self.un1)[0] 
        self.dun2 = fe.grad(self.un2)[0]

        # -- Setup weakform terms
        B0     = self.getB(self.u01, self.u02)
        Bn     = self.getB(self.un1, self.un2)
        H0     = self.getH(self.u01, self.u02)
        Hn     = self.getH(self.un1, self.un2)
        HdUdz0 = matMult(H0,[self.du01, self.du02])
        HdUdzn = matMult(Hn,[self.dun1, self.dun2])

        # -- Setup initial condition
        theta = 0.5
        nt = 1000
        self.nt = nt
        T = 2*0.165
        time = np.linspace(0,(T/2+(0.25-0.165)),int(nt))
        dt = time[1]-time[0]
        self.time = time
        self.dt = dt
        time = np.array(time)
        Pin = 2e4*np.sin(2*np.pi*time/T) * np.heaviside(T/2-time,1)
        self.Ain = (Pin*self.A0/self.beta+np.sqrt(self.A0))**2;

        # -- Setup weakform
        wf  = -self.un1*self.v1 - self.un2*self.v2
        wf +=  self.u01*self.v1 + self.u02*self.v2
        wf += -dt*theta     * ( HdUdzn[0] + Bn[0] )*self.v1 - dt*theta     * ( HdUdzn[1] + Bn[1] )*self.v2 
        wf += -dt*(1-theta) * ( HdUdz0[0] + B0[0] )*self.v1 - dt*(1-theta) * ( HdUdz0[1] + B0[1] )*self.v2  
        wf = wf*fe.dx
        self.wf = wf
        self.J = fe.derivative(wf, self.Un, fe.TrialFunction(self.V))

    
    def getB(self,A,Q):
        return [0,KR*Q/A]

    def getH(self,A,Q):
        beta = self.beta
        A0 = self.A0
        H11=0 ; H12 = 1
        H21 = -alpha*(Q/A)**2+beta/(2*A0*rho)*A**(0.5) ; H22 = 2*Q/A*alpha
        return [ [H11,H12], [H21,H22] ]


    def getWavespeed(self,A):
        return np.sqrt(self.beta/(2*rho*self.A0))*A**(1./4.)

    def getCharacteristics(self,A,Q):
        c  = self.getWavespeed(A)
        W1 = Q/A + 4*c
        W2 = Q/A - 4*c
        return (W1,W2)

    def getAQfromChar(self,W1,W2):
        A = (2*rho*self.A0/self.beta)**2 * ( (W1-W2)/8 )**4
        Q = A * (W1+W2)/2
        return (A,Q)




    def solve(self):

        def bcL(x, on_boundary):
            return on_boundary and x[0] < fe.DOLFIN_EPS
        def bcR(x, on_boundary):
            return on_boundary and self.L-x[0] < fe.DOLFIN_EPS

        self.AinBC  = fe.Expression("Ain"  , Ain =self.A0 , degree=1)
        self.AoutBC = fe.Expression("Aout" , Aout=self.A0 , degree=1)
        self.QoutBC = fe.Expression("Qout" , Qout=0       , degree=1) 
        bc1 = fe.DirichletBC(self.V_A, self.AinBC  , bcL)
        bc2 = fe.DirichletBC(self.V_A, self.AoutBC , bcR)
        bc3 = fe.DirichletBC(self.V_Q, self.QoutBC , bcR)
        bcs = [bc1, bc2, bc3]
        (tmp,W2R) = self.getCharacteristics(self.A0,self.Q0) 
        
        # -- Setup problem
        problem = fe.NonlinearVariationalProblem(self.wf, self.Un, bcs, J=self.J)
        self.solver = fe.NonlinearVariationalSolver(problem)

        # -- Solve
        tid = 0
        for t in self.time:

            # -- Get nonreflecting bc
            SR0 = self.U0.compute_vertex_values()[self.ne]
            QR0 = self.U0.compute_vertex_values()[2*self.ne+1]
            c = self.getWavespeed(SR0)
            # c = np.sqrt(self.beta/(2*rho*self.A0))*SR0**(1./4.)
            lamR0 = alpha*QR0/SR0 + np.sqrt(c**2+alpha*(alpha-1)*(QR0/SR0)**2)
            xW1R = fe.Point(self.L-lamR0*self.dt,0,0)
            (AR,QR) = self.U0.split()
            AR = AR(xW1R)
            QR = QR(xW1R)
            (W1R,tmp) = self.getCharacteristics(AR,QR)
            (ARBC, QRBC) = self.getAQfromChar(W1R,W2R)

            # -- Apply boundary conditions
            self.AinBC.Ain   = self.Ain[tid]
            self.AoutBC.Aout = ARBC
            self.QoutBC.Qout = QRBC
            self.solver.solve()
            self.U0.assign(self.Un)
            (Asol,Qsol) = self.Un.split()
            print('Timestep %d out of %d completed' % (tid,self.nt))
            tid += 1
        
            # -- Plot for diagnostics
            plt.plot(Asol.compute_vertex_values())
            plt.ylim([0.6,1.1])
            plt.pause(0.01)
            plt.cla()
        
        





