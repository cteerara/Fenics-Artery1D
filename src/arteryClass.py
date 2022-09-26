import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import *

class Artery:
    '''
    This class contains the artery's properties 
    '''
    def __init__(self, L, ne, r0, Q0, E, h0, theta, dt, fluidProp, degA=1, degQ=1):

        # Setup domain
        self.fluidProp = fluidProp
        self.L = L
        self.ne = ne
        self.dt = dt
        self.mesh = fe.IntervalMesh( ne, 0, L )
        QE = fe.FiniteElement("Lagrange", cell=self.mesh.ufl_cell(), degree=degQ)
        AE = fe.FiniteElement("Lagrange", cell=self.mesh.ufl_cell(), degree=degA)
        ME = fe.MixedElement([AE,QE])

        # Setup functionspaces
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

        # Setup initial conditions
        self.U0.assign( fe.Expression( ( 'A0', 'Q0' ) , A0=self.A0, Q0=Q0, degree=1 ) )
        self.Un.assign( fe.Expression( ( 'A0', 'Q0' ) , A0=self.A0, Q0=Q0, degree=1 ) )
        (self.u01, self.u02) = fe.split(self.U0)
        (self.un1, self.un2) = fe.split(self.Un)
        self.du01 = fe.grad(self.u01)[0] 
        self.du02 = fe.grad(self.u02)[0]
        self.dun1 = fe.grad(self.un1)[0] 
        self.dun2 = fe.grad(self.un2)[0]
        (self.W1_initial,self.W2_initial) = self.getCharacteristics(self.A0,self.Q0) 

        # Setup weakform terms
        B0     = self.getB(self.u01, self.u02)
        Bn     = self.getB(self.un1, self.un2)
        H0     = self.getH(self.u01, self.u02)
        Hn     = self.getH(self.un1, self.un2)
        HdUdz0 = matMult(H0,[self.du01, self.du02])
        HdUdzn = matMult(Hn,[self.dun1, self.dun2])

        # Setup weakform
        wf  = -self.un1*self.v1 - self.un2*self.v2
        wf +=  self.u01*self.v1 + self.u02*self.v2
        wf += -dt*theta     * ( HdUdzn[0] + Bn[0] )*self.v1 - dt*theta     * ( HdUdzn[1] + Bn[1] )*self.v2 
        wf += -dt*(1-theta) * ( HdUdz0[0] + B0[0] )*self.v1 - dt*(1-theta) * ( HdUdz0[1] + B0[1] )*self.v2  
        wf = wf*fe.dx
        self.wf = wf
        self.J = fe.derivative(wf, self.Un, fe.TrialFunction(self.V))

    
    def getB(self,A,Q):
        KR = self.fluidProp['KR']
        return [0,KR*Q/A]

    def getH(self,A,Q):
        rho = self.fluidProp['rho']
        alpha = self.fluidProp['alpha']
        beta = self.beta
        A0 = self.A0
        H11=0 ; H12 = 1
        H21 = -alpha*(Q/A)**2+beta/(2*A0*rho)*A**(0.5) ; H22 = 2*Q/A*alpha
        return [ [H11,H12], [H21,H22] ]


    def getWavespeed(self,A):
        rho = self.fluidProp['rho']
        return np.sqrt(self.beta/(2*rho*self.A0))*A**(1./4.)

    def getCharacteristics(self,A,Q):
        alpha = self.fluidProp['alpha']
        c  = self.getWavespeed(A)
        W1 = alpha*Q/A + 4*c
        W2 = alpha*Q/A - 4*c
        return (W1,W2)

    def getAQfromChar(self,W1,W2):
        rho = self.fluidProp['rho']
        A = (2*rho*self.A0/self.beta)**2 * ( (W1-W2)/8 )**4
        Q = A * (W1+W2)/2
        return (A,Q)

    def getEigenvalues(self,A,Q):
        ''' Return eigenvalues of H with input values A and Q '''
        rho = self.fluidProp['rho']
        alpha = self.fluidProp['alpha']
        c = self.getWavespeed(A)
        lam1 = alpha*Q/A + np.sqrt( c**2 + alpha*(alpha-1)*(Q/A)**2 )
        lam2 = alpha*Q/A - np.sqrt( c**2 + alpha*(alpha-1)*(Q/A)**2 )
        return (lam1,lam2)

    def getBoundaryAQ(self,LeftOrRight):
        ''' Return boundary A and Q for current solution field U0 '''
        if LeftOrRight.lower() == 'left':
            A = self.U0.compute_vertex_values()[0]
            Q = self.U0.compute_vertex_values()[self.ne+1]
        elif LeftOrRight.lower() == 'right':
            A = self.U0.compute_vertex_values()[self.ne]
            Q = self.U0.compute_vertex_values()[2*self.ne+1]
        else:
            raise ValueError('LeftOrRight must be either "left" or "right"')
        return (A,Q)

    def getNoReflectionBC(self):
        (AR0,QR0) = self.getBoundaryAQ("right")
        c = self.getWavespeed(AR0)
        (lamR0,tmp) = self.getEigenvalues(AR0,QR0)
        xW1R = fe.Point(self.L-lamR0*self.dt,0,0)
        (AR,QR) = self.U0.split()
        AR = AR(xW1R)
        QR = QR(xW1R)
        (W1R,tmp) = self.getCharacteristics(AR,QR)
        (ARBC,QRBC) = self.getAQfromChar(W1R,self.W2_initial)
        return (ARBC,QRBC)


    def solve(self, Ain=None , Qin=None , Aout=None , Qout=None):

        # -- Define boundaries
        def bcL(x, on_boundary):
            return on_boundary and x[0] < fe.DOLFIN_EPS
        def bcR(x, on_boundary):
            return on_boundary and self.L-x[0] < fe.DOLFIN_EPS

        # -- Define initial conditions
        bcs = []
        if Ain is not None:
            bc_Ain  = fe.DirichletBC(self.V_A, fe.Expression("Ain" ,Ain=Ain  ,degree=1) , bcL)
            bcs.append(bc_Ain)
        if Qin is not None:
            bc_Qin  = fe.DirichletBC(self.V_Q, fe.Expression("Qin" ,Qin=Qin  ,degree=1) , bcL)
            bcs.append(bc_Qin)
        if Aout is not None:
            bc_Aout = fe.DirichletBC(self.V_A, fe.Expression("Aout",Aout=Aout,degree=1) , bcR) 
            bcs.append(bc_Aout)
        if Qout is not None:
            bc_Qout = fe.DirichletBC(self.V_Q, fe.Expression("Qout",Qout=Qout,degree=1) , bcR)
            bcs.append(bc_Qout)

        # -- Setup problem
        problem = fe.NonlinearVariationalProblem(self.wf, self.Un, bcs, J=self.J)
        solver = fe.NonlinearVariationalSolver(problem)

        # -- Solve
        solver.solve()
        self.U0.assign(self.Un)

    # ------------------------------------------
    # --- Getters and Diagnostics --------------
    # ------------------------------------------

    def getAatPoint(self,x):
        ''' Return A at point x '''
        pt = fe.Point(x,0,0)
        (Asol,Qsol) = self.U0.split()
        return Asol(pt)

    def getAatPoint(self,x):
        ''' Return A at point x '''
        pt = fe.Point(x,0,0)
        (Asol,Qsol) = self.U0.split()
        return Asol(pt)
    def getQatPoint(self,x):
        ''' Return Q at point x '''
        pt = fe.Point(x,0,0)
        (Asol,Qsol) = self.U0.split()
        return Qsol(pt)


    def getSol(self,func):
        (Asol,Qsol) = self.U0.split()
        if func == "A":
            return Asol
        elif func == "Q":
            return Qsol
        else:
            raise ValueError('input can either be "Q" or "A"')
 
    def plotSol(self,func):
        (Asol,Qsol) = self.U0.split()
        if func == "A":
            fe.plot(Asol)
        elif func == "Q":
            fe.plot(Qsol)
        else:
            raise ValueError('input can either be "Q" or "A"')
        
        
        





