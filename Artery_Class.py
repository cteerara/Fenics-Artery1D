from Constants import *
import matplotlib.pyplot as plt
import fenics as fe

class Artery:
    '''
    This class contains the artery's properties 
    '''
    def __init__(self, xStart, xEnd, nx, beta, dbeta, A0, dA0 ):
        
        # -- Constant data
        self.beta = beta
        self.dbeta = dbeta
        self.A0 = A0
        self.dA0 = dA0
        self.xStart = xStart
        self.xEnd = xEnd
        self.mesh = fe.IntervalMesh(nx,xStart,xEnd)

        # -- Finite Element Space Data
        QE = fe.FiniteElement("Lagrange", cell=self.mesh.ufl_cell(), degree=1)
        AE = fe.FiniteElement("Lagrange", cell=self.mesh.ufl_cell(), degree=1)
        ME = fe.MixedElement([QE,AE])
        self.W = fe.FunctionSpace(self.mesh,ME)
        (self.w,self.q) = fe.TestFunction(self.W)
        (self.Qn,self.An) = fe.Function(self.W)
        (self.Q0,self.A0) = fe.Function(self.W)

        # -- Governing equation data



    def getF(self,Q,A):
        return [Q, alpha*Q**2/A + self.beta/(3*rho*self.A0)*A**(3/2)]

    def plotMesh(self):
        fe.plot(self.mesh)
        plt.show()
        

