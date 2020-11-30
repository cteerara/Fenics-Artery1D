import Global
import matplotlib.pyplot as plt
import fenics as fe

class Artery:
    '''
    This class contains the artery's properties 
    '''
    def __init__(self, xStart, xEnd, nx, beta, A0, degQ=1, degA=1):
        
        # -- Constant data
        self.xStart = xStart
        self.xEnd   = xEnd
        self.mesh   = fe.IntervalMesh(nx,xStart,xEnd)

        # -- Finite Element Space Data
        QE     = fe.FiniteElement("Lagrange", cell=self.mesh.ufl_cell(), degree=degQ)
        AE     = fe.FiniteElement("Lagrange", cell=self.mesh.ufl_cell(), degree=degA)
        ME     = fe.MixedElement([QE,AE])
        self.W = fe.FunctionSpace(self.mesh,ME)
        (self.w,self.q)            = fe.TestFunction(self.W)
        (self.QTrial, self.ATrial) = fe.TrialFunction(self.W)

        # -- Get beta and A
        self.beta  = fe.interpolate(beta, self.W.sub(1).collapse())
        self.A0    = fe.interpolate(A0, self.W.sub(1).collapse())
        self.dbeta = fe.grad( self.beta )[0]
        self.dA0   = fe.grad( self.A0   )[0]

        # -- Initialize solutions
        self.currentSol  = fe.Function(self.W)
        self.previousSol = fe.Function(self.W)
        fe.assign(self.previousSol.sub(1),self.A0)
        self.wf = self.getWeakform()


    def getWeakform(self):

        # -- Weakform data 
        w = self.w
        q = self.q
        dw = fe.grad(w)[0]
        dq = fe.grad(q)[0]
        dt = Global.dt
        (Q0,A0) = fe.split(self.previousSol)
        (Qn,An) = fe.split(self.currentSol)

        # -- Define weakform term
        F = self.getF(Q0,A0)
        B = self.getB(Q0,A0)
        H = self.getH(Q0,A0)
        FLW = self.getFLW(Q0,A0)
        BLW = self.getBLW(Q0,A0)
        BU = self.getBU(Q0,A0)
        dFdz = [fe.grad(F[0])[0], fe.grad(F[1])[0]]
        BU_dFdz = self.matMult(BU,dFdz)
        H_dFdz = self.matMult(H,dFdz)

        # -- Trial term
        wf  = -(self.QTrial*w + self.ATrial*q)
        # -- Q term
        wf += Q0*w
        wf += dt*FLW[0]*dw
        wf += -dt**2/2 * BU_dFdz[0]*w
        wf += -dt**2/2 * H_dFdz[0]*dw
        wf += dt*BLW[0]*w
        # -- A term
        wf += A0*q
        wf += dt*FLW[1]*dq
        wf += -dt**2/2 * BU_dFdz[1]*q
        wf += -dt**2/2 * H_dFdz[1]*dq
        wf += dt*BLW[1]*q
        wf *= fe.dx
        self.wf = wf


    def getF(self,Q,A):
        alpha = Global.alpha
        rho = Global.rho
        beta = self.beta
        A0 = self.A0
        F = [Q, alpha*Q**2/A + beta/(3*rho*A0)*A**(3/2)]
        return F

    def getB(self,Q,A):
        KR = Global.KR
        rho = Global.rho
        A0 = self.A0
        beta = self.beta
        dbeta = self.dbeta
        dA0 = self.dA0
        B = [0,KR*Q/A + A/(A0*rho)*(2/3*A**(1/2)-A0**(1/2))*dbeta - beta/rho*A/A0**2*(2/3*A**(1/2)-1/2*A0**(1/2))*dA0]
        return B

    def getH(self,Q,A):
        alpha = Global.alpha
        rho = Global.rho
        beta = self.beta
        A0 = self.A0
        H = [ [0,1] , [-alpha*Q**2/A**2 + beta/(2*rho*A0)*A**(1/2), 2*alpha*Q/A]  ]
        return H

    def getBU(self,Q,A):
        KR = Global.KR
        rho = Global.rho
        A0 = self.A0
        beta = self.beta
        dbeta = self.dbeta
        dA0 = self.dA0
        BU1  = -KR*Q/A**2 
        BU1 += (beta*dA0 - 2*A0*dbeta)/(2*rho*A0**(3/2)) 
        BU1 += (A**(1/2)/rho*A0**2) * (-beta*dA0 + A0*dbeta)
        BU2 = KR/A
        BU = [ [0,0] , [BU1, BU2] ]
        return BU

    def getFLW(self, Q, A):
        dt = Global.dt
        F = self.getF(Q,A)
        H = self.getH(Q,A)
        B = self.getB(Q,A)
        HB = self.matMult(H,B)
        FLW1 = F[0] + dt/2*HB[0]
        FLW2 = F[1] + dt/2*HB[1]
        FLW = [FLW1, FLW2]
        return FLW

    def getBLW(self,Q,A):
        dt = Global.dt
        B = self.getB(Q,A)
        BU = self.getBU(Q,A)
        BUB = self.matMult(BU,B)
        BLW1 = B[0] + dt/2*BUB[0] 
        BLW2 = B[1] + dt/2*BUB[1]
        BLW = [BLW1, BLW2]
        return BLW


    # -- Utility functions
    def matMult(self,A,x):
        # -- Multiply 2-by-2 list with 2-by-1 list
        # -- return [A]{x}
        return [ A[0][0]*x[0] + A[0][1]*x[1] , A[1][0]*x[0] + A[1][1]*x[1] ] 


    def updateSol(self):
        self.previousSol.assign(self.currentSol)


    

    def plotMesh(self):
        fe.plot(self.mesh)
        plt.show()
        

