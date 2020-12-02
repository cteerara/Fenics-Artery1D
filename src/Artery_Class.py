import Global
import matplotlib.pyplot as plt
import fenics as fe
from utils import *
import myMPI

class Artery:
    '''
    This class contains the artery's properties 
    '''
    def __init__(self, xStart, xEnd, nx, beta, A0, degQ=1, degA=1):
        
        # -- MPI variables
        self.comm = fe.MPI.comm_world
        self.rank = fe.MPI.rank(self.comm)
        self.nPE  = fe.MPI.size(self.comm)
        
        # -- Constant data
        self.xStart  = xStart
        self.xEnd    = xEnd
        self.numElem = nx 
        self.mesh    = fe.IntervalMesh(nx,xStart,xEnd)
        # -- Finite Element Space Data
        QE     = fe.FiniteElement("Lagrange", cell=self.mesh.ufl_cell(), degree=degQ)
        AE     = fe.FiniteElement("Lagrange", cell=self.mesh.ufl_cell(), degree=degA)
        ME     = fe.MixedElement([QE,AE])
        self.W = fe.FunctionSpace(self.mesh,ME)
        (self.w,self.q)            = fe.TestFunction(self.W)
        (self.QTrial, self.ATrial) = fe.TrialFunction(self.W)
        # -- Coordinate values
        x = fe.interpolate(fe.Expression("x[0]", degree=1), self.W.sub(0).collapse())
        self.coords = self.gatherField(x)
        print("rank",self.rank,"coord",self.coords)
        # sys.exit()
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

        x2 = fe.interpolate(fe.Expression("x[0]*x[0]", degree=1), self.W.sub(0).collapse())
        print(self.interpolate(x2,10))



        # self.getBoundaryChar_Pos()
        # self.getBoundaryChar_Neg()

    # -- Get boundary values
    def getBoundaryA(self):
        (Q,A) = self.previousSol.split(deepcopy=True)
        return self.getBoundaryValues(A) 
    def getBoundaryQ(self):
        (Q,A) = self.previousSol.split(deepcopy=True)
        return self.getBoundaryValues(Q) 

    def getBoundaryValues(self, field):
        ''' Return Left and Right boundary values of the field '''
        fieldArr = field.compute_vertex_values()
        fieldL = 0
        fieldR = 0
        FGL = np.zeros(self.nPE)
        FGR = np.zeros(self.nPE)
        if self.rank == 0:
            fieldL = fieldArr[0]
        if self.rank == self.nPE-1:
            fieldR = fieldArr[-1]
        self.comm.Allgather( np.array([fieldL],dtype=np.float64), FGL  )
        self.comm.Allgather( np.array([fieldR],dtype=np.float64), FGR  )
        return [ FGL[0], FGR[-1] ]


    # -- Get Coordinate values
    def gatherField(self,field):
        field_vv = field.compute_vertex_values()
        if self.rank == self.nPE-1:
            field_send = field_vv[0:]
        else:
            field_send = field_vv[0:len(field_vv)-1]
        return myMPI.Allgatherv(field_send,self.comm)

    # -- Interpolate
    def interpolate(self,field,pt):
        ''' Performs linear interpolation of a value at point pt on field '''
        if pt < self.coords[0] or pt > self.coords[-1]:
            raise Exception('Point out of bound.')
        field_gathered = self.gatherField(field) 
        print("rank:",self.rank,"field:",field_gathered)
        for i in range(0,len(field_gathered)):
            if pt < self.coords[i]:
                break
        f1 = field_gathered[i-1]
        f2 = field_gathered[i]
        x1 = self.coords[i-1]
        x2 = self.coords[i]
        print(self.rank,"f",f1,f2)
        print(self.rank,"x",x1,x2)
        print(self.rank,"pt",pt)
        print(self.rank,"i",i)
        a = (f2-f1)/(x2-x1)
        b = f1-a*x1
        return a*pt+b




    # -- Get characteristics
    def getBoundaryChar_Pos(self):
        ''' Get the positive drisciminant of the characteristic solution '''
        (Q,A) = self.previousSol.split(deepcopy=True)
        alpha = Global.alpha
        rho = Global.rho
        (betaL , betaR) = self.getBoundaryValues(self.beta)
        (A0L   , A0R  ) = self.getBoundaryValues(self.A0)
        (QL    , QR   ) = self.getBoundaryQ()
        (AL    , AR   ) = self.getBoundaryA()
        charL_Pos = alpha*QL/AL + 4*AL**(1/4)*(betaL/(2*rho*A0L))**(1/2)
        charR_Pos = alpha*QR/AR + 4*AR**(1/4)*(betaR/(2*rho*A0R))**(1/2)
        return [charL_Pos, charR_Pos]

    def getBoundaryChar_Neg(self):
        ''' Get the positive drisciminant of the characteristic solution '''
        (Q,A) = self.previousSol.split(deepcopy=True)
        alpha = Global.alpha
        rho = Global.rho
        (betaL , betaR) = self.getBoundaryValues(self.beta)
        (A0L   , A0R  ) = self.getBoundaryValues(self.A0)
        (QL    , QR   ) = self.getBoundaryQ()
        (AL    , AR   ) = self.getBoundaryA()
        charL_Neg = alpha*QL/AL - 4*AL**(1/4)*(betaL/(2*rho*A0L))**(1/2)
        charR_Neg = alpha*QR/AR - 4*AR**(1/4)*(betaR/(2*rho*A0R))**(1/2)
        return [charL_Neg, charR_Neg]


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
        

