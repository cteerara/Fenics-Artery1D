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
        ME     = fe.MixedElement([AE,QE])
        self.W = fe.FunctionSpace(self.mesh,ME)
        (self.vA    , self.vQ)     = fe.TestFunction(self.W)
        (self.ATrial, self.QTrial) = fe.TrialFunction(self.W)
        # -- Coordinate values
        x = fe.interpolate(fe.Expression("x[0]", degree=1), self.W.sub(0).collapse())
        self.coords = self.gatherField(x)
        # print("rank",self.rank,"coord",self.coords)
        # sys.exit()
        # -- Get beta and A
        self.beta  = fe.interpolate(beta, self.W.sub(0).collapse())
        self.A0    = fe.interpolate(A0, self.W.sub(0).collapse())
        self.dbeta = fe.grad( self.beta )[0]
        self.dA0   = fe.grad( self.A0   )[0]
        # -- Initialize solutions
        self.nextSol  = fe.Function(self.W)
        self.currentSol = fe.Function(self.W)
        fe.assign(self.currentSol.sub(0),self.A0)
        self.wf = self.getWeakform()


    # -- Get boundary values
    def getBoundaryA(self):
        (A,Q) = self.currentSol.split(deepcopy=True)
        return self.getBoundaryValues(A) 
    def getBoundaryQ(self):
        (A,Q) = self.currentSol.split(deepcopy=True)
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
        for i in range(0,len(field_gathered)):
            if pt < self.coords[i]:
                break
        f1 = field_gathered[i-1]
        f2 = field_gathered[i]
        x1 = self.coords[i-1]
        x2 = self.coords[i]
        a = (f2-f1)/(x2-x1)
        b = f1-a*x1
        return a*pt+b



    def getc1(self,beta,A0,A):
        ''' Return c1 with values beta, A0, A '''
        rho = Global.rho
        c1 = ( beta/(2*rho*A0) )**(0.5) * A**(0.25) 
        return c1

    def getAQFromChar(self, beta, A0, char_pos, char_neg):
        rho = Global.rho 
        A = (2*rho*A0/beta)**2 * ( (char_pos - char_neg)/8 )**4
        Q = A/2*(char_pos - char_neg)
        return (Q,A)

    def getCharacteristic(self,beta,A0,Q,A):
        ''' Compute characteristics at values beta, A0, Q, A postive and negative discriminants
            return ( char+ , char- )'''
        c1 = self.getc1(beta,A0,A)
        char_pos = Q/A + 4*c1
        char_neg = Q/A - 4*c1
        return (char_pos, char_neg)

    def getLambda(self,beta,A0,Q,A):
        ''' Compute eigenvalues of H at values beta, A0, Q, A postive and negative discriminants
            return ( lambda+, lambda- )'''
        alpha = Global.alpha
        c1 = self.getc1(beta,A0,A)
        lambda_pos = alpha*Q/A + ( c1**2 + alpha*(alpha-1)*Q**2/A**2    )**(0.5)
        lambda_neg = alpha*Q/A - ( c1**2 + alpha*(alpha-1)*Q**2/A**2    )**(0.5)
        return (lambda_pos, lambda_neg)

    def getBoundaryCharacteristic(self,LeftOrRight):
        ''' Compute positive or negative discriminant boundary characteristics
            return ( charLeft+  , charLeft-  ) if LeftOrRight=="left" 
            return ( charRight+ , charRight- ) if LeftOrRight=="right"
        '''
        (betaL , betaR) = self.getBoundaryValues(self.beta)
        (A0L   , A0R  ) = self.getBoundaryValues(self.A0)
        (QL    , QR   ) = self.getBoundaryQ()
        (AL    , AR   ) = self.getBoundaryA()
        if   LeftOrRight.lower() == "left":
            (char_pos, char_neg) = self.getCharacteristic(betaL, A0L, QL, AL)
        elif LeftOrRight.lower() == "right":
            (char_pos, char_neg) = self.getCharacteristic(betaR, A0R, QR, AR)
        else:
            raise ValueError("LeftOrRight must be either \"left\" or \"right\"")
        return (char_pos, char_neg)

    def getBoundaryLambda(self,LeftOrRight):
        ''' Compute Left or Right boundary characteristics
            return (lambdaLeft+ , lambdaLeft- ) if LeftOrRight=="left" 
            return (lambdaRight+, lambdaRight-) if LeftOrRight=="right"
        '''
        (betaL , betaR) = self.getBoundaryValues(self.beta)
        (A0L   , A0R  ) = self.getBoundaryValues(self.A0)
        (QL    , QR   ) = self.getBoundaryQ()
        (AL    , AR   ) = self.getBoundaryA()
        if   LeftOrRight.lower() == "left":
            (lambda_pos, lambda_neg) = self.getLambda(betaL, A0L, QL, AL)
        elif LeftOrRight.lower() == "right":
            (lambda_pos, lambda_neg) = self.getLambda(betaR, A0R, QR, AR)
        else:
            raise ValueError("LeftOrRight must be either \"left\" or \"right\"")
        return (lambda_pos, lambda_neg)


    def getCharacteristicAtPt(self, pt):
        ''' Return characteristics postive and negative at a designated point '''
        (ACurrent,QCurrent) = self.currentSol.split(deepcopy=True)
        beta = self.interpolate( self.beta    , pt )
        A0   = self.interpolate( self.A0      , pt )
        A    = self.interpolate( ACurrent     , pt )
        Q    = self.interpolate( QCurrent     , pt )
        (char_pos, char_neg) = self.getCharacteristic(beta, A0, Q, A)
        return (char_pos, char_neg)

    def getLambdaAtPt(self, pt):
        ''' Return eigenvalues of H positive and negative at a designated point '''
        (ACurrent,QCurrent) = self.currentSol.split(deepcopy=True)
        beta = self.interpolate( self.beta    , pt )
        A0   = self.interpolate( self.A0      , pt )
        A    = self.interpolate( ACurrent     , pt )
        Q    = self.interpolate( QCurrent     , pt )
        (lambda_pos, lambda_neg) = self.getLambdaAtPt(beta, A0, Q, A)
        return (lambda_pos, lambda_neg)


    def getWeakform(self):
        # -- Weakform data 
        vA = self.vA
        vQ = self.vQ
        dvA = fe.grad(vA)[0]
        dvQ = fe.grad(vQ)[0]
        dt = Global.dt
        (A,Q) = fe.split(self.currentSol)
        # -- Define weakform term
        F   = self.getF   (Q,A)
        B   = self.getB   (Q,A)
        H   = self.getH   (Q,A)
        FLW = self.getFLW (Q,A)
        BLW = self.getBLW (Q,A)
        BU  = self.getBU  (Q,A)
        dFdz = [fe.grad(F[0])[0], fe.grad(F[1])[0]]
        BU_dFdz = self.matMult(BU,dFdz)
        H_dFdz = self.matMult(H,dFdz)
        # -- Trial term
        wf  = -(self.QTrial*vQ + self.ATrial*vA)
        # -- A term
        wf += A*vA
        wf += Q*vQ

        wf +=  dt*FLW[0]*dvA
        wf +=  dt*FLW[1]*dvQ

        wf += -dt**2/2 * BU_dFdz[0]*vA
        wf += -dt**2/2 * BU_dFdz[1]*vQ

        wf += -dt**2/2 * H_dFdz[0]*dvA
        wf += -dt**2/2 * H_dFdz[1]*dvQ 

        wf +=  dt * BLW[0]*vA 
        wf +=  dt * BLW[1]*vQ

        wf *=  fe.dx
        return wf


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
        B = [fe.Constant(0),KR*Q/A + A/(A0*rho)*(2/3*A**(1/2)-A0**(1/2))*dbeta - beta/rho*A/A0**2*(2/3*A**(1/2)-1/2*A0**(1/2))*dA0]
        return B

    def getH(self,Q,A):
        alpha = Global.alpha
        rho = Global.rho
        beta = self.beta
        A0 = self.A0
        H = [ [fe.Constant(0), fe.Constant(1)] , [-alpha*Q**2/A**2 + beta/(2*rho*A0)*A**(1/2), 2*alpha*Q/A]  ]
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
        BU1 += (  A**(1/2)/(rho*A0**2) * ( -beta*dA0 + A0*dbeta   )   )
        BU2 = KR/A
        BU = [ [fe.Constant(0), fe.Constant(0)] , [BU1, BU2] ]
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
        self.currentSol.assign(self.nextSol)

    def plotMesh(self):
        fe.plot(self.mesh)
        plt.show()
        

