import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
from Artery_class import *
from Global import *
from utils import *
plt.rcParams.update({'font.size': 12})

fe.set_log_level(40)

def str2lst(strs):
    ''' Convert a string of comma delimited numbers into a list of those numbers  '''
    lst = []
    for s in strs.strip().split(","):
        try:
            lst.append(int(s))
        except:
            lst.append(float(s))
    return lst

# Artyp = Artery(L, ne, r0, Q0,   E, h0, theta, dt, degA=degA,degQ=degQ)
class Artery_Network:
    ''' Class defines the artery network and its connectivity '''
    def __init__(self, inputFile, dt, theta):
        self.input = inputFile
        (self.numVessels, self.connectivity, self.vesselIDs) = self.getConnectivity()
        self.paramDict = self.getParameters()
        self.Arteries = []
        for i in range(0,self.numVessels):
            L    = self.paramDict["L"][i]
            ne   = self.paramDict["ne"][i]
            r0   = self.paramDict["r0"][i]
            Q0   = self.paramDict["Q0"][i]
            E    = self.paramDict["E"][i]
            h0   = self.paramDict["h0"][i]
            degA = self.paramDict["degA"][i]
            degQ = self.paramDict["degQ"][i]
            Arty = Artery(L,ne,r0,Q0,E,h0,theta,dt,degA=degA,degQ=degQ)
            self.Arteries.append(Arty)
            


    
    def getKR(self,A,Q,gamma,sigma,parents,daughters):
        ''' A and Q are vectors containing boundary values of A and Q to be solved in the compatibility condition  '''
        parentID = parents
        n = len(A)-1
        K = np.zeros( (2*n+2,2*n+2) )
        Ap = A[0] ; Qp = Q[0]
        gammap = gamma[0]  ; sigmap = sigma[0]
        K[0,0] = 1
        K[0,1:n+1] = -1
        K[1:n+1,0] = Qp/Ap**2
        K[n+1,0] = alpha/Ap
        K[1:n+1,n+1] = -Qp**2/Ap**3 + gammap/(2*np.sqrt(Ap)) 
        K[n+1,n+1] = -alpha*Qp/Ap**2 + sigmap/(4*Ap**(3/4))
        Ki11 = []
        Ki12 = []
        Ki21 = []
        Ki22 = []
        for i in range(1,n+1):
            Ki11.append( -Q[i]/A[i]**2  )
            Ki12.append( Q[i]**2/A[i]**3 - gamma[i]/(2*np.sqrt(A[i])) )
            Ki21.append( alpha/A[i] )
            Ki22.append( -alpha*Q[i]/A[i]**2 - sigma[i]/(4*A[i]**(3./4)) )
        Ki11 = np.diag(Ki11)
        Ki12 = np.diag(Ki12)
        Ki21 = np.diag(Ki21)
        Ki22 = np.diag(Ki22)
        K[ 1:(n+1) , 1:(n+1) ]             = Ki11
        K[ 1:(n+1) , (n+2):(2*n+2) ]       = Ki12
        K[ (n+2):(2*n+2) , 1:(n+1) ]       = Ki21
        K[ (n+2):(2*n+2) , (n+2):(2*n+2) ] = Ki22

        R = np.zeros( 2*n+2 )
        (A_p_0, Q_p_0) = self.Arteries[parentID].getBoundaryAQ("right")
        (lam1_p, lam2_p) = self.Arteries[parentID].getEigenvalues( A_p_0, Q_p_0 )
        (W1_p, W2_p) = self.Arteries[parentID].getCharacteristics( self.Arteries[parentID].getAatPoint( self.Arteries[parentID].L - lam1_p*self.Arteries[parentID].dt ) , \
                                                            self.Arteries[parentID].getQatPoint( self.Arteries[parentID].L - lam1_p*self.Arteries[parentID].dt )  )
        ptp = gammap*( np.sqrt(Ap) - np.sqrt(self.Arteries[parentID].A0) ) + 1/2*(Qp/Ap)**2
        R[0] = Qp
        for i in range(1,n+1):
            d = daughters[i-1]
            R[0] -= Q[i]
            pti = gamma[i]*( np.sqrt(A[i]) - np.sqrt(self.Arteries[d].A0) ) + 1/2*(Q[i]/A[i])**2
            R[i] = ptp - pti

        
        R[n+1] = alpha*Qp/Ap + sigmap*Ap**(1./4) - W1_p
        for i in range(1,n+1):
            d = daughters[i-1]
            (A_p_i, Q_p_i) = self.Arteries[d].getBoundaryAQ("left")
            (lam1_i, lam2_i) = self.Arteries[d].getEigenvalues( A_p_i, Q_p_i )
            (W1_i, W2_i) = self.Arteries[d].getCharacteristics( self.Arteries[d].getAatPoint( -lam2_i*self.Arteries[d].dt ) , \
                                                                self.Arteries[d].getQatPoint( -lam2_i*self.Arteries[d].dt ) )
            j = i+n+1
            R[j] = alpha*Q[i]/A[i] - sigma[i]*A[i]**(1./4) - W2_i
        return (K,R)

    def getParameters(self):
        fid = open(self.input)
        L  = np.zeros(self.numVessels)
        ne = np.zeros(self.numVessels)
        r0 = np.zeros(self.numVessels)
        Q0 = np.zeros(self.numVessels)
        E  = np.zeros(self.numVessels)
        h0 = np.zeros(self.numVessels)
        paramDict = {}
        for line in fid.readlines():
            if not line.strip().startswith("#") and not line.strip()=='' and line.strip().split(":")[0].strip() == 'Value':
                ll = line.strip().split(":")[1].split("=")
                paramDict[ll[0].strip()] = str2lst(ll[1].strip())
        fid.close()
        return paramDict

    def getConnectivity(self):
        fid = open(self.input)
        # -- Create a list of parent and daughter vessels
        parentList = []
        daughterList = []
        vesselIDs = [] # Overall vessel IDs used to check if the IDs are consecutive
        for line in fid.readlines():
            if not line.strip().startswith("#") and not line.strip()=='' and line.strip().split(":")[0].strip() == 'Connectivity':
                # parent,daughter = line.strip().split(":")[1].split(":")
                parent = line.strip().split(":")[1]
                daughter = line.strip().split(":")[2]
                p = int(parent)
                if p not in vesselIDs:
                    vesselIDs.append(p)
                dlist = str2lst(daughter)
                for d in dlist:
                    if d < p:
                        raise ValueError("Daughter of parent vessel ID %d is smaller than daughter ID %d" % (p,d) )
                    if d not in vesselIDs:
                        vesselIDs.append(d)
                parentList.append(p)
                daughterList.append(dlist)
        vesselIDs = (np.sort(np.array(vesselIDs)))
        # -- Check if 0 exist
        if vesselIDs[0] != 0:
            raise ValueError("Parent vessel 0 does not exist")
        numVessels = len(vesselIDs)
        # -- Check if list is consecutive
        for i in range(1,numVessels):
            if vesselIDs[i] - vesselIDs[i-1] != 1:
                raise ValueError("Vessel IDs are not consecutive. Vessel IDs are:",list(vesselIDs))
        # -- Create connectivity list
        connectivity = []
        # Create a list of empty list for each vessel. 
        for i in range(0,numVessels):
            connectivity.append([])
        count = 0
        for p in parentList:
            connectivity[p] = daughterList[count]
            count += 1
        fid.close()
        return (numVessels, connectivity, vesselIDs)

    def getBoundaryConditions(self,inputArea):
        # Boundary conditions for each vessel
        # The source vessel (ID=0) has inputArea as the input BC and no Qin BC
        # which is why we use ``None`` here.
        Ain=[];Qin=[];Aout=[];Qout=[]
        for i in range(0,self.numVessels):
            Ain.append(None) 
            Qin.append(None)
            Aout.append(None)
            Qout.append(None)
        Ain[0] = inputArea

        # Get boundary contitions at the interface between vessels
        for p in range(0,self.numVessels):
            # p is the parent vessel
            # ds are the daughter vessels
            ds = self.connectivity[p]
            if ds != []: # If daughter exist
                # Get parent A,Q at the outlet
                A = [] ; Q = [] ; gamma = [] ; sigma = []
                (Ap,Qp) = self.Arteries[p].getBoundaryAQ("right")
                A.append(Ap) ; Q.append(Qp)
                gamma.append( self.Arteries[p].beta / self.Arteries[p].A0 )
                sigma.append( 4 * np.sqrt( self.Arteries[p].beta / (2*rho*self.Arteries[p].A0)  )  )
                # Assign initial guess of A and Q and constants gamma and sigma
                for d in ds:
                    # Get inlet boundary values for the daughter vessels
                    (Ai,Qi) = self.Arteries[d].getBoundaryAQ("left")
                    A.append(Ai) ; Q.append(Qi)
                    gamma.append( self.Arteries[d].beta / self.Arteries[d].A0 )
                    sigma.append( 4 * np.sqrt( self.Arteries[d].beta / (2*rho*self.Arteries[d].A0)  )  )
                NR_itmax = 10000
                tol = 1e-5
                # Compute the boundary values of the parent and daughter vessels
                for NR_it in range(0,NR_itmax):
                    (K,R) = self.getKR(A,Q,gamma,sigma,p,ds)
                    # print("Parent vessel: %d. NR iteration: %d. Residue = %f" %(p, NR_it, np.linalg.norm(R)) )
                    if np.linalg.norm(R) < tol:
                        break
                    dU = np.linalg.solve(K,-R)
                    nn = int(len(dU)/2)
                    for i in range(0,nn):
                        Q[i] += dU[i]
                        A[i] += dU[i+nn]
                # Assign BC to the final BC list
                Aout[p] = A[0]
                Qout[p] = Q[0]
                i = 1
                for d in ds:
                    Ain[d] = A[i]
                    Qin[d] = Q[i]
                    i += 1

            else: # No daughters, compute no reflection BCs
                (ANoReflect, QNoReflect) = self.Arteries[p].getNoReflectionBC() 
                Aout[p] = ANoReflect
                Qout[p] = QNoReflect

        return (Ain,Qin,Aout,Qout)

    def solve(self,AinBC, AoutBC, QinBC, QoutBC):
        for i in range(0,self.numVessels):
            self.Arteries[i].solve( Ain=AinBC[i] , Aout=AoutBC[i] , Qin=QinBC[i] , Qout=QoutBC[i] )

        







