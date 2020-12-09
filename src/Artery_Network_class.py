import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
from Artery_class import *
from Global import *
plt.rcParams.update({'font.size': 12})

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
            


    
    def getNRSystem(self,A,Q,gamma,sigma):
        ''' A and Q are vectors containing boundary values of A and Q to be solved in the compatibility condition  '''
        print("Avec",A)
        print("Qvec",Q)
        print("GammeVec",gamma)
        print("SigmaVec",sigma)
        print()

        n = len(A)-1
        K = np.zeros( (2*n+2,2*n+2) )
        R = np.zeros( 2*n+2 )
        Ap = A[0] ; Qp = Q[0]
        gammap = gamma[0]  ; sigmap = sigma[0]
        K[0,0] = 1
        K[0,1:n+1] = -1
        # K[1:n+1,0] = Qp/Ap**2
        K[n+1,0] = alpha/Ap
        print(alpha/Ap)
        # K[1:n+1,n+1] = -Qp**2/Ap**3 + gammap/(2*np.sqrt(Ap)) 
        # K[n+1,n+1] = -alpha*Qp/Ap**2 + sigmap/(2*Ap**(3/4))
        # Ki11 = []
        # Ki12 = []
        # Ki21 = []
        # Ki22 = []
        # for i in range(1,n+1):
        #     Ki11.append( -Q[i]/A[i]**2  )
        #     Ki12.append( Q[i]**2/A[i]**3 - gamma[i]/(2*np.sqrt(A[i])) )
        #     Ki21.append( alpha/A[i] )
        #     Ki22.append( -alpha*Q[i]/A[i]**2 - sigma[i]/(4*A[i]**(3./4)) )
        # K[1:n+1,1:n+1] = Ki11
        # K[1:n+1,n+2:2*n+2] = Ki12
        # K[n+2:2*n+2,1:n+1] = Ki21
        # K[n+2:2*n+2,1:n+1] = Ki22
        return K






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


inputFile = 'testInput.in'
inputFile = 'TwoSection.in'
A = Artery_Network(inputFile,1,1)
(A_p, Q_p) = A.Arteries[0].getBoundaryAQ("right")
(A_1, Q_1) = A.Arteries[1].getBoundaryAQ("left") 
(lam1_p, lam2_p) = A.Arteries[0].getEigenvalues(A_p,Q_p)
(lam1_1, lam2_1) = A.Arteries[1].getEigenvalues(A_1,Q_1)
gamma_p = A.Arteries[0].beta / A.Arteries[0].A0
gamma_1 = A.Arteries[1].beta / A.Arteries[1].A0
sigma_p = 4 * np.sqrt(A.Arteries[0].beta/(2*rho*A.Arteries[0].A0))
sigma_1 = 4 * np.sqrt(A.Arteries[1].beta/(2*rho*A.Arteries[1].A0))


K = A.getNRSystem([A_p,A_1], [Q_p,Q_1], [gamma_p,gamma_1], [sigma_p,sigma_1])
print(K)
# print(A.numVessels, A.connectivity, A.vesselIDs)

