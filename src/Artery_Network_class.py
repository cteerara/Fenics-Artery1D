import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
from Artery_class import *
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
A = Artery_Network(inputFile)
# print(A.numVessels, A.connectivity, A.vesselIDs)

