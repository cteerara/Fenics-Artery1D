import numpy as np
import sys
from mpi4py import MPI
import fenics as fe

def Allgatherv(a,comm):
    rank = fe.MPI.rank(comm)
    nPE  = fe.MPI.size(comm)
    sizes = np.zeros(nPE)
    offsets = np.zeros(nPE)
    comm.Allgather( np.array([len(a)], dtype=np.float64 ), sizes )
    sizes.astype(int)
    offsets[1:] = np.cumsum(sizes)[0:nPE-1]
    offsets.astype(int)
    a_gathered = np.zeros( int(np.sum(sizes)) )
    print(sizes)
    print(offsets)
    comm.Allgatherv( a, [a_gathered, sizes, offsets, MPI.DOUBLE]  )
    return a_gathered



