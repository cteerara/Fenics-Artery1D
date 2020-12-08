import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rcParams.update({'font.size': 12})

n = 2
# First element is the parent's vessel value
# The rest are the daughter vessels
alpha = 1
rho = 1
beta = np.zeros(n+1)+0.5
A0 = np.zeros(n+1)+2

gamma = beta/A0
sigma = 4*np.sqrt(beta/(2*rho*A0))
W = np.zeros(n+1)+3

# K = np.zeros( (n+1,n+1)  )
# R = np.zeros(n+1)
# k1 = np.zeros( (n,n) )
# k2 = np.zeros( (n,n) )
# k3 = np.zeros( (n,n) )

def getK(A,Q):
    n = len(A) 
    n = n-1 # To make this consistent with the notes
    K = np.zeros( (2*n+2,2*n+2) )
    K[0,0] = 1
    K[n+1,0] = alpha/A[0]
    K[n+1,n+1] = -alpha*Q[0]/A[0]**2 + sigma[0]/(4*A[0]**(4./3))
    k1 = np.zeros((n,n))
    k2 = np.zeros((n,n))
    k3 = np.zeros((n,n))
    for i in range(1,n+1):
        K[0,i] = -1
        K[i,n+1] = gamma[0]/(2*np.sqrt(A[0]))
        k1[i-1,i-1] = -gamma[i]/(2*np.sqrt(A[i]))
        k2[i-1,i-1] = alpha/A[i]
        k3[i-1,i-1] = -alpha*Q[i]/A[i]**2 - sigma[i]/(4*A[i]**(4./3))
    K[1:n+1,n+2:2*n+2] = k1
    K[n+2:2*n+2, n+2:2*n+2] = k3
    K[n+2:2*n+2, 1:n+1 ] = k2
    return K
    
def getR(A,Q,W):
    n = len(A)
    n = n-1
    R1 = Q[0]
    R = np.zeros(2*n+2)
    for i in range(1,n+1):
        R1 -= Q[i]
    R[0] = R1
    R[n+1] = alpha*Q[0]/A[0] + sigma[0]*A[0]**(1./4) - W[0]
    for i in range(1,n+1):
        R[i] = gamma[0]*(np.sqrt(A[0]) - np.sqrt(A0[0])) - gamma[i]*( np.sqrt(A[i]) - np.sqrt(A0[i])  )
        j = i + n + 1
        R[j] = alpha*Q[i]/A[i] - sigma[i]*A[i]**(1./4) - W[i]
    return R


A = np.zeros(n+1)+2
Q = np.zeros(n+1)
itmax = 2
for i in range(0,itmax):
    n = len(A)
    n = n-1
    K = getK(A,Q)
    # print(K)
    R = getR(A,Q,W)
    # print(R)
    # residue = np.linalg.norm(R)
    # print(residue)
    du = np.linalg.solve(K,-R)
    Q += du[0:n+1]
    A += du[n+1:2*n+2]
    print(Q)
    print(A)


# print(R)

# K = getK(A,Q)
# print(K)
# print(getR(A,Q,W))



