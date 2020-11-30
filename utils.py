import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rcParams.update({'font.size': 12})

def P2A(beta, A0, A):
    P = beta/A0 * (A**(1/2) - A0**(1/2) ) 
    return P

def A2P(beta, A0, P):
    A = A0/beta * P + A0**(1/2)
    A = A**2
    return A

def getBeta(E, h0):
    return E*h0*np.sqrt(np.pi)


