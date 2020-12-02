import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rcParams.update({'font.size': 12})

def P2A(beta, A0, A):
    P = beta/A0 * (A**(1/2) - A0**(1/2) ) 
    return P
def A2P(beta, A0, P):
    A = ( A0/beta*P + A0**(1/2) )**2
    return A
def getBeta(E, h0):
    return E*h0*np.sqrt(np.pi)

# class Beta(fe.UserExpression):
#     def __init__(self, E, h0, **kwargs):
#         self.E  = E
#         self.h0 = h0
#         super().__init__(**kwargs)
#     def eval(self, value, x):
#         E = self.E.ufl_evaluate(x,0,())
#         value[0] = E*self.h0*np.sqrt(np.pi) 
#     def value_shape(self):
#         return ()



