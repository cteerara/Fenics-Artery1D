global rho, alpha, KR, theta 
import numpy as np
pi = np.pi

# -- Physical parameters
rho = 1 # g/cm^3
alpha = 1
nu = 0.035 
KR = 8*nu*pi


# def A2P(beta, A0, A):
#     P = beta/A0 * (A**(1/2) - A0**(1/2) ) 
#     return P
# def P2A(beta, A0, P):
#     A = ( A0/beta*P + A0**(1/2) )**2
#     return A
