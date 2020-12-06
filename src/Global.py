global rho, alpha, KR
import numpy as np
pi = np.pi

# -- Physical parameters
rho = 1 # g/cm^3
alpha = 1
nu = 0.035
KR = 8*nu*pi

# -- Time parameters
T = 2*0.165
nt = 1e3
time = np.linspace(0,T/2+(0.25-0.165),int(nt))
# dt = 0.001
dt = time[1]-time[0]
# print(dt)

def A2P(beta, A0, A):
    P = beta/A0 * (A**(1/2) - A0**(1/2) ) 
    return P
def P2A(beta, A0, P):
    A = ( A0/beta*P + A0**(1/2) )**2
    return A
