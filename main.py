import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
from Artery_Class import *
import Global
plt.rcParams.update({'font.size': 12})

xStart = 0
xEnd = 1
L = xEnd - xStart
nx = 100
beta = fe.Expression("1",degree=1)
A0 = fe.Expression("1",degree=1)
# A0 = 1
Arty1 = Artery(xStart, xEnd, nx, beta, A0)


# sol0 = Arty1.previousSol
# (Q0,A0) = fe.split(sol0)
# fe.plot(A0)
# plt.show()



# # -- Test 
# x = np.linspace(xStart, xEnd, nx)
# fe.plot(Arty1.A0)
# # fe.plot(Arty1.dA0)
# plt.show()



