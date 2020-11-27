import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
from Artery_Class import *
from Constants import *
plt.rcParams.update({'font.size': 12})

xStart = 0
xEnd = 1
nx = 10
beta = 1
dbeta = 0 
A0 = 1
dA0 = 0
Arty1 = Artery(xStart, xEnd, nx, beta, dbeta, A0, dA0)
x = Arty1.getF()
print(x)
Arty1.plotMesh()
