import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
from Artery_Class import *
import Global
plt.rcParams.update({'font.size': 12})

xStart = 0
xEnd = 1
nx = 10
beta = 1
dbeta = 0 
A0 = 1
dA0 = 0
Arty1 = Artery(xStart, xEnd, nx, beta, dbeta, A0, dA0)
Arty1.plotMesh()
