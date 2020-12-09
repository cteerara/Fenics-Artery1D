import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rcParams.update({'font.size': 12})
from Global import *
from Artery_class import *



fe.set_log_level(40)
# -- Constants
L = 15
ne = 2**7
r0 = 0.5
E = 3e6
h0 = 0.05
Q0 = 0
A = Artery(L,ne,r0,Q0,E,h0,degA=1,degQ=1)
A.solve()

