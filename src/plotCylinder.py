import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
plt.rcParams.update({'font.size': 12})

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
nz = 100
L = 5
z = np.linspace(0,L,nz)
theta = np.linspace(0,2*np.pi,nz)
# theta = np.hstack(  (theta[0:nz],theta[0] ))
Z,THETA = np.meshgrid(z,theta)
r0 = 0*Z + 0.5
X = r0*np.cos(THETA)
Y = r0*np.sin(THETA)
mycol = cm.jet(THETA/(2*np.pi))
surf = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, facecolors=mycol)
fig.colorbar(surf)
plt.show()

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
# my_col = cm.jet(np.random.rand(Z.shape[0],Z.shape[1]))
# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors = my_col,
#         linewidth=0, antialiased=False)
# ax.set_zlim(-1.01, 1.01)
# plt.show()

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# u = np.linspace(0, 2 * np.pi, 100)
# v = np.linspace(0, np.pi, 100)
# x = 10 * np.outer(np.cos(u), np.sin(v))
# y = 10 * np.outer(np.sin(u), np.sin(v))
# z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
# z1 = z * np.cos(0.5*x)
# N = z1 / z1.max()  # normalize 0..1
# surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.jet(N), linewidth=0, antialiased=False, shade=False)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()

