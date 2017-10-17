import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl
import scipy as sp
import numpy as np
from msvcrt import getch
import time
from Prak1_z2_triangles import isIntersecting

a1 = []
a2 = []
b1 = []
b2 = []
c1 = []
c2 = []

# 1
a1.append(np.array([0, 0, 0]))
b1.append(np.array([0, 5, 0]))
c1.append(np.array([6, 5, 0]))

a2.append(np.array([1, 4, 0]))
b2.append(np.array([2, 4, 0]))
c2.append(np.array([2, 3, 0]))

# 2
a1.append(np.array([-1, 0, 0]))
b1.append(np.array([0, -1, 0]))
c1.append(np.array([0, 0, 0]))

a2.append(np.array([0, 0, 0]))
b2.append(np.array([0, 3, 0]))
c2.append(np.array([5, 0, 0]))

# 3
a1.append(np.array([-1, 0, 0]))
b1.append(np.array([0, 2, 0]))
c1.append(np.array([0, 0, 0]))

a2.append(np.array([0, 0, 0]))
b2.append(np.array([5, 0, 0]))
c2.append(np.array([0, 4, 0]))

# 4
a1.append(np.array([0, 0, 0]))
b1.append(np.array([0, 2, 0]))
c1.append(np.array([1, 0, 0]))

a2.append(np.array([0, -1, 0]))
b2.append(np.array([0, 3, 0]))
c2.append(np.array([7, -1, 0]))

# 5
a1.append(np.array([0, 0, 0]))
b1.append(np.array([0, 4, 0]))
c1.append(np.array([4, 0, 0]))

a2.append(np.array([1, 2, 0]))
b2.append(np.array([1, 1, -3]))
c2.append(np.array([0.5, 2, -2]))

# 6
a1.append(np.array([0, 0, 0]))
b1.append(np.array([0, 4, 0]))
c1.append(np.array([4, 0, 0]))

a2.append(np.array([4, 0, 0]))
b2.append(np.array([1, 1, -3]))
c2.append(np.array([0.5, 2, -2]))

# 7
a1.append(np.array([0, 0, 0]))
b1.append(np.array([0, 4, 0]))
c1.append(np.array([4, 0, 0]))

a2.append(np.array([1, 2, 0]))
b2.append(np.array([2, 1, 0]))
c2.append(np.array([0.5, 2, -2]))

# 8
a1.append(np.array([0, 0, 0]))
b1.append(np.array([0, 4, 0]))
c1.append(np.array([4, 0, 0]))

a2.append(np.array([-1, 2, 2]))
b2.append(np.array([0, 2, 2]))
c2.append(np.array([0, 0, -2]))

# 9
a1.append(np.array([0, 0, 0]))
b1.append(np.array([0, 4, 0]))
c1.append(np.array([5, 0, 0]))

a2.append(np.array([1, 1, 2]))
b2.append(np.array([5, 6, -2]))
c2.append(np.array([3, -4, 1]))

i = 0
ran = range(len(a1))
#ran = range(6, 7)
for i in ran:
    ax = a3.Axes3D(pl.figure())
    ax.set_xlim((-5, 5))
    ax.set_ylim((-5, 7))
    ax.set_zlim((-5, 7))
    vtx1 = [a1[i], b1[i], c1[i]]
    vtx2 = [a2[i], b2[i], c2[i]]
    tri1 = a3.art3d.Poly3DCollection([vtx1])
    tri2 = a3.art3d.Poly3DCollection([vtx2])
    tri1.set_color(colors.rgb2hex(sp.rand(3)))
    tri1.set_edgecolor('k')
    tri2.set_color(colors.rgb2hex(sp.rand(3)))
    tri2.set_edgecolor('k')
    ax.add_collection3d(tri1)
    ax.add_collection3d(tri2)
    print(isIntersecting(a1[i], b1[i], c1[i], a2[i], b2[i], c2[i]))
    # i += 1
    pl.show()
