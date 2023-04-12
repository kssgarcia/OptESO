# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import matplotlib.pyplot as plt
from truss_opt import  plot_truss, grid_truss


#%% Problem definition
length = 6
height = 3
nx = 6
ny = 3
P = 1.0
nodes, elements = grid_truss(length, height, nx, ny)
x = nodes[:, 1]
y = nodes[:, 2]
nodes[x==-length/2, 3] = -1
nodes[np.bitwise_and(x == -length/2, y == -height/2), 4] = -1
nels = elements.shape[0]
P = 1.0
mats = np.ones((nels, 2))
loads = np.zeros((ny, 3))
loads[:, 0] = nodes[x==length/2, 0]
loads[:, 2] = -P/ny
areas = 0.5*np.ones((nels))
mats[:, 1] = areas
plot_truss(nodes, elements, mats, loads)
plt.show()