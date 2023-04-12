# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from truss_opt import fem_sol, weight, compliance, plot_truss, grid_truss


#%% Problem definition
length = 10
height = 4
nx = 10
ny = 4
P = 1.0
nodes, elements = grid_truss(length, height, nx, ny)
x = nodes[:, 1]
y = nodes[:, 2]
nodes[np.bitwise_and(x == -length/2, y == -height/2), 3:] = -1
nodes[np.bitwise_and(x == length/2, y == -height/2), 3:] = -1
nels = elements.shape[0]
nloads = nodes[y==height/2, 0].shape[0]
loads = np.zeros((nloads, 3))
loads[:, 0] = nodes[y==height/2, 0]
loads[:, 2] = -P/nloads
areas = 0.5*np.ones((nels))
mats = np.ones((nels, 2))
mats[:, 1] = areas



#%% Optimization of compliance
bnds = [(1e-6, 0.5) for cont in range(nels)]
def max_weight(areas, nodes, elements):
    return 5.0 - weight(areas, nodes, elements)
cons = [{'type': 'ineq', 'fun': max_weight,
         'args': (nodes, elements)}]
res = minimize(compliance, x0=areas, args=(nodes, elements, loads, mats),
               bounds=bnds,
               constraints=cons, method="SLSQP", tol=1e-3,
               options={"maxiter": 100, "disp":True})

#%% Results
mats2 = mats.copy()
mats2[:, 1] = res.x
disp = fem_sol(nodes, elements, mats, loads)
print("Initial weigth: {}".format(weight(areas, nodes, elements)))
disp = fem_sol(nodes, elements, np.column_stack((mats[:,0], res.x)), loads)
print("Final weigth: {}".format(weight(res.x, nodes, elements)))

#%% Plotting
plt.figure(figsize=(12, 4))
plt.subplot(121)
plot_truss(nodes, elements, mats, loads)
plt.subplot(122)
plot_truss(nodes, elements, mats2, loads, tol=1e-5)
plt.show()