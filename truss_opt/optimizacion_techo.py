# -*- coding: utf-8 -*-
"""
Ejemplo de

@author: Nicolas Guarin-Zapata
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import solidspy.postprocesor as pos
from truss_opt import weight, compliance, fem_sol, plot_truss

nodes = np.array([
           [ 0,  0,  0, -1, -1],
           [ 1,  2,  0,  0,  0],
           [ 2,  4,  0,  0,  0],
           [ 3,  6,  0,  0,  0],
           [ 4,  8,  0,  0,  0],
           [ 5, 10,  0,  0,  0],
           [ 6, 12,  0,  -1, -1],
           [ 7,  2,  1,  0,  0],
           [ 8, 10,  1,  0,  0],
           [ 9,  4,  2,  0,  0],
           [10,  8,  2,  0,  0],
           [11,  6,  3,  0,  0]])
mats = np.column_stack((np.full(21, 1e6),
                        np.full(21, 1e-2)))
eles = np.array([
           [ 0,  6,  0,  0,  1],
           [ 1,  6,  1,  1,  7],
           [ 2,  6,  2,  7,  0],
           [ 3,  6,  3,  1,  2],
           [ 4,  6,  4,  2,  7],
           [ 5,  6,  5,  2,  3],
           [ 6,  6,  6,  3,  9],
           [ 7,  6,  7,  9,  2],
           [ 8,  6,  8,  3,  4],
           [ 9,  6,  9,  4, 10],
           [10,  6, 10, 10,  3],
           [11,  6, 11,  3, 11],
           [12,  6, 12,  4,  5],
           [13,  6, 13,  5,  8],
           [14,  6, 14,  8,  4],
           [15,  6, 15,  8, 10],
           [16,  6, 16,  5,  6],
           [17,  6, 17,  6,  8],
           [18,  6, 18, 10, 11],
           [19,  6, 19, 11,  9],
           [20,  6, 20,  9,  7]])
loads = np.array([
           [ 0,  0, -2],
           [ 6,  0, -2],
           [ 7,  0, -2],
           [ 8,  0, -2],
           [ 9,  0, -2],
           [10,  0, -2],
           [11,  0, -2]])

areas = mats[:, 1].copy()

#%% Optimization
nels = len(areas)
tot_w = 0.5

# Bounds on the areas
bnds = [(1e-3, 0.1) for cont in range(nels)]

# Weigth constraint
weight_fun = lambda areas, nodes, elements, tot_w:\
        tot_w - weight(areas, nodes, elements)
weight_cons = [{'type': 'ineq', 'fun': weight_fun,
    'args': (nodes, eles, tot_w)}]
cons = weight_cons

# Optimization
res = minimize(compliance, x0=areas, args=(nodes, eles, loads, mats),
               bounds=bnds, constraints=cons, method="SLSQP",
               tol=1e-6, options={"maxiter": 500, "disp":True})

#%% Results

# Original design
disp0 = fem_sol(nodes, eles, mats, loads)
weight0 = weight(areas, nodes, eles)
stress0 = pos.stress_truss(nodes, eles, mats, disp0)
compliance0 = compliance(areas, nodes, eles, loads, mats)
print("\nOriginal design: {}".format(areas))
print("Weigth: {}".format(weight0))
print("Stresses: {}".format(stress0))
print("Compliance: {}".format(compliance0))

# Optimized design
mats1 = mats.copy()
mats1[:, 1] = res.x
disp1 = fem_sol(nodes, eles, mats1, loads)
weight1 = weight(res.x, nodes, eles)
stress1 = pos.stress_truss(nodes, eles, mats1, disp1)
compliance1 = compliance(res.x, nodes, eles, loads, mats1)
print("\nOptimized design: ", res.x)
print("Weigth: {}".format(weight1))
print("Stresses: {}".format(stress1))
print("Compliance: {}".format(compliance1))


#%% Plotting
plt.figure(figsize=(12, 4))
plt.subplot(121)
plot_truss(nodes, eles, mats, loads)
plt.subplot(122)
plot_truss(nodes, eles, mats1, loads)
plt.show()