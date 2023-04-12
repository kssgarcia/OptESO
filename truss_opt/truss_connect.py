# -*- coding: utf-8 -*-
"""

"""
#%% Auxiliar functions
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import solidspy.postprocesor as pos
import solidspy.assemutil as ass
import solidspy.solutil as sol


def fem_sol(nodes, elements, mats, loads):
    DME, IBC , neq = ass.DME(nodes, elements)
    KG = ass.assembler(elements, mats, nodes, neq, DME)
    RHSG = ass.loadasem(loads, IBC, neq)
    UG = sol.static_sol(KG, RHSG)
    UC = pos.complete_disp(IBC, nodes, UG)
    return UC


def weight(areas, nodes, elements):
    nels = areas.shape[0]
    lengths = np.zeros_like(areas)
    for cont in range(nels):
        ini = elements[cont, 3]
        end = elements[cont, 4]
        lengths[cont] = np.linalg.norm(nodes[end, 1:3] - nodes[ini, 1:3])
    return np.sum(areas * lengths)


def stiff(areas, nodes, elements, mats, loads):
    mats2 = mats.copy()
    mats2[:, 1] = areas
    disp = fem_sol(nodes, elements, mats2, loads)
    return 10.0 - np.linalg.norm(disp)**2


def stress_cons(areas, nodes, elements, mats, loads, stresses, comp):
    mats2 = mats.copy()
    mats2[:, 1] = areas
    disp = fem_sol(nodes, elements, mats2, loads)
    cons = np.asarray(stresses) -\
        pos.stress_truss(nodes, elements, mats2, disp)
    return cons[comp]


def plot_truss(nodes, elements, mats):
    min_area = mats[:, 1].min()
    max_area = mats[:, 1].max()
    areas = mats[:, 1].copy()
    max_val = 4
    min_val = 0.5
    if max_area - min_area > 1e-6:
        widths = (max_val - min_val)*(areas - min_area)/(max_area - min_area)\
            + min_val
    else:
        widths = np.ones_like(areas)
    for el in elements:
        if areas[el[2]] > 1e-5:
            ini, end = el[3:]
            plt.plot([nodes[ini, 1], nodes[end, 1]],
                     [nodes[ini, 2], nodes[end, 2]],
                     color="black", lw=widths[el[2]])


#%% Problem definition
length = 4
height = 3
nx = 4
ny = 3
nels = nx*ny*(nx*ny - 1)
y, x = np.mgrid[0:height:ny*1j, 0:length:nx*1j]
x.shape = nx*ny
y.shape = nx*ny
nodes = np.zeros((nx*ny, 5))
nodes[:, 0] = range(nx*ny)
nodes[:, 1] = x
nodes[:, 2] = y
print(nels, x.shape)
#nodes[np.bitwise_and(x == 0, y == height/2), 3:] = -1
#nodes[np.bitwise_and(x == length, y == 0), 3:] = -1
#nodes[np.bitwise_and(x == 0, np.abs(y) < height/4), 3:] = -1
nodes[x == 0, 3:] = -1
elements = np.zeros((nels, 5), dtype=np.int)
elements[:, 0] = range(nels)
elements[:, 1] = 6
elements[:, 2] = range(nels)
elements[:, 3:] = [[ini, end]
                    for ini in range(nx*ny)
                    for end in range(nx*ny) if ini != end]
mats = np.ones((nels, 2))
loads = np.zeros((1, 3))
loads[0, 0] = nodes[np.bitwise_and(x == length, y == height/2), 0]
loads[0, 2] = -1.0

#areas = np.random.uniform(low=0.1, high=1.0, size=nels)
areas = np.ones(nels)
mats[:, 1] = areas

#%% Optimization
bnds = [(1e-6, 1.0) for cont in range(nels)]
stiff_cons = [{'type': 'ineq', 'fun': stiff,
         'args': (nodes, elements, mats, loads)}]
res = minimize(weight, x0=areas, args=(nodes, elements), bounds=bnds,
               constraints=stiff_cons, method="SLSQP", tol=1e-6,
               options={"maxiter": 100, "disp":True})

#%% Results
mats2 = mats.copy()
mats2[:, 1] = res.x
disp = fem_sol(nodes, elements, mats, loads)
print("Weigth: {}".format(weight(areas, nodes, elements)))
disp = fem_sol(nodes, elements, np.column_stack((mats[:,0], res.x)), loads)
print("Weigth: {}".format(weight(res.x, nodes, elements)))

#%% Plotting
plt.figure(figsize=(12, 4))
plt.subplot(121)
plot_truss(nodes, elements, mats)
plt.subplot(122)
plot_truss(nodes, elements, mats2)
plt.show()
