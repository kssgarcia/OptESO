# -*- coding: utf-8 -*-
"""
Examples of truss optimization.

"""
import numpy as np
from scipy.optimize import minimize
import solidspy.postprocesor as pos
import solidspy.assemutil as ass
import solidspy.solutil as sol


#%% Auxiliar functions
def fem_sol(nodes, elements, mats, loads):
    DME, IBC , neq = ass.DME(nodes, elements)
    KG = ass.assembler(elements, mats, nodes, neq, DME)
    RHSG = ass.loadasem(loads, IBC, neq)
    UG = sol.static_sol(KG, RHSG)
    UC = pos.complete_disp(IBC, nodes, UG)
    return UC


def weight(areas, nodes, elements):
    lengths = np.zeros_like(areas)
    ini = elements[:, 3]
    end = elements[:, 4]
    lengths = np.linalg.norm(nodes[end, 1:3] - nodes[ini, 1:3], axis=1)

    return np.sum(areas * lengths)


def stiff(areas, nodes, elements, mats, loads):
    mats2 = mats.copy()
    mats2[:, 1] = areas
    disp = fem_sol(nodes, elements, mats2, loads)
    return 1 - np.linalg.norm(disp)**2


def stress_cons(areas, nodes, elements, mats, loads, stresses, comp):
    mats2 = mats.copy()
    mats2[:, 1] = areas
    disp = fem_sol(nodes, elements, mats2, loads)
    cons = np.asarray(stresses) -\
        pos.stress_truss(nodes, elements, mats2, disp)
    return cons[comp]


#%% Examples from An Introduction to Structural Optimization

example = 5

## 2.1 Weight min two-bar truss w/ stress contraints
if example == 1:
    angle = np.pi/6
    load = 1.0
    nodes = np.array([
        [0 , 0.0,  0.0, -1, -1],
        [1 , 1.0,  0.0, 0, 0],
        [2 ,  1.0,  1.0,  -1,  -1]])
    elements = np.array([
        [0, 6, 0, 0, 1],
        [1, 6, 1, 1, 2]])
    mats = np.array([
        [1.0, 0.9],
        [1.0, 0.9]])
    loads = np.array([[1, load*np.cos(angle), -load*np.sin(angle)]])

# 2.3 Weight min two-bar truss w/ stress contraints
if example == 3:
    angle = np.pi/6
    load = 1.0
    nodes = np.array([
        [0 , 0.0,  0.0, -1, -1],
        [1 , 1.0,  0.0, 0, 0],
        [2 ,  0.0,  np.tan(angle),  -1,  -1]])
    elements = np.array([
        [0, 6, 0, 2, 1],
        [1, 6, 1, 0, 1]])
    mats = np.array([
        [1.0, 2.0],
        [1.0, 2.0]])
    loads = np.array([[1, 0, -load]])

# 2.5 Weight min three-bar truss w/ stress contraints: Case B
if example == 5:
    load = 1.0
    beta = 10.0
    nodes = np.array([
        [0 , 0.0,  0.0, 0, 0],
        [1 , -1.0,  0.0, -1, -1],
        [2 ,  -np.tan(np.pi/4),  np.tan(np.pi/4),  -1,  -1],
        [3 ,  0,  beta,  -1,  -1]])
    elements = np.array([
        [0, 6, 0, 1, 0],
        [1, 6, 1, 2, 0],
        [2, 6, 2, 3, 0],])
    mats = np.array([
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0]])
    loads = np.array([[0, load, 0]])

areas = mats[:, 1].copy()

#%% Optimization
stresses = [2, 2, 1]
nels = 3
bnds = [(1e-12, None) for cont in range(nels)]
stiff_cons = [{'type': 'ineq', 'fun': stiff,
         'args': (nodes, elements, mats, loads)}]
stress_con = [{'type': 'ineq', 'fun':stress_cons,
    'args': (nodes, elements, mats, loads, stresses, cont)}
    for cont in range(nels)]
eq_cons = [{'type': 'eq', 'fun': lambda areas: areas[0] - areas[2]}]
cons = stiff_cons + eq_cons
res = minimize(weight, x0=areas, args=(nodes, elements), bounds=bnds,
               constraints=cons, method="SLSQP")

#%% Results
mats2 = mats.copy()
mats2[:, 1] = areas[:]
disp = fem_sol(nodes, elements, mats, loads)
print("Original design: {}".format(areas))
print("Weigth: {}".format(weight(areas, nodes, elements)))
print("Stresses: {}".format(pos.stress_truss(nodes, elements, mats, disp)))
disp = fem_sol(nodes, elements, np.column_stack((mats[:,0], res.x)), loads)
print("Optimized design: ", res.x)
print("Weigth: {}".format(weight(res.x, nodes, elements)))
print("Stresses: {}".format(pos.stress_truss(nodes, elements, mats2, disp)))

