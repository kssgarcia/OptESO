# -*- coding: utf-8 -*-
"""
Ejemplo de

@author: Nicolas Guarin-Zapata
"""
import numpy as np
import matplotlib.pyplot as plt
import solidspy.assemutil as ass
import solidspy.postprocesor as pos
from truss_opt import fem_sol, plot_truss

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

conect = eles[:, -2:]
np.savetxt("techo_conect.txt", conect, fmt="%d")

loads = np.array([
           [ 0,  0, -2],
           [ 6,  0, -2],
           [ 7,  0, -2],
           [ 8,  0, -2],
           [ 9,  0, -2],
           [10,  0, -2],
           [11,  0, -2]])





nodes_unc = nodes.copy()
nodes_unc[:, -2:] = 0
DME, IBC, neq = ass.DME(nodes[:,-2:], eles)
DME_unc, IBC_unc, neq_unc = ass.DME(nodes_unc[:,-2:], eles)
stiff_unc, _ = ass.assembler(eles, mats, nodes_unc[:,:-2], neq_unc, DME_unc)



#%% Results
disp_comp = fem_sol(nodes, eles, mats, loads)



disp_comp.shape = 24
loads_comp = stiff_unc @ disp_comp
loads_comp.shape = 12, 2
coords = np.zeros((12, 4))
coords[:, :2] = nodes[:, 1:3]
coords[:, 2:] = loads_comp
np.savetxt("techo_nodos.txt", coords, fmt="%g")


#%% Plotting
plt.figure(figsize=(6, 4))
plot_truss(nodes, eles, mats, loads)
plt.show()