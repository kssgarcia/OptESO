# %% Initialization
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import solidspy.assemutil as ass # Solidspy 1.1.0
import solidspy.postprocesor as pos 
import solidspy.uelutil as uel 
import solidspy.gaussutil as gaus
import solidspy.femutil as fe
import meshio

from Utils.ESO_utils import *

np.seterr(divide='ignore', invalid='ignore')

mesh = meshio.read("meshes/modelo10x10x30_1.msh")
points = mesh.points
cells = mesh.cells

nodes = np.zeros((points.shape[0], 7))
nodes[:,0] = np.arange(0,points.shape[0])
nodes[:,1:4] = points
nodes[np.unique(cells[1].data.flatten()),-3:] = -1

n_loads = np.unique(cells[0].data.flatten()).shape[0]

loads = np.zeros((n_loads,4))
loads[:,0] = np.unique(cells[0].data.flatten())
loads[:,2] = 1/n_loads

els = np.zeros((cells[-1].data.shape[0], 11), dtype=int)
els[:,0] = np.arange(0,cells[-1].data.shape[0], dtype=int)
els[:, 1:3] = [24,0]
els[:,-8:] = cells[-1].data
nels = els.shape[0]
ndof = 24

mats = np.zeros((els.shape[0], 3))
mats[:] = [1,0.28,1]

BC = np.argwhere(nodes[:,-1]==-1)[:,0]

niter = 200
RR = 0.02 # Initial removal ratio
ER = 0.02 # Removal ratio increment
ELS = None

kloc, _ = ass.retriever(els, mats, nodes[:,:4], -1, uel=uel.elast_hex8)


# %%

def str_el8(coord, ul):
    epsG = np.zeros([6, 8])
    xl = np.zeros([8, 3])
    gpts, _ = gaus.gauss_nd(8,ndim=3)
    for i in range(8):
        ri, si, ti = gpts[i, :]
        H, B, _ = fe.elast_diff_3d(ri, si, ti, coord)
        epsG[:, i] = B @ ul
        xl[i, 0] = np.dot(H[0, ::3], coord[:, 0])
        xl[i, 1] = np.dot(H[0, ::3], coord[:, 1])
        xl[i, 2] = np.dot(H[0, ::3], coord[:, 2])
    return epsG.T, xl

def strain_n(nodes, elements, mats, sol_complete):
    nelems = elements.shape[0]
    nnodes = nodes.shape[0]
    ndof, nnodes_elem = (24,8)

    elcoor = np.zeros([nnodes_elem, 3])
    E_nodes = np.zeros([nnodes, 3])
    S_nodes = np.zeros([nnodes, 3])
    el_nodes = np.zeros([nnodes], dtype=int)
    ul = np.zeros([ndof])
    IELCON = elements[:, 3:]

    for el in range(nelems):
        young, poisson = mats[int(elements[el, 2]), -2:]
        shear = young/(2*(1 + poisson))
        fact1 = young/(1 - poisson**2)
        fact2 = poisson*young/(1 - poisson**2)
        elcoor[:, 0] = nodes[IELCON[el, :], 1]
        elcoor[:, 1] = nodes[IELCON[el, :], 2]
        elcoor[:, 2] = nodes[IELCON[el, :], 3]
        ul[0:8] = sol_complete[IELCON[el, :], 0]
        ul[8:16] = sol_complete[IELCON[el, :], 1]
        ul[16:] = sol_complete[IELCON[el, :], 2]
        epsG, _ = str_el8(elcoor, ul)

        for cont, node in enumerate(IELCON[el, :]):
            E_nodes[node, 0] += epsG[cont, 0]
            E_nodes[node, 1] += epsG[cont, 1]
            E_nodes[node, 2] += epsG[cont, 2]
            S_nodes[node, 0] += fact1*epsG[cont, 0]  + fact2*epsG[cont, 1]
            S_nodes[node, 1] += fact2*epsG[cont, 0]  + fact1*epsG[cont, 1]
            S_nodes[node, 2] += shear*epsG[cont, 2]
            el_nodes[node] = el_nodes[node] + 1

    E_nodes[:, 0] /= el_nodes
    E_nodes[:, 1] /= el_nodes
    E_nodes[:, 2] /= el_nodes
    S_nodes[:, 0] /= el_nodes
    S_nodes[:, 1] /= el_nodes
    S_nodes[:, 2] /= el_nodes
    return E_nodes, S_nodes

for _ in range(1):

    if not is_equilibrium(nodes, mats, els, loads):
        print('Convergence reached')
        break

    assem_op, bc_array, neq = DME(nodes[:, -3:], els, ndof_node=3, ndof_el_max=ndof)


    ELS = els.copy()

    # System assembly
    stiff_mat = sparse_assem(els, mats, nodes[:, :4], neq, assem_op, uel=uel.elast_hex8)
    rhs_vec = ass.loadasem(loads, bc_array, neq, ndof_node=3)

    disp = spsolve(stiff_mat, rhs_vec)
    UC = pos.complete_disp(bc_array, nodes, disp, ndof_node=3)
    strain_nodes, stress_nodes = None, None
    strain_nodes, stress_nodes = strain_n(nodes, els, mats, UC)
    print(stress_nodes)

    RR += ER

    print(RR)

# %% Get data to plot

#E_els = strain_els(ELS, UC)
E_els = strain_els(ELS, strain_nodes)
E_els /= E_els.max()

cmap = plt.get_cmap('viridis')
colors = cmap(UC[:,1])

nodes_plot = nodes[:,1:4]
hexahedra = ELS[:,-8:]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the hexahedra elements by plotting each face
for index, element in enumerate(hexahedra):
    vertices = [nodes_plot[i] for i in element]
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]]
    ]
    
    ax.add_collection3d(Poly3DCollection(faces, facecolors=colors[index], linewidths=0.1, edgecolors='k', alpha=1))

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set the plot limits
ax.set_xlim(0, 5)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

# Show the plot
plt.show()

# %% Scatter plot 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(nodes[:,1], nodes[:,2], nodes[:,3], c=UC[:,2], marker='o', label='Random Points')

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
