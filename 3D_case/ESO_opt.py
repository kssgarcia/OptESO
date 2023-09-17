# %% Initialization
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import solidspy.assemutil as ass # Solidspy 1.1.0
import solidspy.postprocesor as pos 
import solidspy.uelutil as uel 
import meshio

from ESO_utils import *

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

niter = 100
RR = 0.01 # Initial removal ratio
ER = 0.05 # Removal ratio increment
ELS = None

kloc, _ = ass.retriever(els, mats, nodes[:,:4], -1, uel=uel.elast_hex8)

for _ in range(niter):

    if not is_equilibrium(nodes, mats, els, loads):
        print('Convergence reached')
        break

    assem_op, bc_array, neq = DME(nodes[:, -3:], els, ndof_node=3, ndof_el_max=ndof)

    # System assembly
    stiff_mat = sparse_assem(els, mats, nodes[:, :4], neq, assem_op, uel=uel.elast_hex8)
    rhs_vec = ass.loadasem(loads, bc_array, neq, ndof_node=3)

    disp = spsolve(stiff_mat, rhs_vec)
    UC = pos.complete_disp(bc_array, nodes, disp, ndof_node=3)

    # Compute Sensitivity number
    sensi_number = sensi_el(nodes, mats, els, UC, kloc) # Sensitivity number
    print(sensi_number.max())
    mask_del = sensi_number < RR # Mask of elements to be removed
    mask_els = protect_els(els, loads[:,0], BC) # Mask of elements to do not remove
    mask_del *= mask_els # Mask of elements to be removed and not protected
    
    # Remove/add elements
    els = np.delete(els, mask_del, 0) # Remove elements
    del_node(nodes, els, loads[:,0], BC)
    print(els.shape[0])

    RR += ER

    print(RR)

# %% Get data to plot

E_els = strain_els(els, UC)
E_els /= E_els.max()

cmap = plt.get_cmap('viridis')
colors = cmap(E_els)

nodes_plot = nodes[:,1:4]
hexahedra = els[:,-8:]

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