# %% Initialization
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import solidspy.assemutil as ass # Solidspy 1.1.0
import solidspy.postprocesor as pos 
import solidspy.uelutil as uel 
import meshio

from SIMP_utils_3d import *

np.seterr(divide='ignore', invalid='ignore')

mesh = meshio.read("modelo1.msh")
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

niter = 20
penal = 3 # Penalization factor
Emin=1e-9 # Minimum young modulus of the material
Emax=1.0 # Maximum young modulus of the material
volfrac = 0.5 # Volume fraction

# Initialize the design variables
change = 100 # Change in the design variable
g = 0 # Constraint
rho = volfrac * np.ones(nels, dtype=float) # Initialize the density
sensi_rho = np.ones(nels) # Initialize the sensitivity
rho_old = rho.copy() # Initialize the density history
d_c = np.ones(nels) # Initialize the design change

centers = center_els(nodes, els) # Calculate centers
r_min = np.linalg.norm(centers[0] - centers[1]) # Radius for the sensitivity filter
E = mats[0,0] # Young modulus
nu = mats[0,1] # Poisson ratio

assem_op, bc_array, neq = DME(nodes[:, -3:], els, ndof_node=3, ndof_el_max=ndof)
kloc, _ = ass.retriever(els, mats, nodes[:,:4], -1, uel=uel.elast_hex8)

for _ in range(niter):

    # Check convergence
    if change < 0.01 or not is_equilibrium(nodes, mats, els, loads):
        print('Convergence reached')
        break

    # Change density 
    mats[:,2] = Emin+rho**penal*(Emax-Emin)

    # System assembly
    stiff_mat = sparse_assem(els, mats, nodes[:, :4], neq, assem_op, uel=uel.elast_hex8)
    rhs_vec = ass.loadasem(loads, bc_array, neq, ndof_node=3)

    disp = spsolve(stiff_mat, rhs_vec)
    UC = pos.complete_disp(bc_array, nodes, disp, ndof_node=3)

    # Sensitivity analysis
    sensi_rho[:] = (np.dot(UC[els[:,-8:]].reshape(nels,ndof),kloc) * UC[els[:,-8:]].reshape(nels,ndof) ).sum(1)
    d_c[:] = (-penal*rho**(penal-1)*(Emax-Emin))*sensi_rho
    d_c[:] = density_filter(centers, r_min, rho, d_c)

    # Optimality criteria
    rho_old[:] = rho.copy()
    rho[:], g = optimality_criteria(nels, rho, d_c, g)

    # Compute the change
    change = np.linalg.norm(rho.reshape(nels,1)-rho_old.reshape(nels,1),np.inf)
    print(change)

# %% Get data to plot

mask = rho > 0.5
mask_els = protect_els(els[np.invert(mask)], els.shape[0], loads[:,0], BC)
mask = np.bitwise_or(mask, mask_els)
del_node(nodes, els[mask], loads[:,0], BC)
els = els[mask]

nodes_plot = nodes[:,1:4]
hexahedra = els[:,-8:]

assem_op, bc_array, neq = DME(nodes[:, -3:], els, ndof_node=3, ndof_el_max=ndof)
kloc, _ = ass.retriever(els, mats, nodes[:,:4], -1, uel=uel.elast_hex8)

# System assembly
stiff_mat = sparse_assem(els, mats, nodes[:, :4], neq, assem_op, uel=uel.elast_hex8)
rhs_vec = ass.loadasem(loads, bc_array, neq, ndof_node=3)

disp = spsolve(stiff_mat, rhs_vec)
UC = pos.complete_disp(bc_array, nodes, disp, ndof_node=3)

E_els = strain_els(els, UC)
E_els /= E_els.max()

cmap = plt.get_cmap('viridis')
colors = cmap(E_els)

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
