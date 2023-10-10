# %%
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import solidspy.assemutil as ass # Solidspy 1.1.0
import solidspy.postprocesor as pos 
import solidspy.uelutil as uel 
import meshio
import pyvista as pv

from ESO_utils import *

np.seterr(divide='ignore', invalid='ignore')

mesh = meshio.read("meshes/modelo10x10x30_3.msh")
points = mesh.points
cells = mesh.cells

nodes = np.zeros((points.shape[0], 7))
nodes[:,0] = np.arange(0,points.shape[0])
nodes[:,1:4] = points
nodes[np.unique(cells[1].data.flatten()),-3:] = -1

n_loads = np.unique(cells[0].data.flatten()).shape[0]

loads = np.zeros((n_loads,4))
loads[:,0] = np.unique(cells[0].data.flatten())
loads[:,1] = 1/n_loads

els = np.zeros((cells[-1].data.shape[0], 11), dtype=int)
els[:,0] = np.arange(0,cells[-1].data.shape[0], dtype=int)
els[:,1:3] = [24,0]
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


# FEM
kloc, _ = ass.retriever(els, mats, nodes[:,:4], -1, uel=uel.elast_hex8)

assem_op, bc_array, neq = DME(nodes[:, -3:], els, ndof_node=3, ndof_el_max=ndof)

# System assembly
stiff_mat = sparse_assem(els, mats, nodes[:, :4], neq, assem_op, uel=uel.elast_hex8)
rhs_vec = ass.loadasem(loads, bc_array, neq, ndof_node=3)

disp = spsolve(stiff_mat, rhs_vec)
UC = pos.complete_disp(bc_array, nodes, disp, ndof_node=3)

colormap = 'viridis'
d_uc = UC[:,1]
normalized_disp = (d_uc-d_uc.min())/(d_uc.max()-d_uc.min())
pv.set_plot_theme("document")

p = pv.Plotter()
p.add_mesh(
    mesh=pv.from_meshio(mesh),
    scalars=normalized_disp,
    cmap=colormap,
    show_scalar_bar=True,
    show_edges=True,
    lighting=True
)
p.show_axes()
p.show()