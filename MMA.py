# %%
import matplotlib.pyplot as plt # Package for plotting
import numpy as np # Package for scientific computing
import solidspy.assemutil as ass # Solidspy 1.1.0
from scipy.sparse.linalg import spsolve
import solidspy.postprocesor as pos 

from beams import * # Functions for mesh generation
from MMA_utils import * # Fucntions for FEM analysis and postprocessing
# Solidspy 1.1.0
np.seterr(divide='ignore', invalid='ignore')

# Mesh
length = 160
height = 40
nx = 200
ny= 40
nodes, mats, els, loads, BC = beam(L=length, H=height, nx=nx, ny=ny, n=2) # Generate mesh

niter = 60

E = mats[0,0] # Young modulus
nu = mats[0,1] # Poisson ratio
k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8]) # Coefficients
kloc = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]], 
[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]]); # Local stiffness matrix
assem_op, bc_array, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8) 

change = 10

for _ in range(1):

    # Check convergence
    if change < 0.01:
        print('Convergence reached')
        break


    # System assembly
    stiff_mat = sparse_assem(els, mats, nodes[:, :3], neq, assem_op, kloc)
    rhs_vec = ass.loadasem(loads, bc_array, neq)

    # System solution
    disp = spsolve(stiff_mat, rhs_vec)
    UC = pos.complete_disp(bc_array, nodes, disp)

# %%
E_nodes, S_nodes = pos.strain_nodes(nodes, els, mats[:,:2], UC)
pos.fields_plot(els, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes)