# %%
import matplotlib.pyplot as plt # Package for plotting
from matplotlib import colors 
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

x_j = np.ones(ny*nx)*0.5 # Initialize the sensitivity
x_min=1e-8 # Minimum young modulus of the material
x_max=1.0 # Maximum young modulus of the material
s = 0.0002
volfrac = 0.7
v_j = np.ones((els.shape[0])) * volume(length, height, nx, ny)
v_max = volume(length, height, nx, ny) * int(els.shape[0] * volfrac)
lamb = 5
penal = 3 # Penalization factor
L_j = x_j - s*(x_max-x_min)

for _ in range(100):

    # Check convergence
    if change < 0.01:
        print('Convergence reached')
        break

    mats[:,2] = x_min+x_j**penal*(x_max-x_min) # Update the Young modulus

    # System solve
    stiff_mat = sparse_assem(els, mats, nodes[:, :3], neq, assem_op, kloc)
    rhs_vec = ass.loadasem(loads, bc_array, neq)
    disp = spsolve(stiff_mat, rhs_vec)
    UC = pos.complete_disp(bc_array, nodes, disp)

    g = rhs_vec.T.dot(disp)
    d_g = sensi_el(els, UC, kloc)

    x_j = optimality_criterion(x_j, x_min, x_max)
    x_j_ = x_j.copy()

    q_o = -(x_j - L_j)**2 * d_g
    r_o = g - q_o.sum()

    for _ in range(1000):
        grad = gradient(x_j, v_max, v_j)
        
        lamb -= s * grad
        
        # Check for convergence
        if abs(grad) < 0.01:
            break

    x_j = L_j + np.sqrt(q_o/(lamb*v_j))
    x_j = optimality_criterion(x_j, x_min, x_max)

    print(x_j.min(),  x_j.max())

    L_j = x_j - s*(x_j_-L_j)

    change = np.linalg.norm(x_j-x_j_)
    print(change)


# %%
E_nodes, S_nodes = pos.strain_nodes(nodes, els, mats[:,:2], UC)
pos.fields_plot(els, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes)

# %% Plot
plt.ion() 
fig,ax = plt.subplots()
im = ax.imshow(-x_j.reshape(ny, nx), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
fig.show()