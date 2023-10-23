# %% Initialization
import matplotlib.pyplot as plt # Package for plotting
from matplotlib import colors 
from matplotlib import animation # Package for animation
import numpy as np # Package for scientific computing

from beams import * # Functions for mesh generation
from SIMP_utils import * # Fucntions for FEM analysis and postprocessing
# Solidspy 1.1.0
np.seterr(divide='ignore', invalid='ignore')

# Mesh
length = 60
height = 60
nx = 60
ny= 60
nodes, mats, els, loads, BC = beam(L=length, H=height, nx=nx, ny=ny, n=2) # Generate mesh

niter = 60
centers = center_els(nodes, els) # Calculate centers
Vi = volume(els, length, height, nx, ny) # Initial volume

r_min = np.linalg.norm(nodes[0,1:3] - nodes[1,1:3]) * 4 # Radius for the sensitivity filter
penal = 3 # Penalization factor
Emin=1e-9 # Minimum young modulus of the material
Emax=1.0 # Maximum young modulus of the material
volfrac = 0.5 # Volume fraction
change = 10 # Change in the design variable
g = 0 # Constraint

rho = volfrac * np.ones(ny*nx,dtype=float) # Initialize the density
sensi_rho = np.ones(ny*nx) # Initialize the sensitivity
rho_old = rho.copy() # Initialize the density history
d_c = np.ones(ny*nx) # Initialize the design change
d_v = np.ones(ny*nx) # Initialize the volume change
rho_data = [] # History of the density

# %% Optimization loop

for i in range(niter):

    if change < 0.01:
        print('Convergence reached')
        break

    mats[:,2] = Emin+rho**penal*(Emax-Emin) # Update the Young modulus

    IBC, UG, rhs_vec = preprocessing(nodes, mats, els, loads) # Calculate boundary conditions and global stiffness matrix
    UC, *_ = postprocessing(nodes, mats[:,:2], els, IBC, UG, strain_sol=False) # Calculate displacements

    # Sensitivity analysis
    sensi_rho[:] = sensitivity_els(nodes, mats, els, UC, nx, ny) # Calculate the sensitivity
    obj = ((Emin+rho**penal*(Emax-Emin))*sensi_rho).sum() # Calculate the objective function
    d_c[:] = (-penal*rho**(penal-1)*(Emax-Emin))*sensi_rho # Calculate the design change
    d_v[:] = np.ones(ny*nx) # Calculate the volume change
    d_c[:] = density_filter(centers, r_min, rho, d_c) # Perform the sensitivity filter

    # Optimality criteria
    rho_old[:] = rho # Save the old density
    rho[:], g = optimality_criteria(nx, ny, rho, d_c, d_v, g) # Update the density

    change = np.linalg.norm(rho.reshape(nx*ny,1)-rho_old.reshape(nx*ny,1),np.inf) # Calculate the change

    if i%5 == 0:
        rho_data.append(-rho.reshape((ny, nx))) # Save the density

    # Write iteration history to screen (req. Python 2.6 or newer)
    print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(i,obj,(g+volfrac*nx*ny)/(nx*ny),change))

# %% Animation
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((ny,nx)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))

def update(frame):
    rho_frame = rho_data[frame]
    im.set_array(rho_frame)
    return im,
ani = animation.FuncAnimation(fig, update, frames=len(rho_data), interval=200, blit=True)
output_file = "animation.gif"
ani.save(output_file, writer="pillow")

# %% Plot
plt.ion() 
fig,ax = plt.subplots()
im = ax.imshow(-rho.reshape(ny, nx), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
fig.show()

# %%
plt.figure()
plt.scatter(centers[:,0], centers[:,1], c=-rho, cmap='gray')
plt.colorbar()
plt.axis("image");
