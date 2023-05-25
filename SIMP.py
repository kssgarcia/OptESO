# %% Initialization
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation
import numpy as np
from beams import *
from SIMP_utils import *
# Solidspy 1.1.0

np.seterr(divide='ignore', invalid='ignore')

# Mesh
length = 160
height = 40
nx = 200
ny= 40
nodes, mats, els, loads, BC = beam(L=length, H=height, nx=nx, ny=ny, n=2)

# Calculate centers and volumes
niter = 60
centers = center_els(nodes, els)
Vi = volume(els, length, height, nx, ny)

# Initialize the design variables
r_min = np.linalg.norm(nodes[0,1:3] - nodes[1,1:3]) * 4
penal = 3
Emin=1e-9
Emax=1.0
volfrac = 0.5
change = 10
g = 0

# Initialize the density, sensitivity and the iteration history
rho = volfrac * np.ones(ny*nx,dtype=float)
sensi_rho = np.ones(ny*nx)
rho_old = rho.copy()
d_c = np.ones(ny*nx)
d_v = np.ones(ny*nx)
rho_data = []

# %% Optimization loop

for i in range(niter):

    if change < 0.01:
        print('Convergence reached')
        break

    mats[:,2] = Emin+rho**penal*(Emax-Emin)

    IBC, UG, rhs_vec = preprocessing(nodes, mats, els, loads)
    UC, *_ = postprocessing(nodes, mats[:,:2], els, IBC, UG, strain_sol=False)

    # Sensitivity analysis
    sensi_rho[:] = sensitivity_els(nodes, mats, els, UC, nx, ny)
    obj = ((Emin+rho**penal*(Emax-Emin))*sensi_rho).sum()
    d_c[:] = (-penal*rho**(penal-1)*(Emax-Emin))*sensi_rho
    d_v[:] = np.ones(ny*nx)
    d_c[:] = density_filter(centers, r_min, rho, d_c)

    # Optimality criteria
    rho_old[:] = rho
    rho[:], g = optimality_criteria(nx, ny, rho, d_c, d_v, g)

    # Compute the change
    change = np.linalg.norm(rho.reshape(nx*ny,1)-rho_old.reshape(nx*ny,1),np.inf)

    if i%5 == 0:
        rho_data.append(-rho.reshape((ny, nx)))

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