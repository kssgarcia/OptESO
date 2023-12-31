# %%
import matplotlib.pyplot as plt # Package for plotting
import numpy as np # Package for scientific computing

from Utils.ESO_utils import * # Fucntions for FEM analysis and postprocessing
from Utils.beams import * # Functions for mesh generation

# Solidspy 1.1.0
import solidspy.postprocesor as pos # SolidsPy package for postprocessing
np.seterr(divide='ignore', invalid='ignore') # Ignore division by zero error

def optimization_ESO_stiff():
    length = 60
    height = 60
    nx = 60
    ny= 40
    dirs = np.array([[0,-1]])
    positions = np.array([[10,1]])
    nodes, mats, els, loads, BC = beam(L=length, H=height, nx=nx, ny=ny, dirs=dirs, positions=positions, n=1)
    elsI= np.copy(els)

    IBC, UG = preprocessing(nodes, mats, els, loads) # Calculate boundary conditions and global stiffness matrix
    UCI, E_nodesI, S_nodesI = postprocessing(nodes, mats[:,:2], els, IBC, UG) # Calculate displacements, strains and stresses

    niter = 200
    RR = 0.005 # Initial removal ratio
    ER = 0.05 # Removal ratio increment
    V_opt = volume(els, length, height, nx, ny) * 0.20 # Optimal volume
    ELS = None
    for _ in range(niter):
        # Check equilibrium
        if not is_equilibrium(nodes, mats, els, loads) or volume(els, length, height, nx, ny) < V_opt: 
            print('gla')
            break # Check equilibrium/volume and stop if not
        
        # FEW analysis
        IBC, UG = preprocessing(nodes, mats, els, loads) # Calculate boundary conditions and global stiffness matrix
        UC, E_nodes, S_nodes = postprocessing(nodes, mats[:,:2], els, IBC, UG) # Calculate displacements, strains and stresses

        # Compute Sensitivity number
        sensi_number = sensi_el(nodes, mats, els, UC) # Sensitivity number
        mask_del = sensi_number < RR # Mask of elements to be removed
        mask_els = protect_els(els, loads, BC) # Mask of elements to do not remove
        mask_del *= mask_els # Mask of elements to be removed and not protected
        ELS = els # Save last iteration elements
        
        # Remove/add elements
        els = np.delete(els, mask_del, 0) # Remove elements
        del_node(nodes, els) # Remove nodes

        RR += ER

    pos.fields_plot(elsI, nodes, UCI, E_nodes=E_nodesI, S_nodes=S_nodesI) # Plot initial mesh
    pos.fields_plot(ELS, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes) # Plot optimized mesh

    fill_plot = np.ones(E_nodes.shape[0])
    plt.figure()
    tri = pos.mesh2tri(nodes, ELS)
    plt.tricontourf(tri, fill_plot, cmap='binary')
    plt.axis("image");

if __name__ == "__main__":
    pass
    #optimization_ESO_stiff()