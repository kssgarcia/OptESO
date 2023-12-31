# %%
import matplotlib.pyplot as plt # Package for plotting
import numpy as np # Package for scientific computing

from Utils.ESO_utils import * # Fucntions for FEM analysis and postprocessing
from Utils.beams import * # Functions for mesh generation

# Solidspy 1.1.0
import solidspy.postprocesor as pos # SolidsPy package for postprocessing
np.seterr(divide='ignore', invalid='ignore') # Ignore division by zero error

def optimization_ESO_stress():
    length = 20
    height = 10
    nx = 50
    ny= 20
    dirs = np.array([[0,-1]])
    positions = np.array([[10,1]])
    nodes, mats, els, loads, BC = beam(L=length, H=height, nx=nx, ny=ny, dirs=dirs, positions=positions, n=1)
    elsI,nodesI = np.copy(els), np.copy(nodes)

    IBC, UG = preprocessing(nodes, mats, els, loads) # Calculate boundary conditions and global stiffness matrix
    UCI, E_nodesI, S_nodesI = postprocessing(nodes, mats[:,:2], els, IBC, UG) # Calculate displacements, strains and stresses

    niter = 200
    RR = 0.001 # Initial removal ratio
    ER = 0.005 # Removal ratio increment
    V_opt = volume(els, length, height, nx, ny) * 0.50 # Optimal volume

    ELS = None
    for _ in range(niter):

        # Check equilibrium
        if not is_equilibrium(nodes, mats, els, loads) or volume(els, length, height, nx, ny) < V_opt: break  # Check equilibrium/volume and stop if not
        ELS = els
        
        # FEW analysis
        IBC, UG = preprocessing(nodes, mats, els, loads) # Calculate boundary conditions and global stiffness matrix
        UC, E_nodes, S_nodes = postprocessing(nodes, mats[:,:2], els, IBC, UG) # Displacements, strains and stresses
        E_els, S_els = strain_els(els, E_nodes, S_nodes) # Calculate strains and stresses in elements
        vons = np.sqrt(S_els[:,0]**2 - (S_els[:,0]*S_els[:,1]) + S_els[:,1]**2 + 3*S_els[:,2]**2)

        # Remove/add elements
        RR_el = vons/vons.max() # Relative stress
        mask_del = RR_el < RR # Mask for elements to be deleted
        mask_els = protect_els(els, loads, BC) # Mask for elements to be protected
        mask_del *= mask_els  
        els = np.delete(els, mask_del, 0) # Delete elements
        del_node(nodes, els) # Delete nodes that are not connected to any element

        RR += ER
    print(RR)
    pos.fields_plot(elsI, nodes, UCI, E_nodes=E_nodesI, S_nodes=S_nodesI) # Plot initial state
    pos.fields_plot(ELS, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes) # Plot final state
    fill_plot = np.ones(E_nodes.shape[0])
    plt.figure()
    tri = pos.mesh2tri(nodes, ELS)
    plt.tricontourf(tri, fill_plot, cmap='binary')
    plt.axis("image");

if __name__ == "__main__":
    pass
    #optimization_ESO_stress()