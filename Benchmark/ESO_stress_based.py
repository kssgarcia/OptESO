# %%
import matplotlib.pyplot as plt # Package for plotting
import numpy as np # Package for scientific computing

from Utils.ESO_utils import * # Fucntions for FEM analysis and postprocessing
from Utils.beams import * # Functions for mesh generation

# Solidspy 1.1.0
import solidspy.postprocesor as pos # SolidsPy package for postprocessing
np.seterr(divide='ignore', invalid='ignore') # Ignore division by zero error

length = 60
height = 60
nx = 60
ny= 60
nodes, mats, els, loads, BC = beam(L=length, H=height, nx=nx, ny=ny, n=61)
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
