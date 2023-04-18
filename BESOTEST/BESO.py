# %%
import matplotlib.pyplot as plt
import numpy as np

from opt_beam import *
from beams import *

# Solidspy 1.1.0
import solidspy.preprocesor as pre
import solidspy.assemutil as ass    
import solidspy.solutil as sol      
import solidspy.postprocesor as pos 
import solidspy.uelutil as uel 

np.seterr(divide='ignore', invalid='ignore')

def is_equilibrium(nodes, mats, els, loads):
    """
    Check if the system is in equilibrium
    
    Get from: https://github.com/AppliedMechanics-EAFIT/SolidsPy/blob/master/solidspy/solids_GUI.py
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    mats : ndarray
        Array with models materials
    els : ndarray
        Array with models elements
    loads : ndarray
        Array with models loads
        
    Returns
    -------
    equil : bool
        Variable True when the system is in equilibrium and False when it doesn't
    """   

    equil = True
    assem_op, bc_array, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8)
    stiff_mat, _ = ass.assembler(els, mats, nodes[:, :3], neq, assem_op)
    rhs_vec = ass.loadasem(loads, bc_array, neq)
    disp = sol.static_sol(stiff_mat, rhs_vec)
    if not np.allclose(stiff_mat.dot(disp)/stiff_mat.max(), rhs_vec/stiff_mat.max()):
        equil = False

    return equil
    
def preprocessing(nodes, mats, els, loads):
    """
    Compute IBC matrix and the static solve.
    
    Get from: https://github.com/AppliedMechanics-EAFIT/SolidsPy/blob/master/solidspy/solids_GUI.py
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    mats : ndarray
        Array with models materials
    els : ndarray
        Array with models elements
    loads : ndarray
        Array with models loads
        
    Returns
    -------
    bc_array : ndarray 
        Boundary conditions array
    disp : ndarray 
        Static displacement solve.
    """   

    assem_op, bc_array, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8)

    # System assembly
    stiff_mat, _ = ass.assembler(els, mats, nodes[:, :3], neq, assem_op)
    rhs_vec = ass.loadasem(loads, bc_array, neq)

    # System solution
    disp = sol.static_sol(stiff_mat, rhs_vec)
    if not np.allclose(stiff_mat.dot(disp)/stiff_mat.max(),
                       rhs_vec/stiff_mat.max()):
        print("The system is not in equilibrium!")
    return bc_array, disp


def postprocessing(nodes, mats, els, bc_array, disp):
    """
    Compute the nodes displacements, strains and stresses.
    
    Get from: https://github.com/AppliedMechanics-EAFIT/SolidsPy/blob/master/solidspy/solids_GUI.py
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    mats : ndarray
        Array with models materials
    els : ndarray
        Array with models elements
    IBC : ndarray 
        Boundary conditions array
    UG : ndarray 
        Static solve.
        
    Returns
    -------
    disp_complete : ndarray 
        Displacements at elements.
    strain_nodes : ndarray 
        Strains at elements.
    stress_nodes : ndarray 
        Stresses at elements.
    """   
    
    disp_complete = pos.complete_disp(bc_array, nodes, disp)
    strain_nodes, stress_nodes = None, None
    strain_nodes, stress_nodes = pos.strain_nodes(nodes, els, mats, disp_complete)
    
    return disp_complete, strain_nodes, stress_nodes

#%% Adjacency matrix

def adjacency_nodes(nodes, els):
    """
    Create an adjacency matrix for the elements connected to each node.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes.
    els : ndarray
        Array with models elements.
        
    Returns
    -------
    adj_nodes : ndarray, nodes.shape[0]
        Adjacency elements for each node.
    """
    adj_nodes = []
    for n in range(nodes.shape[0]):
        adj_els = np.argwhere(els[:, -4:] == n)[:,0]
        adj_nodes.append(adj_els)
    return adj_nodes

def center_els(nodes, els):
    """
    Calculate the center of each element.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes.
    els : ndarray
        Array with models elements.
        
    Returns
    -------
    centers : ndarray, nodes.shape[0]
        Adjacency elements for each node.
    """
    centers = []
    for el in els:
        n = nodes[el[-4:], 1:3]
        center = np.array([n[1,0] + (n[0,0] - n[1,0])/2, n[2,1] + (n[0,1] - n[2,1])/2])
        centers.append(center)
    centers = np.array(centers)

    return centers

def sensitivity_nodes(nodes, adj, centers, sensi_els):
    n_sensi = []
    for n in nodes:
        connected_els = adj[int(n[0])]
        if len(connected_els) > 1:
            coord = n[1:3]
            r_ij = np.array([np.linalg.norm(c) for c in coord - centers[connected_els]]) # We can remove this line and just use a constant because the distance is always the same
            w_i = 1/(len(connected_els) - 1) * (1 - r_ij/r_ij.sum())
            sensi = (w_i * sensi_els[connected_els]).sum(axis=0)
        else:
            sensi = sensi_els[connected_els[0]]
        n_sensi.append(sensi)
    return n_sensi

def sensitivity_els(sensi_nodes, centers, r_min):
    sensi_els = []
    for i, c in enumerate(centers):
        delta = c-centers
        r_ij = np.array([np.linalg.norm(d) for d in delta])

    return sensi_els
    
#%%
centers = center_els(nodes, els)
adj_nodes = adjacency_nodes(nodes, els)

sensi_nodes = sensitivity_nodes(nodes, adj, centers, sensi_I)
r_min = 0.2
sensi_els = sensitivity_els(sensi_nodes, centers, r_min)
# %%
length = 10
height = 24
nx = 20
ny= 40
nodes, mats, els, loads, BC = beam_2(L=length, H=height, nx=nx, ny=ny)
elsI,nodesI = np.copy(els), np.copy(nodes)
mask_del = np.ones(els.shape[0], dtype=bool)

# %%
IBC, UG = preprocessing(nodes, mats, els, loads)
UCI, E_nodesI, S_nodesI = postprocessing(nodes, mats, els, IBC, UG)
sensi_I = sensi_el(nodes, mats, els, UCI)

# %%
niter = 50
RR = 0.01
ER = 0.005
r_min = 0.01

MASK_DEL = np.zeros(els.shape[0], dtype=bool)
sensi_I = sensi_el(nodes, mats, els, UCI)

V_opt = volume(els, length, height, nx, ny) * 0.50
ELS = None

for _ in range(niter):
    if not is_equilibrium(nodes, mats, els, loads) or volume(els, length, height, nx, ny) < V_opt: 
        print('Is not equilibrium')
        break
    
    IBC, UG = preprocessing(nodes, mats, els, loads)
    UC, E_nodes, S_nodes = postprocessing(nodes, mats, els, IBC, UG)

    sensi_number = sensi_el(nodes, mats, els, UC)
    # Filter of nodes 
    # Filter of elements
    mask_del = sensi_number < RR
    mask_els = protect_els(els, loads, BC)
    mask_del *= mask_els
    ELS = els
    
    els = np.delete(els, mask_del, 0)
    del_node(nodes, els)
    RR += ER
print(RR)

# %%
pos.fields_plot(elsI, nodes, UCI, E_nodes=E_nodesI, S_nodes=S_nodesI)

# %%
pos.fields_plot(ELS, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes)
