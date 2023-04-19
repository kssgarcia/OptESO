# %%
import matplotlib.pyplot as plt
import numpy as np
from opt_beam import *
from beams import *

import solidspy.assemutil as ass    
import solidspy.solutil as sol      
import solidspy.postprocesor as pos 
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
    DME, IBC, neq = ass.DME(nodes, els)
    KG = ass.assembler(els, mats, nodes, neq, DME)
    RHSG = ass.loadasem(loads, IBC, neq)
    UG = sol.static_sol(KG, RHSG)
    
    if not(np.allclose(KG.dot(UG)/KG.max(), RHSG/KG.max())):
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
    IBC : ndarray 
        Boundary conditions array
    UG : ndarray 
        Static displacement solve.
    """   

    # Pre-processing
    DME, IBC, neq = ass.DME(nodes, els)
    print("Number of elements: {}".format(els.shape[0]))

    # System assembly
    KG = ass.assembler(els, mats, nodes, neq, DME)
    RHSG = ass.loadasem(loads, IBC, neq)

    # System solution
    UG = sol.static_sol(KG, RHSG)
    if not(np.allclose(KG.dot(UG)/KG.max(), RHSG/KG.max())):
        print("The system is not in equilibrium!")
    return IBC, UG

def postprocessing(nodes, mats, els, IBC, UG):
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
    UC : ndarray 
        Displacements at elements.
    E_nodes : ndarray 
        Strains at elements.
    S_nodes : ndarray 
        Stresses at elements.
    """   
    
    UC = pos.complete_disp(IBC, nodes, UG)
    E_nodes, S_nodes = None, None
    E_nodes, S_nodes = pos.strain_nodes(nodes , els, mats, UC)
    
    return UC, E_nodes, S_nodes

# %%
length = 10
height = 24
nx = 20
ny= 40
nodes, mats, els, loads, BC = beam_2(L=length, H=height, nx=nx, ny=ny)
elsI,nodesI = np.copy(els), np.copy(nodes)

# %%
IBC, UG = preprocessing(nodes, mats, els, loads)
UCI, E_nodesI, S_nodesI = postprocessing(nodes, mats, els, IBC, UG)

# %%
niter = 40
RR = 0.01
ER = 0.01
V_opt = volume(els, length, height, nx, ny) * 0.50

ELS = None
for _ in range(niter):

    if not is_equilibrium(nodes, mats, els, loads) or volume(els, length, height, nx, ny) < V_opt: break
    
    IBC, UG = preprocessing(nodes, mats, els, loads)
    UC, E_nodes, S_nodes = postprocessing(nodes, mats, els, IBC, UG)
    E_els, S_els = strain_els(els, E_nodes, S_nodes)
    vons = np.sqrt(S_els[:,0]**2 - (S_els[:,0]*S_els[:,1]) + S_els[:,1]**2 + 3*S_els[:,2]**2)
    RR_el = vons/vons.max()
    mask_del = RR_el < RR
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


