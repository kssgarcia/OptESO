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
    print("Number of elements: {}".format(els.shape[0]))

    # System assembly
    stiff_mat, _ = ass.assembler(els, mats, nodes[:, :3], neq, assem_op)
    rhs_vec = ass.loadasem(loads, bc_array, neq)

    # System solution
    disp = sol.static_sol(stiff_mat, rhs_vec)
    if not np.allclose(stiff_mat.dot(disp)/stiff_mat.max(),
                       rhs_vec/stiff_mat.max()):
        print("The system is not in equilibrium!")
    return bc_array, disp, rhs_vec


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
    for n in nodes[:, 0]:
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

def sensitivity_nodes(nodes, adj_nodes, centers, sensi_els):
    sensi_nodes = []
    for n in nodes:
        connected_els = adj_nodes[int(n[0])]
        if connected_els.shape[0] > 1:
            delta = centers[connected_els] - n[1:3]
            r_ij = np.linalg.norm(delta, axis=1) # We can remove this line and just use a constant because the distance is always the same
            w_i = 1/(connected_els.shape[0] - 1) * (1 - r_ij/r_ij.sum())
            sensi = (w_i * sensi_els[connected_els]).sum(axis=0)
        else:
            sensi = sensi_els[connected_els[0]]
        sensi_nodes.append(sensi)
    sensi_nodes = np.array(sensi_nodes)

    return sensi_nodes

def sensitivity_els(nodes, sensi_nodes, centers, r_min):
    sensi_els = []
    for i, c in enumerate(centers):
        delta = nodes[:,1:3]-c
        r_ij = np.linalg.norm(delta, axis=1)
        omega_i = (r_ij < r_min)
        w = 1/(omega_i.sum() - 1) * (1 - r_ij[omega_i]/r_ij[omega_i].sum())
        sensi_els.append((w*sensi_nodes[omega_i]).sum()/w.sum())
        
    sensi_els = np.array(sensi_els)
    sensi_els = sensi_els/sensi_els.max()

    return sensi_els

def cal_error(C, N, k):
    for n in range(N):
        t_1 = C[k+1-i]
        t_2 = C[k+1-i-N]
    error = (t_1 - t_2)/t_1
    return error

# %%
length = 8
height = 5
nx = 40
ny= 30
nodes, mats, els, loads, BC = beam_2(L=length, H=height, nx=nx, ny=ny)

elsI,nodesI = np.copy(els), np.copy(nodes)
IBC, UG, rhs_vec = preprocessing(nodes, mats, els, loads)
UCI, E_nodesI, S_nodesI = postprocessing(nodes, mats, els, IBC, UG)


r_min = np.linalg.norm(nodes[0,1:3] - nodes[1,1:3])
adj_nodes = adjacency_nodes(nodes, els)
centers = center_els(nodes, els)
mask = np.ones(els.shape[0], dtype=bool)

sensi_e = sensi_el(nodes, mats, els, mask, UCI)
sensi_nodes = sensitivity_nodes(nodes, adj_nodes, centers, sensi_e) #3.4
sensi_number = sensitivity_els(nodes, sensi_nodes, centers, r_min) #3.6

# %%
niter = 50
ER = 0.05 # Evolutinary volume ratio
AR_max = 0.01
t = 0.0001

r_min = np.linalg.norm(nodes[0,1:3] - nodes[1,1:3]) * 1.5
adj_nodes = adjacency_nodes(nodes, els)
centers = center_els(nodes, els)

Vi = volume(els, length, height, nx, ny)
V_opt = Vi.sum() * 0.50
ELS = None
mask = np.ones(els.shape[0], dtype=bool)
sensi_I = None

C_h = []
error = 1000
for i in range(niter):
    els_del = els[mask].copy()
    V = Vi[mask].sum()

    if not is_equilibrium(nodes, mats, els_del, loads): 
        print('Is not equilibrium')
        break
    
    ELS = els_del
    # FEW
    IBC, UG, rhs_vec = preprocessing(nodes, mats, els_del, loads)
    UC, E_nodes, S_nodes = postprocessing(nodes, mats, els_del, IBC, UG)

    # Sensitivity filter
    sensi_e = sensi_el(nodes, mats, els, mask, UC)
    sensi_nodes = sensitivity_nodes(nodes, adj_nodes, centers, sensi_e) #3.4
    sensi_number = sensitivity_els(nodes, sensi_nodes, centers, r_min) #3.6


    if i > 0: sensi_number = (sensi_number + sensi_I)/2 # 3.8
    sesni_number = sensi_number/sensi_number.max()

    V_r = False
    if V <= V_opt:
        els_k = els_del.shape[0]
        V_r = True
        break
    else:
        V_k = V * (1 + ER) if V < V_opt else V * (1 - ER)

    # Remove/add threshold
    sensi_sort = np.sort(sensi_number)[::-1]
    els_k = els_del.shape[0]*V_k/V
    alpha_del = sensi_sort[int(els_k)]

    mask = sensi_number > alpha_del
    mask_els = protect_els(els[np.invert(mask)], els.shape[0], loads, BC)
    mask = np.bitwise_or(mask, mask_els)
    del_node(nodes, els[mask], loads, BC)

    els_test = els[mask]

    C = 0.5*rhs_vec.T@UG
    C_h.append(C)

    if i > 10: error = (sum(C_h[-5:]) - sum(C_h[-10:-5]))/sum(C_h[-5:])

    if error <= t and V_r == True:
        print("convergence")
        break

    sensi_I = sensi_number.copy()

plt.figure()
plt.title('Nodes')
plt.scatter(nodes[:,1], nodes[:,2], c=sensi_nodes)
plt.show()
plt.figure()
plt.title('Elements')
plt.scatter(centers[:,0], centers[:,1], c=sensi_e)
plt.show()
plt.figure()
plt.title('Number')
plt.scatter(centers[:,0], centers[:,1], c=sensi_number)
plt.show()

# %%
pos.fields_plot(elsI, nodes, UCI, E_nodes=E_nodesI, S_nodes=S_nodesI)

# %%
pos.fields_plot(ELS, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes)
