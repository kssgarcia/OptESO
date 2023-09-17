import numpy as np
import solidspy.assemutil as ass    
import solidspy.uelutil as uel 
from scipy.sparse.linalg import spsolve
import solidspy.postprocesor as pos 

from scipy.sparse import coo_matrix

def sparse_assem(elements, mats, nodes, neq, assem_op, uel=None):
    """
    Assembles the global stiffness matrix
    using a sparse storing scheme

    The scheme used to assemble is COOrdinate list (COO), and
    it converted to Compressed Sparse Row (CSR) afterward
    for the solution phase [1]_.

    Parameters
    ----------
    elements : ndarray (int)
      Array with the number for the nodes in each element.
    mats    : ndarray (float)
      Array with the material profiles.
    nodes    : ndarray (float)
      Array with the nodal numbers and coordinates.
    assem_op : ndarray (int)
      Assembly operator.
    neq : int
      Number of active equations in the system.
    uel : callable function (optional)
      Python function that returns the local stiffness matrix.

    Returns
    -------
    kglob : sparse matrix (float)
      Array with the global stiffness matrix in a sparse
      Compressed Sparse Row (CSR) format.

    References
    ----------
    .. [1] Sparse matrix. (2017, March 8). In Wikipedia,
        The Free Encyclopedia.
        https://en.wikipedia.org/wiki/Sparse_matrix

    """

    rows = []
    cols = []
    stiff_vals = []
    nels = elements.shape[0]
    kloc, _ = ass.retriever(elements, mats, nodes, -1, uel=uel)
    for ele in range(nels):
        kloc_ = kloc * mats[elements[ele, 0], 2]
        ndof = kloc_.shape[0]
        dme = assem_op[ele, :ndof]
        for row in range(ndof):
            glob_row = dme[row]
            if glob_row != -1:
                for col in range(ndof):
                    glob_col = dme[col]
                    if glob_col != -1:
                        rows.append(glob_row)
                        cols.append(glob_col)
                        stiff_vals.append(kloc_[row, col])

    stiff = coo_matrix((stiff_vals, (rows, cols)), shape=(neq, neq)).tocsr()
    return stiff

def DME(cons, elements, ndof_node=2, ndof_el_max=18, ndof_el=None):
    """Create assembly array operator

    Count active equations, create boundary conditions array ``bc_array``
    and the assembly operator ``assem_op``.

    Parameters
    ----------
    cons : ndarray.
      Array with constraints for each degree of freedom in each node.
    elements : ndarray
      Array with the number for the nodes in each element.
    ndof_node : int, optional
      Number of degrees of freedom per node. By default it is 2.
    ndof_el_max : int, optional
      Number of maximum degrees of freedom per element. By default it is
      18.
    ndof_el : callable, optional
      Function that return number of degrees of freedom for elements. It
      is needed for user elements.

    Returns
    -------
    assem_op : ndarray (int)
      Assembly operator.
    bc_array : ndarray (int)
      Boundary conditions array.
    neq : int
      Number of active equations in the system.

    """
    nels = elements.shape[0]
    assem_op = np.zeros([nels, ndof_el_max], dtype=np.integer)
    neq, bc_array = ass.eqcounter(cons, ndof_node=ndof_node)
    for ele in range(nels):
        iet = elements[ele, 1]
        ndof = iet
        assem_op[ele, :ndof] = bc_array[elements[ele, 3:]].flatten()
    return assem_op, bc_array, neq

def sensi_el(nodes, mats, els, UC, kloc):
    """
    Calculate the sensitivity number for each element.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    mats : ndarray
        Array with models materials
    els : ndarray
        Array with models elements
    UC : ndarray
        Displacements at nodes

    Returns
    -------
    sensi_number : ndarray
        Sensitivity number for each element.
    """   
    sensi_number = []
    for el in range(len(els)):
        node_el = els[el, -8:]
        U_el = UC[node_el]
        U_el = np.reshape(U_el, (24,1))
        a_i = 0.5 * U_el.T.dot(kloc.dot(U_el))[0,0]
        sensi_number.append(a_i)
    sensi_number = np.array(sensi_number)
    sensi_number = sensi_number/sensi_number.max()

    return sensi_number

def del_node(nodes, els, loads, BC):
    """
    Retricts nodes dof that aren't been used and free up the nodes that are in use.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    els : ndarray
        Array with models elements
    loads : ndarray
        Array with models loads
    BC : ndarray 
        Boundary conditions nodes

    Returns
    -------
    """   
    protect_nodes = np.hstack((loads, BC)).astype(int)
    for n in nodes[:,0]:
        if n not in els[:, -8:]:
            nodes[int(n), -3:] = -1
        elif n not in protect_nodes and n in els[:, -8:]:
            nodes[int(n), -3:] = 0

def protect_els(els, loads, BC):
    """
    Compute an mask array with the elements that don't must be deleted.
    
    Parameters
    ----------
    els : ndarray
        Array with models elements
    nels : ndarray
        Number of elements
    loads : ndarray
    BC : ndarray 
        Boundary conditions nodes
        
    Returns
    -------
    mask_els : ndarray 
        Array with the elements that don't must be deleted.
    """   
    mask_els = np.zeros(els.shape[0], dtype=bool)
    protect_nodes = np.hstack((loads, BC)).astype(int)
    protect_index = None
    for p in protect_nodes:
        protect_index = np.argwhere(els[:, -8:] == p)[:,0]
        mask_els[protect_index] = True
        
    return mask_els

def strain_els(els, E_nodes):
    """
    Compute the elements strains and stresses.
    
    Get from: https://github.com/AppliedMechanics-EAFIT/SolidsPy/blob/master/solidspy/solids_GUI.py
    
    Parameters
    ----------
    els : ndarray
        Array with models elements
    E_nodes : ndarray
        Strains at nodes.
        
    Returns
    -------
    E_els : ndarray (nnodes, 3)
        Strains at elements.
    """   
    
    E_els = []
    for el in els:
        strain_nodes = np.take(E_nodes, list(el[3:]), 0)
        strain_nodes = np.linalg.norm(strain_nodes, axis=1)
        strain_elemt = strain_nodes.sum(0) / strain_nodes.shape[0]
        E_els.append(strain_elemt)

    E_els = np.array(E_els)
    
    return E_els

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
    assem_op, bc_array, neq = DME(nodes[:, -3:], els, ndof_node=3, ndof_el_max=24)
    # System assembly
    stiff_mat = sparse_assem(els, mats, nodes[:, :4], neq, assem_op, uel=uel.elast_hex8)
    rhs_vec = ass.loadasem(loads, bc_array, neq, ndof_node=3)

    disp = spsolve(stiff_mat, rhs_vec)
    UC = pos.complete_disp(bc_array, nodes, disp, ndof_node=3)
    if not np.allclose(stiff_mat.dot(disp)/stiff_mat.max(), rhs_vec/stiff_mat.max()):
        equil = False

    return equil