import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from scipy.spatial.distance import cdist


def sparse_assem(elements, mats, neq, assem_op, kloc):
    """
    Assembles the global stiffness matrix
    using a sparse storing scheme

    Parameters
    ----------
    elements : ndarray (int)
      Array with the number for the nodes in each element.
    mats    : ndarray (float)
      Array with the material profiles.
    neq : int
      Number of active equations in the system.
    assem_op : ndarray (int)
      Assembly operator.
    kloc : ndarray 
      Stiffness matrix of a single element

    Returns
    -------
    stiff : sparse matrix (float)
      Array with the global stiffness matrix in a sparse
      Compressed Sparse Row (CSR) format.

    """
    rows = []
    cols = []
    stiff_vals = []
    nels = elements.shape[0]
    for ele in range(nels):
        kloc_ = kloc * mats[elements[ele, 0], 2]
        ndof = kloc.shape[0]
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

def sensi_el(els, UC, kloc):
    """
    Calculate the sensitivity number for each element.
    
    Parameters
    ----------
    els : ndarray
        Array with models elements
    UC : ndarray
        Displacements at nodes
    kloc : ndarray 
      Stiffness matrix of a single element

    Returns
    -------
    sensi_number : ndarray
        Sensitivity number for each element.
    """   
    sensi_number = []
    for el in range(len(els)):
        node_el = els[el, -4:]
        U_el = UC[node_el]
        U_el = np.reshape(U_el, (8,1))
        x_i = -U_el.T.dot(kloc.dot(U_el))[0,0]
        sensi_number.append(x_i)
    sensi_number = np.array(sensi_number)

    return sensi_number

def volume(length, height, nx, ny):
    """
    Volume calculation.
    
    Parameters
    ----------
    length : ndarray
        Length of the beam.
    height : ndarray
        Height of the beam.
    nx : float
        Number of elements in x direction.
    ny : float
        Number of elements in y direction.

    Return 
    ----------
    V: float
        Volume of a single element.
    """

    dy = length / nx
    dx = height / ny
    V = dy*dx

    return V

def x_star(lamb, q_o, L_j, v_j, alpha, x_max):
    """
    Calculates the optimal x value (x_star) for a given lambda.

    Parameters
    ----------
    lamb : float
        Lambda value
    q_o : float
        Initial value of q
    L_j : float
        Lower bound for x
    v_j : float
        Coefficient for the x term
    alpha : float
        Minimum possible value for x
    x_max : float
        Maximum possible value for x

    Returns
    -------
    x_star: float
        Optimal x value (x_star)
    """
    x_t = L_j + np.sqrt(q_o / (lamb * v_j))
    x_star = np.clip(x_t, alpha, x_max)
    return x_star

def objective_function(lamb, r_o, v_max, q_o, L_j, v_j, alpha, x_max):
    """
    Calculates the value of the objective function for a given lambda.

    Parameters
    ----------
    lamb : float
        Lambda value
    r_o : float
        Initial value of r
    v_max : float
        Maximum possible value for v
    q_o : float
        Initial value of q
    L_j : float
        Lower bound for x
    v_j : float
        Coefficient for the x term
    alpha : float
        Minimum possible value for x
    x_max : float
        Maximum possible value for x

    Returns
    -------
    obj : float
        Value of the objective function
    """
    x_star_value = x_star(lamb, q_o, L_j, v_j, alpha, x_max)
    obj = -(r_o - lamb*v_max + (q_o/(x_star_value-L_j) + lamb*v_j*x_star_value).sum())
    return obj

def gradient(lamb, v_max, q_o, L_j, v_j, alpha, x_max):
    """
    Calculates the gradient of the objective function for a given lambda.

    Parameters
    ----------
    lamb : float
        Lambda value
    v_max : float
        Maximum possible value for v
    q_o : float
        Initial value of q
    L_j : float
        Lower bound for x
    v_j : float
        Coefficient for the x term
    alpha : float
        Minimum possible value for x
    x_max : float
        Maximum possible value for x

    Returns
    -------
    grad : float
        Gradient of the objective function
    """
    x_star_value = x_star(lamb, q_o, L_j, v_j, alpha, x_max)
    grad = (v_j * x_star_value).sum() - v_max
    return grad