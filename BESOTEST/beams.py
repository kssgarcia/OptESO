import numpy as np
import solidspy.preprocesor as pre

def beam_1(L=10, H=10, F=-1000000, E=206.8e9, v=0.28, nx=20, ny=20):
    """
    Make the mesh for a cuadrilateral model.

    Parameters
    ----------
    L : float (optional)
        Beam's lenght
    H : float (optional)
        Beam's height
    F : float (optional)
        Vertical force.
    E : string (optional)
        Young module
    v : string (optional)
        Poisson ratio
    nx : int (optional)
        Number of element in x direction
    ny : int (optional)
        Number of element in y direction

    Returns
    -------
    nodes : ndarray
        Nodes array
    mats : ndarray (1, 2)
        Mats array
    els : ndarray
        Elements array
    loads : ndarray
        Loads array
    BC : ndarray
        Boundary conditions nodes

    """
    mats = np.array([[E,v]])
    x, y, els = pre.rect_grid(L, H, nx, ny)
    nodes = np.zeros(((nx + 1)*(ny + 1), 5))
    nodes[:, 0] = range((nx + 1)*(ny + 1))
    nodes[:, 1] = x
    nodes[:, 2] = y
    nodes[x==L/2, 3] = -1
    nodes[x==L/2, 4] = -1    
    mask_loads = (x == -L/2) & (y < H/6) & (y > -H/6)
    loads_nodes = nodes[mask_loads, 0]
    loads = np.zeros((len(loads_nodes), 3))
    loads[:, 0] = loads_nodes
    loads[:, 2] = F
    BC = nodes[mask_loads, 0]
    return nodes, mats, els, loads, BC

def beam_2(L=10, H=10, F=-1000000, E=206.8e9, v=0.28, nx=20, ny=20):
    """
    Make the mesh for a cuadrilateral model.

    Parameters
    ----------
    L : float (optional)
        Beam's lenght
    H : float (optional)
        Beam's height
    F : float (optional)
        Vertical force.
    E : string (optional)
        Young module
    v : string (optional)
        Poisson ratio
    nx : int (optional)
        Number of element in x direction
    ny : int (optional)
        Number of element in y direction

    Returns
    -------
    nodes : ndarray
        Nodes array
    mats : ndarray (1, 2)
        Mats array
    els : ndarray
        Elements array
    loads : ndarray
        Loads array
    BC : ndarray
        Boundary conditions nodes

    """
    mats = np.array([[E,v]])
    x, y, els = pre.rect_grid(L, H, nx, ny)
    nodes = np.zeros(((nx + 1)*(ny + 1), 5))
    nodes[:, 0] = range((nx + 1)*(ny + 1))
    nodes[:, 1] = x
    nodes[:, 2] = y
    mask_1 = (x == L/2) & (y < H/2.5)
    mask_2 = (x == L/2) & (y > -H/2.5)
    nodes[mask_1, 3] = -1
    nodes[mask_1, 4] = -1
    nodes[mask_2, 3] = -1
    nodes[mask_2, 4] = -1

    mask_loads = (x == -L/2) & (y < H/6) & (y > -H/6)
    loads_nodes = nodes[mask_loads, 0]
    loads = np.zeros((len(loads_nodes), 3))
    loads[:, 0] = loads_nodes
    loads[:, 2] = F
    #look here
    BC = nodes[mask_loads, 0]
    return nodes, mats, els, loads, BC