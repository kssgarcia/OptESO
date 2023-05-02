# %%
import matplotlib.pyplot as plt
import numpy as np

from ESO_utils import *
from beams import *

# Solidspy 1.1.0
import solidspy.postprocesor as pos 
np.seterr(divide='ignore', invalid='ignore')

# %%
length = 20
height = 10
nx = 50
ny= 20
nodes, mats, els, loads, BC = beam(L=length, H=height, nx=nx, ny=ny, n=2)
elsI,nodesI = np.copy(els), np.copy(nodes)

# %%
IBC, UG = preprocessing(nodes, mats, els, loads)
UCI, E_nodesI, S_nodesI = postprocessing(nodes, mats, els, IBC, UG)

# %%
niter = 200
RR = 0.01
ER = 0.005
V_opt = volume(els, length, height, nx, ny) * 0.60

ELS = None
for _ in range(niter):

    # Check equilibrium
    if not is_equilibrium(nodes, mats, els, loads) or volume(els, length, height, nx, ny) < V_opt: break
    ELS = els
    
    # FEW analysis
    IBC, UG = preprocessing(nodes, mats, els, loads)
    UC, E_nodes, S_nodes = postprocessing(nodes, mats, els, IBC, UG)
    E_els, S_els = strain_els(els, E_nodes, S_nodes)
    vons = np.sqrt(S_els[:,0]**2 - (S_els[:,0]*S_els[:,1]) + S_els[:,1]**2 + 3*S_els[:,2]**2)

    # Remove/add elements
    RR_el = vons/vons.max()
    mask_del = RR_el < RR
    mask_els = protect_els(els, loads, BC)
    mask_del *= mask_els
    els = np.delete(els, mask_del, 0)
    del_node(nodes, els)

    RR += ER
print(RR)
# %%
pos.fields_plot(elsI, nodes, UCI, E_nodes=E_nodesI, S_nodes=S_nodesI)

# %%
pos.fields_plot(ELS, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes)

# %%
fill_plot = np.ones(E_nodes.shape[0])
plt.figure()
tri = pos.mesh2tri(nodes, ELS)
plt.tricontourf(tri, fill_plot, cmap='binary');
