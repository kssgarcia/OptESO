# %%
import matplotlib.pyplot as plt # Package for plotting
from matplotlib import colors 
import numpy as np # Package for scientific computing
import solidspy.assemutil as ass # Solidspy 1.1.0
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import solidspy.postprocesor as pos 

from Utils.beams import * # Functions for mesh generation
from Utils.MMA_utils import * # Fucntions for FEM analysis and postprocessing
# Solidspy 1.1.0
np.seterr(divide='ignore', invalid='ignore')

def optimization():
    length = 160
    height = 40
    nx = 200
    ny= 40
    dirs = np.array([[0,-1], [0,1], [1,0]])
    positions = np.array([[61,30], [1,30], [30, 1]])
    nodes, mats, els, loads = beam(L=length, H=height, nx=nx, ny=ny, dirs=dirs, positions=positions)

    E = mats[0,0] # Young modulus
    nu = mats[0,1] # Poisson ratio
    k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8]) # Coefficients
    kloc = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]], 
    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]]); # Local stiffness matrix
    assem_op, bc_array, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8) 

    change = 10
    step = 0.002
    volfrac = 0.7

    v_j = np.ones((els.shape[0])) * volume(length, height, nx, ny)
    v_max = volume(length, height, nx, ny) * int(els.shape[0] * volfrac)

    #x_j = mats[:,2]
    x_j = np.ones(ny*nx) * 0.6 # Initialize the sensitivity
    x_min = np.ones(ny*nx)*1e-5 # Minimum young modulus of the material
    x_max = np.ones(ny*nx) # Maximum young modulus of the material
    lamb = 0
    penal = 3 # Penalization factor
    mu = 0.8
    s = 0.1 # S init
    L_j = x_j - s*(x_max-x_min)
    alpha = np.maximum(x_min, L_j + mu*(x_j - L_j))
    x_j_prev = None
    x_j_after_prev = None

    compliance = []
    restriction = []
    #x_h = []
    for iter in range(1):

        # Check convergence
        if change < 0.01:
            print('Convergence reached')
            break

        mats[:,2] = x_min+x_j**penal*(x_max-x_min) # Update the Young modulus

        # System assembly
        stiff_mat = sparse_assem(els, mats, neq, assem_op, kloc)
        rhs_vec = ass.loadasem(loads, bc_array, neq)

        # System solution
        disp = spsolve(stiff_mat, rhs_vec)
        UC = pos.complete_disp(bc_array, nodes, disp)

        # Compliance
        g = rhs_vec.T@disp
        d_g = sensi_el(els, UC, kloc)

        # Compliance data
        compliance.append(g)
        restriction.append(lamb*(v_j.dot(x_j) - v_max))
        #x_h.append(x_j)

        # Filtering the design variable to be in a feasible interval
        x_j_after_prev = x_j_prev
        x_j_prev = x_j.copy()

        # Calculate the asymptotes
        q_o = -(x_j - L_j)**2 * d_g
        r_o = g + ((x_j - L_j) * d_g).sum()

        for _ in range(10000):
            grad = gradient(lamb, r_o, v_max, q_o, L_j, v_j, alpha, x_max)
            lamb += step * grad
            if abs(grad) < 0.001:
                break
        print(lamb)

        dual_problem = minimize(objective_function, 
                                lamb, bounds=[(0, None)],
                                args=(r_o, v_max, q_o, L_j, v_j, alpha, x_max))
        #dual_problem = minimize(objective_function, 1, bounds=[(0, None)], args=(r_o, v_max, q_o, L_j, v_j, alpha, x_max), jac=gradient)
        lamb = dual_problem.x[0]
        print(lamb)

        # Calculate the design variable for the next iteration
        x_j = x_star(lamb, L_j, q_o, v_j, alpha, x_max)

        change = np.linalg.norm(x_j-x_j_prev)
        print(change, '----')

        # Change the distance of the asymptotes
        if iter>1:
            sign = (x_j-x_j_prev)*(x_j_prev-x_j_after_prev)
            if np.all(sign >= 0) or np.all(sign < 0):
                s = 1.01
            else:
                s = 0.01
            L_j = x_j - s*(x_j_prev-L_j)
            alpha = np.maximum(x_min, L_j + mu*(x_j - L_j))
    
    # Generate a range of lambda values for the plot
    lamb_values = np.linspace(0, 10000, 1000)

    # Calculate the objective function values and gradient values for the lambda range
    objective_values = [-objective_function(lamb, r_o, v_max, q_o, L_j, v_j, alpha, x_max) for lamb in lamb_values]
    gradient_values = [gradient(lamb, r_o, v_max, q_o, L_j, v_j, alpha, x_max) for lamb in lamb_values]

    # Plot the objective function
    plt.figure(figsize=(10, 6))
    plt.plot(lamb_values, objective_values, label='Objective Function', color='b')
    plt.xlabel('Lambda')
    plt.ylabel('Objective Function Value')
    plt.title('Objective Function Plot')
    plt.legend()

    # Plot the gradient
    plt.figure(figsize=(10, 6))
    plt.plot(lamb_values, gradient_values, label='Gradient', color='r')
    plt.xlabel('Lambda')
    plt.ylabel('Gradient Value')
    plt.title('Gradient Plot')
    plt.legend()
    plt.show()

    # Plot the compliance
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0,100,len(compliance)), compliance, label='Compliance', color='g')
    plt.xlabel('Iteration')
    plt.ylabel('Compliance')
    plt.title('Compliance Plot')
    plt.legend()
    plt.show()

    # Plot the restriction
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0,100,len(restriction)), restriction, label='Compliance', color='g')
    plt.xlabel('Iteration')
    plt.ylabel('Restriction')
    plt.title('Restriction Plot')
    plt.legend()
    plt.show()

    # %%

    lam = np.linspace(0,1000, 1000)
    q = lambda lam: r_o - lam*v_max + (q_o/(x_j - L_j) + lam*v_j*x_j).sum()
    result = list(map(q, lam))

    plt.figure()
    plt.plot(lam, result)
    plt.show()

    # %%
    E_nodes, S_nodes = pos.strain_nodes(nodes, els, mats[:,:2], UC)
    pos.fields_plot(els, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes)

    # %% Plot
    plt.ion() 
    fig,ax = plt.subplots()
    im = ax.imshow(-x_j.reshape(ny, nx), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
    fig.show()

if __name__ == "__main__":
    optimization()