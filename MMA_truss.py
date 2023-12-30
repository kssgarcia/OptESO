#%% 
import numpy as np
from Utils.MMA_truss_utils import *
import matplotlib.pyplot as plt
import solidspy.assemutil as ass
from scipy.optimize import minimize

def optimization_MMA_truss():
    length = 6
    height = 3
    nx = 11
    ny = 9
    nodes, els, nels, x, y = grid_truss(length, height, nx, ny)

    mask_loads = (x==length/2) & (y==0)
    loads = np.zeros((mask_loads.sum(), 3))
    loads[:, 0] = nodes[mask_loads, 0]
    loads[:, 2] = -1.0

    maskBC_1 = (x == -length/2) & (y==height/2)
    maskBC_2 = (x == -length/2) & (y==-height/2)
    nodes[maskBC_1, 3] = -1
    nodes[maskBC_1, 4] = -1
    nodes[maskBC_2, 3] = -1
    nodes[maskBC_2, 4] = -1
    BC = nodes[(nodes[:,-2] == -1) & (nodes[:,-1] == -1), 0]

    #areas = np.random.uniform(low=0.1, high=1.0, size=nels)
    areas = np.ones(nels)*0.8
    mats = np.ones((nels, 3))
    mats[:, 1] = 0.28
    mats[:, 2] = areas

    change = 10
    step = 0.002
    volfrac = 0.6
    v_j = np.ones((els.shape[0])) * lengths(els, nodes)
    v_max = np.sum(mats[:,2] * lengths(els, nodes)) * volfrac

    #x_j = np.ones(els.shape[0])*0.2
    x_j = mats[:,2]
    x_min = np.ones(els.shape[0])*1e-6 
    x_max = np.ones(els.shape[0]) 
    lamb = 0
    penal = 2 # Penalization factor
    mu = 0.8
    s = 0.1 # S init
    L_j = x_j - s*(x_max-x_min)
    alpha = np.maximum(x_min, L_j + mu*(x_j - L_j))
    x_j_prev = None
    x_j_after_prev = None

    for i in range(1):

        if not is_equilibrium(nodes, els, mats, loads) or change < 0.01: 
            print('Volume reach.')
            break

        mats[:,2] = x_min+x_j**penal*(x_max-x_min) # Update the Young modulus

        # FEM
        disp, UC, rhs_vec = fem_sol(nodes, els, mats, loads)
        stress_nodes = pos.stress_truss(nodes, els, mats, UC)

        # Compliance
        g = rhs_vec.T@disp
        d_g = sensi_el(els, mats, nodes, UC)

        # Calculate the asymptotes
        q_o = -(x_j - L_j)**2 * d_g
        r_o = g + ((x_j - L_j) * d_g).sum()

        x_j = x_star(lamb, L_j, q_o, v_j, alpha, x_max)

        # Filtering the design variable to be in a feasible interval
        x_j_after_prev = x_j_prev
        x_j_prev = x_j.copy()

        # Getting the maximum of the objective function
        for _ in range(100000):
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

    # Generate a range of lambda values for the plot
    lamb_values = np.linspace(0, 100, 100)

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

    plt.figure(figsize=(12, 4))
    plt.title('Original truss')
    plot_truss(nodes, els, mats, stress_nodes)
    plt.show()

if __name__ == "__main__":
    optimization_MMA_truss()