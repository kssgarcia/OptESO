#%% 
import numpy as np
from Utils.MMA_truss_utils import *
import matplotlib.pyplot as plt
import solidspy.assemutil as ass
from scipy.optimize import minimize

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

areas = np.random.uniform(low=0.1, high=1.0, size=nels)
#areas = np.ones(nels)
mats = np.ones((nels, 3))
mats[:, 1] = 0.28
mats[:, 2] = areas

change = 10
step = 0.002
volfrac = 0.7
v_j = np.ones((els.shape[0])) * lengths(els, nodes)
v_max = np.sum(mats[:,1] * lengths(els, nodes)) * volfrac

x_j = mats[:,1] # Initialize the sensitivity
x_min = np.ones(els.shape[0])*1e-5 # Minimum young modulus of the material
x_max = np.ones(els.shape[0]) # Maximum young modulus of the material
lamb = 0
penal = 3 # Penalization factor
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

    disp, UC, rhs_vec = fem_sol(nodes, els, mats, loads)
    stress_nodes = pos.stress_truss(nodes, els, mats, UC)

    # Compliance
    g = rhs_vec.T@disp
    d_g = sensi_el(els, mats, nodes, UC)

    # Filtering the design variable to be in a feasible interval
    x_j_after_prev = x_j_prev
    x_j_prev = x_j.copy()

    # Calculate the asymptotes
    q_o = -(x_j - L_j)**2 * d_g
    r_o = g + ((x_j - L_j) * d_g).sum()

    for _ in range(1000):
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






# %%
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


# %%

plt.figure(figsize=(12, 4))
plt.title('Original truss')
plot_truss(nodes, els, mats, stress_nodes)
plt.show()