# %%
import numpy as np
from scipy.optimize import minimize

# Define the Lagrangian function L(x, lambda)
def lagrangian(x, lambda_):
    x1, x2 = x
    return 2 * x1 + 3 * x2 + lambda_ * (4 - x1 - x2)

# Define the gradient of the Lagrangian function
def lagrangian_gradient(x, lambda_):
    x1, x2 = x
    grad_x1 = 2 - lambda_
    grad_x2 = 3 - lambda_
    return np.array([grad_x1, grad_x2])

# Define the dual objective function for minimization
def dual_objective(lambda_):
    result = minimize(lagrangian, [0, 0], args=(lambda_,), constraints={'type': 'ineq', 'fun': lambda x: 4 - x[0] - x[1]})
    return result.fun

# Initial guess for lambda
lambda_ = 0

# Learning rate
learning_rate = 0.01

# Tolerance for convergence
tolerance = 1e-6

# Maximum number of iterations
max_iterations = 1000

for iteration in range(max_iterations):
    # Calculate the gradient of the dual objective function
    gradient = -lagrangian_gradient([0, 0], lambda_)
    
    # Update lambda using gradient descent
    lambda_ = lambda_ - learning_rate * gradient
    
    # Check for convergence
    if np.linalg.norm(gradient) < tolerance:
        break

# Print the optimized value of lambda
print("Optimal Lambda:", lambda_)

# Calculate the optimal value of g(lambda)
optimal_g_value = dual_objective(lambda_)
print("Optimal Value of g(lambda):", optimal_g_value)
