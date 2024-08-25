import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate sample data
np.random.seed(42)  # For reproducibility
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Step 2: Add the intercept term to X (X_b)
X_b = np.c_[np.ones((100, 1)), X]  # Add x0 = 1 to each instance

# Step 3: Initialize parameters (beta_0 and beta_1)
theta = np.random.randn(2, 1)

# Step 4: Define learning rate and number of iterations
learning_rate = 0.1
n_iterations = 1000
m = 100  # number of instances

# Step 5: Gradient Descent
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients

# Step 6: Output the results
print(f"Intercept (beta_0): {theta[0][0]}")
print(f"Slope (beta_1): {theta[1][0]}")

# Step 7: Predict
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # Add x0 = 1 to each instance
y_predict = X_new_b.dot(theta)

# Step 8: Plot the results
plt.plot(X_new, y_predict, "r-", label="Predictions")
plt.plot(X, y, "b.")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
