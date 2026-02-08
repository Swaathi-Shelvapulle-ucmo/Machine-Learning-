import numpy as np
import matplotlib.pyplot as plt

# --- 1. Dataset Generation ---
np.random.seed(42)
# Create 200 samples between 0 and 5
x_data = np.random.uniform(0, 5, 200)
gaussian_noise = np.random.normal(0, 1, 200)
# Formula: y = 3 + 4x + noise
y_targets = 3 + 4 * x_data + gaussian_noise

# Add a bias column (ones) for the intercept calculation
# Using column_stack for a clean, basic approach
input_matrix = np.column_stack((np.ones(len(x_data)), x_data))

# --- 2. Closed-form Solution (Normal Equation) ---
# theta = (X^T * X)^-1 * X^T * y
xt_x_inv = np.linalg.inv(np.dot(input_matrix.T, input_matrix))
xt_y = np.dot(input_matrix.T, y_targets)
weights_normal = np.dot(xt_x_inv, xt_y)

# --- 3. Gradient Descent Implementation ---
weights_gd = np.array([0.0, 0.0]) # Initialize theta [0, 0]
learning_rate = 0.05
iterations = 1000
num_samples = len(y_targets)
loss_tracker = []

for _ in range(iterations):
    # Calculate predictions (X * theta)
    model_estimates = np.dot(input_matrix, weights_gd)
    
    # Calculate difference from actual data
    residual_error = model_estimates - y_targets
    
    # Compute Gradient of MSE
    gradient_vector = (2 / num_samples) * np.dot(input_matrix.T, residual_error)
    
    # Update weights (theta = theta - eta * gradient)
    weights_gd = weights_gd - (learning_rate * gradient_vector)
    
    # Track the Mean Squared Error (MSE) for the curve
    current_loss = np.mean(np.square(residual_error))
    loss_tracker.append(current_loss)

# --- 4. Comparison & Results Report ---
print("--- Comparison Results ---")
print(f"Normal Equation  | Intercept: {weights_normal[0]:.4f} | Slope: {weights_normal[1]:.4f}")
print(f"Gradient Descent | Intercept: {weights_gd[0]:.4f} | Slope: {weights_gd[1]:.4f}")

# --- 5. Visualization ---
plt.figure(figsize=(14, 5))

# Plot A: Data and Comparison Lines
plt.subplot(1, 2, 1)
plt.scatter(x_data, y_targets, color='lightgray', alpha=0.6, label='Raw Data')
# Drawing the prediction lines
plt.plot(x_data, np.dot(input_matrix, weights_normal), color='firebrick', label='Closed-form (NE)')
plt.plot(x_data, np.dot(input_matrix, weights_gd), color='royalblue', linestyle='--', label='Grad Descent (GD)')
plt.xlabel("X Value")
plt.ylabel("Y Value")
plt.title("Comparison: Normal Equation vs Gradient Descent")
plt.legend()

# Plot B: Loss Curve
plt.subplot(1, 2, 2)

plt.plot(loss_tracker, color='darkorange', linewidth=2)
plt.xlabel("Iterations")
plt.ylabel("MSE (Loss)")
plt.title("Gradient Descent Learning Path")
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()