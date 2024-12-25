import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define the numerical derivative function
def numerical_derivative(f, x, h=1e-3):
    return (f(x + h) - f(x - h)) / (2 * h)

# Generate x values
x = np.linspace(-10, 10, 1000)

# Calculate the actual gradient
actual_gradient = sigmoid_derivative(x)

# Calculate the numerical gradient
numerical_gradient = np.array([numerical_derivative(sigmoid, xi) for xi in x])

# Plot the results
plt.plot(x, actual_gradient, label='Actual Gradient')
plt.plot(x, numerical_gradient, label='Numerical Gradient')
plt.xlabel('x')
plt.ylabel('Gradient')
plt.title('Gradient of Sigmoid Function')
plt.legend()
plt.show()

# Plot the absolute difference between the actual and numerical gradients
plt.plot(x, np.abs(actual_gradient - numerical_gradient))
plt.xlabel('x')
plt.ylabel('Absolute Difference')
plt.title('Absolute Difference between Actual and Numerical Gradients')
plt.show()