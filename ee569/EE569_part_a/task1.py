import numpy as np


class Linear:
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.x = None
        self.grad_A = None
        self.grad_b = None
        self.grad_x = None

    def forward(self, x):

        self.x = x
        # in here we are doing the linear transform
        y = self.A @ x + self.b
        return y

    def backward(self, grad_output):

        # we are calculating the grad (which is the derivative of the loss function) with respect to A, b, and x
        self.grad_A = np.outer(grad_output, self.x)  # dL/dA = grad_output @ x^T
        self.grad_b = grad_output  # dL/db = grad_output
        self.grad_x = self.A.T @ grad_output  # dL/dx = A^T @ grad_output

        return self.grad_x


#manually creating the input, weights, and bias 
# -----------------------------------------------------##
A = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

b = np.array([0.1, 0.2, 0.3])

x = np.array([1.0, 2.0])
# ----------------------------------------------##

linear_layer = Linear(A, b)

# forword
output = linear_layer.forward(x)
print("Forward Pass Output:")
print(output)

# i assumed the gradient from the next layer in here
grad_output = np.array([0.5, -0.5, 1.0])
grad_x = linear_layer.backward(grad_output)


print("\nGradients from Backward Pass:")
print("Gradient with respect to A (grad_A):")
print(linear_layer.grad_A)

print("Gradient with respect to b (grad_b):")
print(linear_layer.grad_b)

print("Gradient with respect to input x (grad_x):")
print(grad_x)
