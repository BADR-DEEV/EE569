import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# ------------------------- #
# Linear Node (from Task 1) (without batch version)

# Linear Node (Updated)
class Linear:
    def __init__(self, A, b):
        self.A = A  # Weights
        self.b = b  # Bias
        self.x = None
        self.grad_A = None
        self.grad_b = None
        self.grad_x = None

    def forward(self, x):
        self.x = x  # Store input
        y = self.A @ x + self.b  # Linear transformation
        return y

    def backward(self, grad_output):
        self.grad_A = np.outer(grad_output, self.x)  # Gradient wrt weights
        self.grad_b = grad_output  # Gradient wrt bias
        self.grad_x = self.A.T @ grad_output  # Gradient wrt input
        return self.grad_x


# Sigmoid Node (Updated)
class Sigmoid:
    def __init__(self):
        self.value = None
        self.grad_input = None

    def forward(self, input_value):
        self.value = 1 / (1 + np.exp(-input_value))  # Sigmoid function
        return self.value

    def backward(self, grad_output):
        self.grad_input = grad_output * self.value * (1 - self.value)  # Sigmoid gradient
        return self.grad_input


# Binary Cross-Entropy Loss (Updated)
class BCE:
    def __init__(self):
        self.value = None
        self.grad_input = None

    def forward(self, target, prediction):
        # Clip predictions to prevent log(0) errors
        y_pred = np.clip(prediction, 1e-7, 1 - 1e-7)
        self.value = -np.mean(target * np.log(y_pred) + (1 - target) * np.log(1 - y_pred))
        return self.value

    def backward(self, target, prediction):
        y_pred = np.clip(prediction, 1e-7, 1 - 1e-7)
        self.grad_input = -(target / y_pred - (1 - target) / (1 - y_pred)) / target.shape[0]
        return self.grad_input




# ------------------------- #
# Data Generation
CLASS1_SIZE = 100
CLASS2_SIZE = 100
N_FEATURES = 2
N_OUTPUT = 1
LEARNING_RATE = 0.02
EPOCHS = 100
TEST_SIZE = 0.25

MEAN1 = np.array([1, 2])
COV1 = np.array([[1, 0], [0, 1]])
MEAN2 = np.array([1, -2])
COV2 = np.array([[1, 0], [0, 1]])

# Generate data
X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE)
X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE)
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))

# Shuffle and split
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
test_size = int(len(X) * TEST_SIZE)
train_indices = indices[test_size:]
test_indices = indices[:test_size]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Plot data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# ------------------------- #
# Model Initialization
n_features = X_train.shape[1]
A = np.random.randn(1, n_features) * 0.1  # Small random weights
b = np.random.randn(1) * 0.1  # Small random bias

linear_node = Linear(A, b)  # Linear Node
sigmoid_node = Sigmoid()    # Sigmoid Node
bce_loss = BCE()            # BCE Loss
 # BCE Loss

LEARNING_RATE = 0.02
EPOCHS = 100

# ------------------------- #
# Training Loop
# Training Loop
for epoch in range(EPOCHS):
    total_loss = 0
    for i in range(X_train.shape[0]):
        x = X_train[i].reshape(-1, 1)  # Reshape input
        target = y_train[i].reshape(-1, 1)  # Reshape target to match dimensions

        # Forward Pass
        linear_output = linear_node.forward(x)
        sigmoid_output = sigmoid_node.forward(linear_output)
        loss_value = bce_loss.forward(target, sigmoid_output)

        # Backward Pass
        grad_loss = bce_loss.backward(target, sigmoid_output)
        grad_sigmoid = sigmoid_node.backward(grad_loss)
        linear_node.backward(grad_sigmoid)

        # Update Weights and Bias
        linear_node.A -= LEARNING_RATE * linear_node.grad_A
        linear_node.b -= LEARNING_RATE * np.squeeze(linear_node.grad_b)


        total_loss += loss_value

    print(f"Epoch {epoch + 1}, Loss: {total_loss / X_train.shape[0]}")


# # Evaluation Loop
correct_predictions = 0
for i in range(X_test.shape[0]):
    x = X_test[i].reshape(-1, 1)
    target = y_test[i]

    # Forward Pass
    linear_output = linear_node.forward(x)
    sigmoid_output = sigmoid_node.forward(linear_output)

    prediction = 1 if sigmoid_output >= 0.5 else 0
    if prediction == target:
        correct_predictions += 1

accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")


# ------------------------- #
# Decision Boundary Plot
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
Z = []
for i, j in zip(xx.ravel(), yy.ravel()):
    x_input = np.array([i, j]).reshape(-1, 1)
    linear_output = linear_node.forward(x_input)
    sigmoid_output = sigmoid_node.forward(linear_output)
    Z.append(sigmoid_output)
Z = np.array(Z).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
plt.title("Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
