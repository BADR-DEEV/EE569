import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# Linear Node (Batch Version)
class Linear:
    def __init__(self, A, b):
        self.A = A
        # this change it to a column vector
        self.b = b.reshape(-1, 1)
        self.x = None
        self.grad_A = None
        self.grad_b = None
        self.grad_x = None

    def forward(self, x):
        self.x = x 
        y = self.A @ x + self.b  
        return y  

    def backward(self, grad_output):
       
        batch_size = grad_output.shape[1]

        self.grad_A = grad_output @ self.x.T / batch_size  # i took avg
        self.grad_b = np.sum(grad_output, axis=1, keepdims=True) / batch_size
        self.grad_x = self.A.T @ grad_output

        return self.grad_x
# ------------------------- #
# Sigmoid Node (Batch Version)
class Sigmoid:
    def __init__(self):
        self.value = None
        self.grad_input = None

    def forward(self, input_node):
        self.value = 1 / (1 + np.exp(-input_node))  
        return self.value

    def backward(self, grad_output):
        self.grad_input = (
            grad_output * self.value * (1 - self.value)
        )  # Gradient of Sigmoid
        return self.grad_input

# Binary Cross-Entropy Loss (Batch Version)
class BCE:
    def __init__(self):
        self.value = None
        self.grad_input = None

    def forward(self, target, prediction):
        #target is ground truth, 
        # prediction is the output of the sigmoid
        # target --> eg (100,32) 100 samples and 32 (features)
        # so tareget.shape[1] = 32
        batch_size = target.shape[1]
        y = target
        y_pred = np.clip(prediction, 1e-7, 1 - 1e-7)
        # we took the average of the loss
        self.value = (
            -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / batch_size
        )
        return self.value

    def backward(self, target, prediction):
        batch_size = target.shape[1]
        y = target
        y_pred = np.clip(prediction, 1e-7, 1 - 1e-7)
        # we took the average of the loss
        self.grad_input = -(y / y_pred - (1 - y) / (1 - y_pred)) / batch_size
        return self.grad_input



# Data Generation
CLASS1_SIZE = 100
CLASS2_SIZE = 100
BATCH_SIZE = 32
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
A = np.random.randn(1, n_features) * 0.1  # Weights
b = np.random.randn(1) * 0.1  # Bias

linear_node = Linear(A, b)
sigmoid_node = Sigmoid()
bce_loss = BCE()

LEARNING_RATE = 0.01
EPOCHS = 150

# ------------------------- #
# Training Loop with Batching
for epoch in range(EPOCHS):
    total_loss = 0
    num_batches = int(np.ceil(X_train.shape[0] / BATCH_SIZE))

    for batch_idx in range(num_batches):
        # Batch indices
        start_index = batch_idx * BATCH_SIZE
        end_index = min(start_index + BATCH_SIZE, X_train.shape[0])

        # Fetch batch
        X_batch = X_train[start_index:end_index].T  # (features, batch_size)
        y_batch = y_train[start_index:end_index].reshape(
            1, -1
        )  # (1, batch_size) this means that we wand it to be a row vector

        # Forward Pass
        linear_output = linear_node.forward(X_batch)
        sigmoid_output = sigmoid_node.forward(linear_output)
        loss_value = bce_loss.forward(y_batch, sigmoid_output)

        # Backward Pass
        grad_loss = bce_loss.backward(y_batch, sigmoid_output)
        grad_sigmoid = sigmoid_node.backward(grad_loss)
        linear_node.backward(grad_sigmoid)

        # Update Parameters
        linear_node.A -= LEARNING_RATE * linear_node.grad_A
        linear_node.b -= LEARNING_RATE * linear_node.grad_b

        total_loss += loss_value

    print(f"Epoch {epoch + 1}, Loss: {total_loss / num_batches}")

# ------------------------- #
# Model Evaluation
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
