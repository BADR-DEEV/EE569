import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# Linear Node (Batch Version)
class Linear:
    def __init__(self, A, b):
        self.A = A
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
        self.grad_A = grad_output @ self.x.T / batch_size
        self.grad_b = np.sum(grad_output, axis=1, keepdims=True) / batch_size
        self.grad_x = self.A.T @ grad_output
        return self.grad_x



# Sigmoid Node (Batch Version)
class Sigmoid:
    def __init__(self):
        self.value = None
        self.grad_input = None

    def forward(self, input_node):
        self.value = 1 / (1 + np.exp(-input_node))
        return self.value

    def backward(self, grad_output):
        self.grad_input = grad_output * self.value * (1 - self.value)
        return self.grad_input



# Binary Cross-Entropy Loss (Batch Version)
class BCE:
    def __init__(self):
        self.value = None
        self.grad_input = None

    def forward(self, target, prediction):
        batch_size = target.shape[1]
        y = target
        y_pred = np.clip(prediction, 1e-7, 1 - 1e-7)
        self.value = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / batch_size
        return self.value

    def backward(self, target, prediction):
        batch_size = target.shape[1]
        y = target
        y_pred = np.clip(prediction, 1e-7, 1 - 1e-7)
        self.grad_input = -(y / y_pred - (1 - y) / (1 - y_pred)) / batch_size
        return self.grad_input


# ----------------------------- #
# Data Generation
CLASS1_SIZE = 100
CLASS2_SIZE = 100
TEST_SIZE = 0.25

MEAN1 = np.array([1, 2])
COV1 = np.array([[1, 0], [0, 1]])
MEAN2 = np.array([1, -2])
COV2 = np.array([[1, 0], [0, 1]])

X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE)
X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE)
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
test_size = int(len(X) * TEST_SIZE)
train_indices = indices[test_size:]
test_indices = indices[:test_size]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, len(X_train)]
LEARNING_RATE = 0.01
EPOCHS = 20

batch_losses = []

for BATCH_SIZE in BATCH_SIZES:
    print(f"\nTraining with Batch Size: {BATCH_SIZE}")
    n_features = X_train.shape[1]
    A = np.random.randn(1, n_features) * 0.1
    b = np.random.randn(1) * 0.1

    linear_node = Linear(A, b)
    sigmoid_node = Sigmoid()
    bce_loss = BCE()

    epoch_losses = []

    for epoch in range(EPOCHS):
        #epoch means the number of times the model will see the entire dataset
        #eg if we had 1000 then the model will see the dataset 1000 times
        total_loss = 0
        num_batches = int(np.ceil(X_train.shape[0] / BATCH_SIZE))

        for batch_idx in range(num_batches):
            #we use batch because we can't pass the entire dataset at once it will be too much
            # so we section it in batches
            start_index = batch_idx * BATCH_SIZE
            end_index = min(start_index + BATCH_SIZE, X_train.shape[0])

            X_batch = X_train[start_index:end_index].T
            y_batch = y_train[start_index:end_index].reshape(1, -1)

            linear_output = linear_node.forward(X_batch)
            sigmoid_output = sigmoid_node.forward(linear_output)
            loss_value = bce_loss.forward(y_batch, sigmoid_output)

            grad_loss = bce_loss.backward(y_batch, sigmoid_output)
            grad_sigmoid = sigmoid_node.backward(grad_loss)
            linear_node.backward(grad_sigmoid)

            linear_node.A -= LEARNING_RATE * linear_node.grad_A
            linear_node.b -= LEARNING_RATE * linear_node.grad_b

            total_loss += loss_value

        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)

    batch_losses.append(epoch_losses)
    print(f"Final Loss for Batch Size {BATCH_SIZE}: {epoch_losses[-1]}")

# ------------------------- #
# Plot Training Loss for Different Batch Sizes
plt.figure(figsize=(10, 6))
for i, BATCH_SIZE in enumerate(BATCH_SIZES):
    plt.plot(range(1, EPOCHS + 1), batch_losses[i], label=f"Batch Size {BATCH_SIZE}")

plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Effect of Batch Size on Training Loss")
plt.legend()
plt.grid(True)
plt.show()
