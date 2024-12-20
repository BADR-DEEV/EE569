import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# Linear Node with batching
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


# Sigmoid function
class Sigmoid:
    def forward(self, x):
        self.value = 1 / (1 + np.exp(-x))
        return self.value

    def backward(self, grad_output):
        return grad_output * self.value * (1 - self.value)


# Binary Cross func 
class BCE:
    def forward(self, target, prediction):
        batch_size = target.shape[1]
        y_pred = np.clip(prediction, 1e-7, 1 - 1e-7)
        loss = (
            -np.sum(target * np.log(y_pred) + (1 - target) * np.log(1 - y_pred))
            / batch_size
        )
        return loss

    def backward(self, target, prediction):
        batch_size = target.shape[1]
        y_pred = np.clip(prediction, 1e-7, 1 - 1e-7)
        return -(target / y_pred - (1 - target) / (1 - y_pred)) / batch_size


# Generating xor data
def generate_xor_data(samples_per_class=100):
    np.random.seed(42)
    mean_00, cov = [-1, -1], [[0.1, 0], [0, 0.1]]
    mean_01 = [1, -1]
    mean_10 = [-1, 1]
    mean_11 = [1, 1]

    # Class 0
    X0_1 = multivariate_normal.rvs(mean_00, cov, samples_per_class)
    X0_2 = multivariate_normal.rvs(mean_11, cov, samples_per_class)
    X_class0 = np.vstack((X0_1, X0_2))

    # Class 1
    X1_1 = multivariate_normal.rvs(mean_01, cov, samples_per_class)
    X1_2 = multivariate_normal.rvs(mean_10, cov, samples_per_class)
    X_class1 = np.vstack((X1_1, X1_2))

    # Combining data with verticl stack and horizentl stack functions
    X = np.vstack((X_class0, X_class1))
    y = np.hstack((np.zeros(len(X_class0)), np.ones(len(X_class1))))

    return X, y


# i took some of the functionallity of the logistic reg function and implemneted it in one file 
def train_logistic_regression(X, y, learning_rate=0.01, epochs=1000, batch_size=32):
    n_features = X.shape[1]
    A = np.random.randn(1, n_features) * 0.1  # Initialize weights
    b = np.random.randn(1) * 0.1  # Initialize bias

    linear_node = Linear(A, b)
    sigmoid_node = Sigmoid()
    bce_loss = BCE()

    losses = []

    for epoch in range(epochs):
        indices = np.random.permutation(X.shape[0])
        X, y = X[indices], y[indices]
        total_loss = 0

        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i : i + batch_size].T
            y_batch = y[i : i + batch_size].reshape(1, -1)

          
            linear_output = linear_node.forward(X_batch)
            sigmoid_output = sigmoid_node.forward(linear_output)
            loss = bce_loss.forward(y_batch, sigmoid_output)

         

            grad_loss = bce_loss.backward(y_batch, sigmoid_output)
            grad_sigmoid = sigmoid_node.backward(grad_loss)
            linear_node.backward(grad_sigmoid)

            # Updatingggg
            linear_node.A -= learning_rate * linear_node.grad_A
            linear_node.b -= learning_rate * linear_node.grad_b
            total_loss += loss

        losses.append(total_loss / (X.shape[0] // batch_size))
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {losses[-1]}")

    return linear_node, sigmoid_node, losses


# Split Data into 60/40
def split_data(X, y, train_ratio=0.6):
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(len(X) * train_ratio)

    train_idx, test_idx = indices[:train_size], indices[train_size:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test


# Ploting decition boundary
def plot_decision_boundary(linear_node, sigmoid_node, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = np.array(
        [
            linear_node.forward(np.array([[i], [j]]))
            for i, j in zip(xx.ravel(), yy.ravel())
        ]
    )
    Z = sigmoid_node.forward(Z.T)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
    plt.title("Decision Boundary for XOR Problem")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# PLOTTING
X, y = generate_xor_data(samples_per_class=100)
X_train, X_test, y_train, y_test = split_data(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr")
plt.title("XOR-Like Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

linear_node, sigmoid_node, losses = train_logistic_regression(X_train, y_train)
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

plot_decision_boundary(linear_node, sigmoid_node, X_test, y_test)
