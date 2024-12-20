import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


#activation fucntion 
def relu(x):
    """ReLU Activation"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)

def sigmoid(x):
    """Sigmoid Activation"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of Sigmoid"""
    return sigmoid(x) * (1 - sigmoid(x))

# ----------------------------
#binary Cross loss
def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Avoid log(0)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


#generating xor data
def generate_xor_data(samples_per_class=100):
    np.random.seed(42)
    mean_00, cov = [-1, -1], [[0.1, 0], [0, 0.1]]
    mean_01, mean_10, mean_11 = [1, -1], [-1, 1], [1, 1]

    # Class 0
    X0 = np.vstack((
        multivariate_normal.rvs(mean_00, cov, samples_per_class),
        multivariate_normal.rvs(mean_11, cov, samples_per_class)
    ))

    # Class 1
    X1 = np.vstack((
        multivariate_normal.rvs(mean_01, cov, samples_per_class),
        multivariate_normal.rvs(mean_10, cov, samples_per_class)
    ))

    # combining data 
    X = np.vstack((X0, X1))
    y = np.hstack((np.zeros(len(X0)), np.ones(len(X1))))
    return X, y

def split_data(X, y, train_ratio=0.6):
    """Split Data into Training and Testing Sets"""
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(len(X) * train_ratio)
    return X[indices[:train_size]].T, X[indices[train_size:]].T, y[indices[:train_size]], y[indices[train_size:]]


#trainging
def train(X, y, W1, b1, W2, b2, epochs, learning_rate, batch_size):
    """Train Neural Network with Batch Updates"""
    losses = []
    for epoch in range(epochs):
    #in here i shuffled the data 
        indices = np.random.permutation(X.shape[1])
        X, y = X[:, indices], y[indices]

        total_loss = 0
        for i in range(0, X.shape[1], batch_size):
            X_batch = X[:, i:i+batch_size]
            y_batch = y[i:i+batch_size].reshape(1, -1)

            z1 = W1 @ X_batch + b1
            a1 = relu(z1)
            z2 = W2 @ a1 + b2
            y_pred = sigmoid(z2)

     
            loss = binary_cross_entropy(y_batch, y_pred)
            total_loss += loss

            dz2 = y_pred - y_batch
            dW2 = dz2 @ a1.T / X_batch.shape[1]
            db2 = np.sum(dz2, axis=1, keepdims=True) / X_batch.shape[1]

            da1 = W2.T @ dz2
            dz1 = da1 * relu_derivative(z1)
            dW1 = dz1 @ X_batch.T / X_batch.shape[1]
            db1 = np.sum(dz1, axis=1, keepdims=True) / X_batch.shape[1]

            # updating the wights
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1

        losses.append(total_loss / (X.shape[1] // batch_size))
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {losses[-1]:.4f}")
    return losses, W1, b1, W2, b2


# plotting decitin boundary
def plot_decision_boundary(W1, b1, W2, b2, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    grid_points = np.c_[xx.ravel(), yy.ravel()].T
    z1 = W1 @ grid_points + b1
    a1 = relu(z1)
    z2 = W2 @ a1 + b2
    predictions = (sigmoid(z2) > 0.5).astype(int).reshape(xx.shape)

    plt.contourf(xx, yy, predictions, alpha=0.5, cmap="bwr")
    plt.scatter(X[0, :], X[1, :], c=y, cmap="bwr", edgecolor="k")
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# Generate with split data 60/40
X, y = generate_xor_data(100)
X_train, X_test, y_train, y_test = split_data(X, y)


input_dim, hidden_dim, output_dim = 2, 64, 1
W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2 / input_dim)
b1 = np.zeros((hidden_dim, 1))
W2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2 / hidden_dim)
b2 = np.zeros((output_dim, 1))


losses, W1, b1, W2, b2 = train(X_train, y_train, W1, b1, W2, b2, epochs=100, learning_rate=0.01, batch_size=32)


plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()






plot_decision_boundary(W1, b1, W2, b2, X_test, y_test)
