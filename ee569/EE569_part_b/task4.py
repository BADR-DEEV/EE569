import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# ReLU Activation and Derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# One-Hot Encoding
def one_hot_encode(y, num_classes):
    one_hot = np.zeros((num_classes, len(y)))
    one_hot[y.astype(int), np.arange(len(y))] = 1
    return one_hot

# Base Node Class for Computation Graph
class Node:
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def forward(self):
        pass

    def backward(self):
        pass

# Parameter Node
class Parameter(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.grad = np.zeros_like(value)

# Linear Layer Class
class Linear(Node):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W = Parameter(np.random.randn(output_dim, input_dim) * np.sqrt(2 / (input_dim + output_dim)))
        self.b = Parameter(np.zeros((output_dim, 1)))

    def forward(self, X):
        self.X = X
        self.Z = self.W.value @ X + self.b.value
        return self.Z

    def backward(self, dZ):
        self.W.grad = dZ @ self.X.T / self.X.shape[1]
        self.b.grad = np.sum(dZ, axis=1, keepdims=True) / self.X.shape[1]
        return self.W.value.T @ dZ

# Neural Network Class with Dynamic Graph
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.nodes = []
        self.trainables = []

        # Define Layers
        self.layer1 = Linear(input_dim, hidden_dim)
        self.layer2 = Linear(hidden_dim, output_dim)

        # Collect Trainables Automatically
        self._build_graph()

    def _build_graph(self):
        for layer in [self.layer1, self.layer2]:
            self.nodes.append(layer)
            self.trainables.append(layer.W)
            self.trainables.append(layer.b)

    def forward(self, X):
        self.z1 = self.layer1.forward(X)
        self.a1 = relu(self.z1)
        self.z2 = self.layer2.forward(self.a1)
        exp_z2 = np.exp(self.z2 - np.max(self.z2, axis=0, keepdims=True))
        self.y_pred = exp_z2 / np.sum(exp_z2, axis=0, keepdims=True)
        return self.y_pred

    def backward(self, y):
        dz2 = self.y_pred - y
        da1 = self.layer2.backward(dz2)
        dz1 = da1 * relu_derivative(self.z1)
        self.layer1.backward(dz1)

    def update(self, lr):
        for param in self.trainables:
            param.value -= lr * param.grad

# Training Function
def train(model, X, y, epochs, learning_rate):
    losses = []
    for epoch in range(epochs):
        current_lr = learning_rate / (1 + 0.01 * epoch)  # Decay learning rate
        
        # Forward and Backward Pass
        y_pred = model.forward(X)
        loss = -np.sum(y * np.log(y_pred + 1e-7)) / X.shape[1]
        model.backward(y)
        model.update(current_lr)
        
        losses.append(loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    return losses

# Evaluate Function
def evaluate(model, X, y):
    y_pred = model.forward(X)
    y_pred_labels = np.argmax(y_pred, axis=0)
    return np.mean(y_pred_labels == y) * 100

# Load MNIST Dataset
mnist = datasets.load_digits()
X, y = mnist['data'], mnist['target'].astype(int)
X = X.T / 16.0  # Normalize the input data

# Split Data into 60-40% train-test ratio
X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.4, random_state=42)
X_train, X_test = X_train.T, X_test.T
y_train_onehot = one_hot_encode(y_train, 10)

# Initialize Model
input_dim, hidden_dim, output_dim = 64, 64, 10
nn = NeuralNetwork(input_dim, hidden_dim, output_dim)

# Train the Model
losses = train(nn, X_train, y_train_onehot, epochs=200, learning_rate=0.1)

# Plot Loss
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Evaluate
accuracy = evaluate(nn, X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}%")

# Visualize First 10 Predictions
y_pred = nn.forward(X_test)
y_pred_labels = np.argmax(y_pred, axis=0)

fig, axes = plt.subplots(1, 10, figsize=(10, 2))
for i, ax in enumerate(axes):
    ax.imshow(X_test[:, i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {y_pred_labels[i]}")
    ax.axis('off')
plt.show()
