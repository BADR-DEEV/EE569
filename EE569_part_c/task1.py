import numpy as np
import matplotlib.pyplot as plt
from keras import datasets

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# One-Hot Encoding
def one_hot_encode(y, num_classes):
    one_hot = np.zeros((num_classes, len(y)))
    one_hot[y.astype(int), np.arange(len(y))] = 1
    return one_hot

class Node:
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def forward(self):
        pass

    def backward(self):
        pass

class Parameter(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.grad = np.zeros_like(value)

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

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dims, output_dim, activation="relu"):
        self.nodes = []
        self.trainables = []
        self.activation_func = relu if activation == "relu" else np.tanh
        self.activation_derivative = relu_derivative if activation == "relu" else lambda x: 1 - np.tanh(x)**2

        self.layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(Linear(previous_dim, hidden_dim))
            previous_dim = hidden_dim
        self.layers.append(Linear(previous_dim, output_dim))

        self._build_graph()

    def _build_graph(self):
        for layer in self.layers:
            self.nodes.append(layer)
            self.trainables.append(layer.W)
            self.trainables.append(layer.b)

    def forward(self, X):
        self.a = [X]  
        self.z = []   
        for i, layer in enumerate(self.layers):
            z = layer.forward(self.a[-1])
            self.z.append(z)
            if i < len(self.layers) - 1:  
                self.a.append(self.activation_func(z))
            else: 
                exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  
                self.a.append(exp_z / np.sum(exp_z, axis=0, keepdims=True))
        return self.a[-1]

    def backward(self, y):
        dz = self.a[-1] - y
        for i in range(len(self.layers) - 1, -1, -1):
            if i < len(self.layers) - 1:
                dz = dz * self.activation_derivative(self.z[i])
            dz = self.layers[i].backward(dz)

    def update(self, lr):
        for param in self.trainables:
            param.value -= lr * param.grad


def train(model, X, y, epochs, learning_rate):
    losses = []
    for epoch in range(epochs):
        current_lr = learning_rate / (1 + 0.01 * epoch)  
        
        y_pred = model.forward(X)
        loss = -np.sum(y * np.log(y_pred + 1e-7)) / X.shape[1]
        model.backward(y)
        model.update(current_lr)
        
        losses.append(loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    return losses

def evaluate(model, X, y):
    y_pred = model.forward(X)
    y_pred_labels = np.argmax(y_pred, axis=0)
    return np.mean(y_pred_labels == y) * 100

print("Loading data mnist")
(train_X, train_y), (test_X, test_y) = datasets.mnist.load_data()

train_X = train_X.reshape(train_X.shape[0], -1).T / 255.0
test_X = test_X.reshape(test_X.shape[0], -1).T / 255.0
train_y = train_y.astype(int)
test_y = test_y.astype(int)

train_y_onehot = one_hot_encode(train_y, 10)

input_dim, hidden_dims, output_dim = 784, [128, 64], 10
activation = "relu"  
nn = NeuralNetwork(input_dim, hidden_dims, output_dim, activation=activation)

losses = train(nn, train_X, train_y_onehot, epochs=100, learning_rate=0.1)

# Plot Loss
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

accuracy = evaluate(nn, test_X, test_y)
print(f"Test Accuracy: {accuracy:.2f}%")

y_pred = nn.forward(test_X)
y_pred_labels = np.argmax(y_pred, axis=0)

fig, axes = plt.subplots(1, 10, figsize=(10, 2))
for i, ax in enumerate(axes):
    ax.imshow(test_X[:, i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Pred: {y_pred_labels[i]}")
    ax.axis('off')
plt.show()
