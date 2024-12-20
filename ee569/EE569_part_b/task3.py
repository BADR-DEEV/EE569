import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


#activitin function
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


#binary Cross loss
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


#parameter node
class Parameter:
    def __init__(self, shape):
        self.value = np.random.randn(*shape) * np.sqrt(2 / np.sum(shape))  
        self.grad = None



class Linear:
    """Fully connected layer with automated parameter initialization."""
    def __init__(self, input_dim, output_dim):
        self.W = Parameter((output_dim, input_dim))
        self.b = Parameter((output_dim, 1))
        self.x = None
        self.z = None

    def forward(self, x):
        self.x = x
        self.z = self.W.value @ x + self.b.value
        return self.z

    def backward(self, grad_output):
        batch_size = grad_output.shape[1]
        self.W.grad = grad_output @ self.x.T / batch_size
        self.b.grad = np.sum(grad_output, axis=1, keepdims=True) / batch_size
        grad_input = self.W.value.T @ grad_output
        return grad_input


# generating xor data
def generate_xor_data(samples_per_class=100):
    np.random.seed(42)
    mean_00, cov = [-1, -1], [[0.1, 0], [0, 0.1]]
    mean_01, mean_10, mean_11 = [1, -1], [-1, 1], [1, 1]
#verticl stack and horiztl stack
    X0 = np.vstack((multivariate_normal.rvs(mean_00, cov, samples_per_class),
                    multivariate_normal.rvs(mean_11, cov, samples_per_class)))
    X1 = np.vstack((multivariate_normal.rvs(mean_01, cov, samples_per_class),
                    multivariate_normal.rvs(mean_10, cov, samples_per_class)))
    X = np.vstack((X0, X1))
    y = np.hstack((np.zeros(len(X0)), np.ones(len(X1))))
    return X, y

def split_data(X, y, train_ratio=0.6):
    """Split Data into Training and Testing Sets."""
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(len(X) * train_ratio)
    return X[indices[:train_size]].T, X[indices[train_size:]].T, y[indices[:train_size]], y[indices[train_size:]]


#training
def train(X, y, layers, epochs, learning_rate, batch_size):
    losses = []
    for epoch in range(epochs):
        indices = np.random.permutation(X.shape[1])
        X, y = X[:, indices], y[indices]
        epoch_loss = 0

        for i in range(0, X.shape[1], batch_size):
            X_batch = X[:, i:i+batch_size]
            y_batch = y[i:i+batch_size].reshape(1, -1)

         
            activations = [X_batch]
            for layer in layers[:-1]:
                activations.append(relu(layer.forward(activations[-1])))
            output = sigmoid(layers[-1].forward(activations[-1]))

       
            loss = binary_cross_entropy(y_batch, output)
            epoch_loss += loss

          
            grad_output = output - y_batch
            grad_input = layers[-1].backward(grad_output)
            for j in range(len(layers) - 2, -1, -1):
                grad_input = layers[j].backward(grad_input * relu_derivative(activations[j + 1]))

            #updating
            for layer in layers:
                layer.W.value -= learning_rate * layer.W.grad
                layer.b.value -= learning_rate * layer.b.grad

        losses.append(epoch_loss / (X.shape[1] // batch_size))
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {losses[-1]:.4f}")
    return losses



def evaluate(layers, X, y):
    activations = [X]
    for layer in layers[:-1]:
        activations.append(relu(layer.forward(activations[-1])))
    output = sigmoid(layers[-1].forward(activations[-1]))
    predictions = (output > 0.5).astype(int)
    accuracy = np.mean(predictions == y.reshape(1, -1)) * 100
    return accuracy


#ploting decition boundary
def plot_decision_boundary(layers, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    grid_points = np.c_[xx.ravel(), yy.ravel()].T
    output = grid_points
    for layer in layers[:-1]:
        output = relu(layer.forward(output))
    output = sigmoid(layers[-1].forward(output)).reshape(xx.shape)

    plt.contourf(xx, yy, output > 0.5, alpha=0.5, cmap="bwr")
    plt.scatter(X[0, :], X[1, :], c=y, cmap="bwr", edgecolor="k")
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


X, y = generate_xor_data(100)
X_train, X_test, y_train, y_test = split_data(X, y)

# in here i defined the nerual network Layers
layers = [
    Linear(2, 64),  #this is my input 
    Linear(64, 1)   # and this is the hidden layer to the output 
]

# training
losses = train(X_train, y_train, layers, epochs=100, learning_rate=0.01, batch_size=32)

# Plotting the trainging 
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


accuracy = evaluate(layers, X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}%")


plot_decision_boundary(layers, X_test, y_test)
