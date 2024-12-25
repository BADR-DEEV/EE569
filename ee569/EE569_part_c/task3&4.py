import numpy as np
import matplotlib.pyplot as plt
from keras import datasets



def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True)) 
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

class Conv:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        self.biases = np.zeros((out_channels, 1))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        self.x = x
        batch_size, in_channels, height, width = x.shape
        out_channels, _, kernel_height, kernel_width = self.kernels.shape

        out_height = (height - kernel_height + 2 * self.padding) // self.stride + 1
        out_width = (width - kernel_width + 2 * self.padding) // self.stride + 1

        self.output = np.zeros((batch_size, out_channels, out_height, out_width))
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for b in range(batch_size):
            for o in range(out_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        region = x_padded[b, :, i*self.stride:i*self.stride+kernel_height, j*self.stride:j*self.stride+kernel_width]
                        self.output[b, o, i, j] = np.sum(region * self.kernels[o]) + self.biases[o]

        return relu(self.output)

    def backward(self, grad_output, learning_rate):
        batch_size, out_channels, out_height, out_width = grad_output.shape
        _, in_channels, kernel_height, kernel_width = self.kernels.shape

        grad_input = np.zeros_like(self.x)
        grad_kernels = np.zeros_like(self.kernels)
        grad_biases = np.zeros_like(self.biases)

        for b in range(batch_size):
            for o in range(out_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        region = self.x[b, :, i*self.stride:i*self.stride+kernel_height, j*self.stride:j*self.stride+kernel_width]
                        grad_kernels[o] += grad_output[b, o, i, j] * region
                        grad_biases[o] += grad_output[b, o, i, j]
                        grad_input[b, :, i*self.stride:i*self.stride+kernel_height, j*self.stride:j*self.stride+kernel_width] += grad_output[b, o, i, j] * self.kernels[o]

        self.kernels -= learning_rate * grad_kernels
        self.biases -= learning_rate * grad_biases

        return grad_input * relu_derivative(self.x)

class MaxPooling:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        self.x = x
        batch_size, in_channels, height, width = x.shape
        out_height = height // self.pool_size
        out_width = width // self.pool_size

        self.output = np.zeros((batch_size, in_channels, out_height, out_width))

        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        region = x[b, c, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size]
                        self.output[b, c, i, j] = np.max(region)

        return self.output

class Linear:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(out_features, in_features) * 0.1
        self.biases = np.zeros((out_features, 1))

    def forward(self, x):
        self.x = x.reshape(-1, 1)  # Flatten input
        return np.dot(self.weights, self.x) + self.biases

    def backward(self, grad_output, learning_rate):
        grad_weights = np.dot(grad_output, self.x.T)
        grad_biases = grad_output
        grad_input = np.dot(self.weights.T, grad_output)

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input

class CNN:
    def __init__(self):
        self.conv1 = Conv(1, 16, kernel_size=3)
        self.pool1 = MaxPooling(pool_size=2)
        self.conv2 = Conv(16, 32, kernel_size=3)
        self.pool2 = MaxPooling(pool_size=2)
        self.fc = Linear(32 * 5 * 5, 10)  

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.pool2.forward(x)
        x = self.fc.forward(x.flatten())
        return softmax(x)

def preprocess_data():
    (train_X, train_y), (test_X, test_y) = datasets.mnist.load_data()
    train_X = train_X / 255.0  
    test_X = test_X / 255.0
    train_X = train_X[:, np.newaxis, :, :]  
    test_X = test_X[:, np.newaxis, :, :]
    return train_X, train_y, test_X, test_y

def plot_predictions(cnn, test_X, test_y, num_images=10):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, ax in enumerate(axes):
        image = test_X[i:i+1]
        logits = cnn.forward(image)
        prediction = np.argmax(logits)
        ax.imshow(image[0, 0], cmap='gray')
        ax.set_title(f"Pred: {prediction}\nTrue: {test_y[i]}")
        ax.axis('off')
    plt.show()

def train_cnn(cnn, train_X, train_y, epochs, learning_rate):
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(train_X)):
            x = train_X[i:i+1]  
            y = np.zeros((10, 1))
            y[train_y[i]] = 1

            logits = cnn.forward(x)
            loss = -np.sum(y * np.log(logits + 1e-7))
            total_loss += loss

            grad_output = logits - y
            cnn.fc.backward(grad_output, learning_rate)

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_X):.4f}")

def evaluate_cnn(cnn, test_X, test_y):
    correct = 0
    for i in range(len(test_X)):
        logits = cnn.forward(test_X[i:i+1])
        prediction = np.argmax(logits)
        if prediction == test_y[i]:
            correct += 1
    return (correct / len(test_X)) * 100

train_X, train_y, test_X, test_y = preprocess_data()

cnn = CNN()
print("traing CNN...")
train_cnn(cnn, train_X[:100], train_y[:100], epochs=20, learning_rate=0.1)

print("calculating CNN...")
accuracy = evaluate_cnn(cnn, test_X[:100], test_y[:100])
print(f"Test Accuracy: {accuracy:.2f}%")

plot_predictions(cnn, test_X[:10], test_y[:10])
