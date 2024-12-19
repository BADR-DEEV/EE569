from EDF import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define constants hyperparamters
CLASS1_SIZE = 100
CLASS2_SIZE = 100
N_FEATURES = 2
N_OUTPUT = 1
LEARNING_RATE = 0.02
EPOCHS = 100
TEST_SIZE = 0.25

# Define the means and covariances of the two components
MEAN1 = np.array([1, 2])
COV1 = np.array([[1, 0], [0, 1]])
MEAN2 = np.array([1, -2])
COV2 = np.array([[1, 0], [0, 1]])

# Generate random points from the two components
X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE)
X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE)

# Combine the points and generate labels
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))

# Plot the generated data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Generated Data')
plt.show()

# Split data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Model parameters
n_features = X_train.shape[1]
n_output = 1

# Initialize weights and biases
W0 = np.zeros(1)
W1 = np.random.randn(1) * 0.1
W2 = np.random.randn(1) * 0.1

# Create nodes
x1_node = Input()
x2_node = Input()
y_node = Input()

w0_node = Parameter(W0)
w1_node = Parameter(W1)
w2_node = Parameter(W2)
#b1_node = Parameter(b1)

# Build computation graph
u1_node = Multiply(x1_node,w1_node)
u2_node = Multiply(x2_node,w2_node)
u12_node = Addition(u1_node,u2_node)
u_node = Addition(u12_node, w0_node)
sigmoid = Sigmoid(u_node)
loss = BCE(y_node, sigmoid)

# Create graph outside the training loop
graph = [x1_node,x2_node,w0_node,w1_node,w2_node,u1_node,u2_node,u12_node,u_node,sigmoid,loss]
trainable = [w0_node,w1_node,w2_node]

# Training loop
epochs = 100
learning_rate = 0.001

# Forward and Backward Pass
def forward_pass(graph):
    for n in graph:
        n.forward()

def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()

# SGD Update
def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t][0]


for epoch in range(epochs):
    loss_value = 0
    for i in range(X_train.shape[0]):
        x1_node.value = X_train[i][0].reshape(1, -1)
        x2_node.value = X_train[i][1].reshape(1, -1)
        y_node.value = y_train[i].reshape(1, -1)

        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, learning_rate)

        loss_value += loss.value

    print(f"Epoch {epoch + 1}, Loss: {loss_value / X_train.shape[0]}")

# Evaluate the model
correct_predictions = 0
for i in range(X_test.shape[0]):
    x1_node.value = X_test[i][0].reshape(1, -1)
    x2_node.value = X_test[i][1].reshape(1, -1)
    forward_pass(graph)

    if np.argmax(sigmoid.value) == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")

x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
Z = []
for i,j in zip(xx.ravel(),yy.ravel()):
    x1_node.value = np.array([i]).reshape(1, -1)
    x2_node.value = np.array([j]).reshape(1, -1)
    forward_pass(graph)
    Z.append(sigmoid.value)
Z = np.array(Z).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()
