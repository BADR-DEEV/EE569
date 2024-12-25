# from keras import datasets
# from matplotlib import pyplot
# import numpy as np

# # Loading the dataset
# (train_X, train_y), (test_X, test_y) = datasets.mnist.load_data()

# # Printing the shapes of the vectors
# print('X_train: ' + str(train_X.shape))
# print('Y_train: ' + str(train_y.shape))
# print('X_test:  ' + str(test_X.shape))
# print('Y_test:  ' + str(test_y.shape))

# # Preprocessing the data
# train_X = train_X.reshape((train_X.shape[0], 28 * 28)).astype('float32') / 255
# test_X = test_X.reshape((test_X.shape[0], 28 * 28)).astype('float32') / 255

# # One-hot encoding the labels
# train_y = np.eye(10)[train_y]
# test_y = np.eye(10)[test_y]

# # Define MLP using basic numpy functions (instead of Keras layers)
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def softmax(x):
#     e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
#     return e_x / e_x.sum(axis=1, keepdims=True)

# def mlp_forward(X, weights, biases):
#     return sigmoid(np.dot(X, weights) + biases)

# # Initialize weights and biases
# input_size = 28 * 28
# hidden_size = 128
# output_size = 10

# np.random.seed(42)

# # Weight initialization (for two layers: input to hidden, hidden to output)
# weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
# biases_input_hidden = np.zeros((1, hidden_size))

# weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
# biases_hidden_output = np.zeros((1, output_size))

# # Training hyperparameters
# learning_rate = 0.1
# epochs = 10
# batch_size = 32

# # Simple training loop (batch gradient descent)
# for epoch in range(epochs):
#     for i in range(0, len(train_X), batch_size):
#         X_batch = train_X[i:i+batch_size]
#         y_batch = train_y[i:i+batch_size]

#         # Forward pass
#         hidden_output = mlp_forward(X_batch, weights_input_hidden, biases_input_hidden)
#         output = softmax(np.dot(hidden_output, weights_hidden_output) + biases_hidden_output)

#         # Compute the loss (cross-entropy)
#         loss = -np.sum(y_batch * np.log(output + 1e-10)) / batch_size

#         # Backward pass (gradient calculation)
#         output_error = output - y_batch
#         hidden_error = np.dot(output_error, weights_hidden_output.T) * hidden_output * (1 - hidden_output)

#         # Update weights and biases
#         weights_hidden_output -= learning_rate * np.dot(hidden_output.T, output_error) / batch_size
#         biases_hidden_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True) / batch_size

#         weights_input_hidden -= learning_rate * np.dot(X_batch.T, hidden_error) / batch_size
#         biases_input_hidden -= learning_rate * np.sum(hidden_error, axis=0, keepdims=True) / batch_size

#     print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# # Evaluate on the test set
# hidden_output_test = mlp_forward(test_X, weights_input_hidden, biases_input_hidden)
# output_test = softmax(np.dot(hidden_output_test, weights_hidden_output) + biases_hidden_output)

# # Calculate accuracy
# predictions = np.argmax(output_test, axis=1)
# accuracy = np.mean(predictions == np.argmax(test_y, axis=1))
# print(f"Test accuracy: {accuracy * 100:.2f}%")

# # Visualize some predictions
# for i in range(9):
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(test_X[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
#     pyplot.title(f"Pred: {predictions[i]}")
# pyplot.show()














from keras import datasets
from matplotlib import pyplot
import numpy as np

# Loading the dataset
(train_X, train_y), (test_X, test_y) = datasets.mnist.load_data()

# Printing the shapes of the vectors
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))

# Visualizing some images
for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()

# ReLU Activation
class ReLU:
    def activate(self, x):
        return np.maximum(0, x)

# Convolution Layer
class Conv:
    def __init__(self, kernel):
        self.kernel = np.array(kernel)
        self.kernel_height, self.kernel_width = self.kernel.shape

    def apply_convolution(self, input_image):
        image_height, image_width = input_image.shape
        output_height = image_height - self.kernel_height + 1
        output_width = image_width - self.kernel_width + 1

        output_image = np.zeros((output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                region = input_image[i:i + self.kernel_height, j:j + self.kernel_width]
                output_image[i, j] = np.sum(region * self.kernel)

        return output_image

# MaxPooling Layer
class MaxPooling:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.pool_height, self.pool_width = pool_size

    def apply_max_pooling(self, feature_map):
        feature_map_height, feature_map_width = feature_map.shape
        output_height = feature_map_height // self.pool_height
        output_width = feature_map_width // self.pool_width

        output_feature_map = np.zeros((output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                region = feature_map[i * self.pool_height:(i + 1) * self.pool_height, 
                                     j * self.pool_width:(j + 1) * self.pool_width]
                output_feature_map[i, j] = np.max(region)

        return output_feature_map

# Linear Layer
class Linear:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros(output_size)

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias

# CNN Implementation
class CNN:
    def __init__(self):
        self.relu = ReLU()

        self.conv1 = Conv(kernel=np.random.randn(3, 3))
        self.pool1 = MaxPooling(pool_size=(2, 2))

        self.conv2 = Conv(kernel=np.random.randn(3, 3))
        self.pool2 = MaxPooling(pool_size=(2, 2))

        self.conv3 = Conv(kernel=np.random.randn(3, 3))
        self.pool3 = MaxPooling(pool_size=(2, 2))

        self.conv4 = Conv(kernel=np.random.randn(3, 3))
        self.linear = Linear(input_size=128, output_size=10)

    def forward(self, x):
        x = self.relu.activate(self.conv1.apply_convolution(x))
        x = self.pool1.apply_max_pooling(x)

        x = self.relu.activate(self.conv2.apply_convolution(x))
        x = self.pool2.apply_max_pooling(x)

        x = self.relu.activate(self.conv3.apply_convolution(x))
        x = self.pool3.apply_max_pooling(x)

        x = self.relu.activate(self.conv4.apply_convolution(x))

        x = x.flatten()
        x = self.linear.forward(x)
        return x

# Example Usage
if __name__ == "__main__":
    # Instantiate the CNN
    cnn = CNN()

    # Forward pass on the first image in the dataset
    output = cnn.forward(train_X[0])
    print("Output of the CNN:", output)







# import numpy as np

# class Conv:
#     def __init__(self, kernel):
#         """
#         Initializes the Conv class with a given kernel.
#         :param kernel: The convolutional kernel (filter) to apply.
#         """
#         self.kernel = kernel
#         self.kernel_height, self.kernel_width = kernel.shape
    
#     def apply_convolution(self, input_image):
#         """
#         Applies the convolution operation on the input image using the kernel.
#         :param input_image: The input image or feature map (2D numpy array).
#         :return: The convolved feature map.
#         """
#         image_height, image_width = input_image.shape
#         output_height = image_height - self.kernel_height + 1
#         output_width = image_width - self.kernel_width + 1

#         # Initialize the output feature map
#         output_image = np.zeros((output_height, output_width))

#         # Perform the convolution operation
#         for i in range(output_height):
#             for j in range(output_width):
#                 region = input_image[i:i + self.kernel_height, j:j + self.kernel_width]
#                 output_image[i, j] = np.sum(region * self.kernel)

#         return output_image


# class MaxPooling:
#     def __init__(self, pool_size):
#         """
#         Initializes the MaxPooling class with a given pooling window size.
#         :param pool_size: The size of the pooling window (e.g., 2x2).
#         """
#         self.pool_size = pool_size
#         self.pool_height, self.pool_width = pool_size
    
#     def apply_max_pooling(self, feature_map):
#         """
#         Applies the max pooling operation on the input feature map.
#         :param feature_map: The input feature map (2D numpy array).
#         :return: The downsampled feature map.
#         """
#         feature_map_height, feature_map_width = feature_map.shape
#         output_height = feature_map_height // self.pool_height
#         output_width = feature_map_width // self.pool_width

#         # Initialize the output feature map
#         output_feature_map = np.zeros((output_height, output_width))

#         # Perform the max pooling operation
#         for i in range(output_height):
#             for j in range(output_width):
#                 region = feature_map[i * self.pool_height:(i + 1) * self.pool_height, 
#                                      j * self.pool_width:(j + 1) * self.pool_width]
#                 output_feature_map[i, j] = np.max(region)

#         return output_feature_map


# # Example usage:

# # Define a sample kernel for convolution (e.g., a simple edge detection kernel)
# kernel = np.array([[1, 0, -1],
#                    [1, 0, -1],
#                    [1, 0, -1]])

# # Define a sample input image (5x5 matrix)
# input_image = np.array([[1, 2, 3, 4, 5],
#                         [6, 7, 8, 9, 10],
#                         [11, 12, 13, 14, 15],
#                         [16, 17, 18, 19, 20],
#                         [21, 22, 23, 24, 25]])

# # Create a Conv object and apply convolution
# conv_layer = Conv(kernel)
# convolved_image = conv_layer.apply_convolution(input_image)

# print("Convolved Image:")
# print(convolved_image)

# # Create a MaxPooling object (e.g., 2x2 pooling)
# max_pool_layer = MaxPooling(pool_size=(2, 2))
# pooled_image = max_pool_layer.apply_max_pooling(convolved_image)

# print("\nPooled Image:")
# print(pooled_image)
























