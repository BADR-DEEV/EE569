import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((num_classes, len(y)))
    one_hot[y.astype(int), np.arange(len(y))] = 1
    return one_hot

class Conv:
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, image):
        kernel_height, kernel_width = self.kernel.shape
        image_height, image_width = image.shape
        output_height = image_height - kernel_height + 1
        output_width = image_width - kernel_width + 1
        output = np.zeros((output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                region = image[i:i + kernel_height, j:j + kernel_width]
                output[i, j] = np.sum(region * self.kernel)
        return output

class MaxPooling:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, feature_map):
        pool_height, pool_width = self.pool_size
        feature_height, feature_width = feature_map.shape
        output_height = feature_height // pool_height
        output_width = feature_width // pool_width
        output = np.zeros((output_height, output_width))

        for i in range(0, feature_height, pool_height):
            for j in range(0, feature_width, pool_width):
                region = feature_map[i:i + pool_height, j:j + pool_width]
                output[i // pool_height, j // pool_width] = np.max(region)
        return output


kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
image = np.random.rand(28, 28)
conv_layer = Conv(kernel)
convolved_image = conv_layer.forward(image)

pool_layer = MaxPooling((2, 2))
pooled_image = pool_layer.forward(convolved_image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Convolved Image")
plt.imshow(convolved_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Pooled Image")
plt.imshow(pooled_image, cmap='gray')

plt.show()




