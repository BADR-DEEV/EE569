import numpy as np
import matplotlib.pyplot as plt

def generate_data(size=25, noise_level=0.2):
    X = np.linspace(0, 2 * np.pi, size)
    y_true = np.sin(X)
    noise = np.random.normal(0, noise_level, size)
    y_noisy = y_true + noise
    return X, y_noisy, y_true

def split_data(X, y, test_size=0.4):
    split_idx = int(len(X) * (1 - test_size))
    indices = np.random.permutation(len(X))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def polynomial_features(X, degree):
    return np.vander(X, degree + 1, increasing=True)

def fit_polynomial(X, y, l=0):
    return np.linalg.inv(X.T.dot(X) + l * np.eye(X.shape[1])).dot(X.T).dot(y)

def predict(X, coeffs):
    return X @ coeffs

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def plot_data(X_train, y_train, X_test, y_test, degrees, plot_degrees, l=0):
    colors = ['blue', 'green']
    labels = ['Training Data', 'Testing Data']
    train_errors, test_errors = [], []

    plt.figure(figsize=(10, 6))
    X_2 = np.linspace(0, 2 * np.pi, len(X_train) * 10)
    plt.plot(X_2, np.sin(X_2), label='True Function', color='red')

    for color, (X, y), label in zip(colors, [(X_train, y_train), (X_test, y_test)], labels):
        plt.scatter(X, y, color=color, label=label)

        for degree in degrees:
            X_poly_train = polynomial_features(X_train, degree)
            coeffs = fit_polynomial(X_poly_train, y_train, l)

            X_plot_poly = polynomial_features(X_2, degree)
            y_plot_poly = predict(X_plot_poly, coeffs)

            if label == "Training Data" and degree in plot_degrees:
                plt.plot(X_2, y_plot_poly, label=f'Degree {degree} Fit ({label})')

            if label == 'Training Data':
                train_pred = predict(X_poly_train, coeffs)
                train_error = mean_squared_error(y_train, train_pred)
                train_errors.append(train_error)
                print(f"Degree {degree} Train Error: {train_error:.4f}")
            else:
                X_poly_test = polynomial_features(X_test, degree)
                test_pred = predict(X_poly_test, coeffs)
                test_error = mean_squared_error(y_test, test_pred)
                test_errors.append(test_error)
                print(f"Degree {degree} Test Error: {test_error:.4f}")

    plt.xlim(0, 2 * np.pi)
    plt.ylim(-1.2, 1.2)
    plt.legend()
    plt.title('Sin Wave with Noisy Samples and Polynomial Fits')
    plt.show()

    return train_errors, test_errors

def plot_errors(degrees, train_errors, test_errors):
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_errors, label='Train Error', marker='o')
    plt.plot(degrees, test_errors, label='Test Error', marker='o')
    plt.ylim(0, 1)
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Mean Squared Error')
    plt.title('Training and Testing Errors by Polynomial Degree')
    plt.legend()
    plt.show()

def main():
    num_samples = 24
    l = 0
    X, y_noisy, _ = generate_data(num_samples, 0.3)
    X_train, X_test, y_train, y_test = split_data(X, y_noisy, test_size=0.4)

    degrees = list(range(1, num_samples // 2))
    plot_degrees = [3,9]

    train_errors, test_errors = plot_data(X_train, y_train, X_test, y_test, degrees, plot_degrees,l)
    plot_errors(degrees, train_errors, test_errors)

if __name__ == "__main__":
    main()
