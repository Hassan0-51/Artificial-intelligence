import numpy as np
import matplotlib.pyplot as plt

# Sigmoid Activation Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of Sigmoid
def sigmoid_derivative(z):
    return z * (1 - z)

# Perceptron Algorithm
def perceptron(X, y, learning_rate=0.1, epochs=10):
    weights = np.zeros(X.shape[1])
    bias = 0

    for _ in range(epochs):
        for i in range(len(y)):
            prediction = 1 if np.dot(X[i], weights) + bias > 0 else 0
            error = y[i] - prediction
            weights += learning_rate * error * X[i]
            bias += learning_rate * error

    return weights, bias

# Plot Decision Boundary for Perceptron
def plot_perceptron_boundary(X, y, weights, bias):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
    x_boundary = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_boundary = -(weights[0] * x_boundary + bias) / weights[1]
    plt.plot(x_boundary, y_boundary, color='green')
    plt.title("Perceptron Decision Boundary")
    plt.show()

# Neural Network for XOR Problem
def train_xor_nn(X, y, learning_rate=0.1, epochs=10000):
    np.random.seed(42)
    input_layer = 2
    hidden_layer = 2
    output_layer = 1

    # Initialize Weights and Biases
    weights_input_hidden = np.random.rand(input_layer, hidden_layer)
    weights_hidden_output = np.random.rand(hidden_layer, output_layer)
    bias_hidden = np.random.rand(hidden_layer)
    bias_output = np.random.rand(output_layer)

    for _ in range(epochs):
        # Forward Propagation
        hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)

        final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        final_output = sigmoid(final_input)

        # Backpropagation
        error = y - final_output
        d_output = error * sigmoid_derivative(final_output)

        error_hidden = d_output.dot(weights_hidden_output.T)
        d_hidden = error_hidden * sigmoid_derivative(hidden_output)

        # Update Weights and Biases
        weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
        weights_input_hidden += X.T.dot(d_hidden) * learning_rate
        bias_output += np.sum(d_output, axis=0) * learning_rate
        bias_hidden += np.sum(d_hidden, axis=0) * learning_rate

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Plot Decision Boundary for XOR
def visualize_xor_boundary(weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    hidden_input = np.dot(grid, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input).reshape(xx.shape)

    plt.contourf(xx, yy, final_output, levels=[0, 0.5, 1], cmap='bwr', alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
    plt.title("XOR Decision Boundary")
    plt.show()

# Datasets
# Perceptron Dataset
X_perceptron = np.array([[1, 1], [2, 3], [3, 3], [4, 1]])
y_perceptron = np.array([0, 0, 1, 1])

# XOR Dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Train Perceptron
weights, bias = perceptron(X_perceptron, y_perceptron)
plot_perceptron_boundary(X_perceptron, y_perceptron, weights, bias)

# Train XOR Neural Network
weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = train_xor_nn(X_xor, y_xor)
visualize_xor_boundary(weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
