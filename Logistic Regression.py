import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    print ('sigmoid', 1 / (1 + np.exp(-z)))
    """
    Compute the sigmoid of z.
    Maps real values to the range [0, 1].
    """
    return 1 / (1 + np.exp(-z))

# Cross-Entropy Loss
def cross_entropy_loss(y_true, y_pred):
    """
    Compute binary cross-entropy loss.
    Measures how close predicted probabilities are to actual labels.
    """
    epsilon = 1e-15  # To prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    print('cel', loss)
    return loss

# Gradient Descent
def gradient_descent(X, y, weights, learning_rate, iterations):
    """
    Perform gradient descent to optimize weights.
    """
    losses = []
    m = len(y)  # Number of samples

    for i in range(iterations):
        # Compute predictions
        z = np.dot(X, weights)
        y_pred = sigmoid(z)

        # Calculate loss
        loss = cross_entropy_loss(y, y_pred)
        losses.append(loss)

        # Gradient calculation
        gradient = np.dot(X.T, (y_pred - y)) / m

        # Update weights
        weights -= learning_rate * gradient

        # Debugging output for weights and loss
        if i % 100 == 0:
            print(f"Learning Rate: {learning_rate}, Iteration {i}, Loss: {loss:.4f}")

    print("gd", weights, losses)
    return weights, losses

# Prediction function
def predict(X, weights):
    """
    Predict labels (0 or 1) using sigmoid probabilities and threshold of 0.5.
    """
    probabilities = sigmoid(np.dot(X, weights))
    print ('predict', (probabilities >= 0.5).astype(int))
    return (probabilities >= 0.5).astype(int)

# Logistic Regression Model
def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    """
    Train a logistic regression model using gradient descent.
    """
    # Initialize weights to zeros
    weights = np.zeros(X.shape[1])

    # Optimize weights using gradient descent
    weights, losses = gradient_descent(X, y, weights, learning_rate, iterations)
    print ("lr", weights, losses)
    return weights, losses

# Evaluation Function
def evaluate(y_true, y_pred):
    """
    Evaluate model accuracy as the percentage of correct predictions.
    """
    accuracy = np.mean(y_true == y_pred) * 100
    return accuracy

def main():
    # Data Preprocessing
    data = np.array([
        [0.1, 1.1, 0], [1.2, 0.9, 0], [1.5, 1.6, 1], [2.0, 1.8, 1],
        [2.5, 2.1, 1], [0.5, 1.5, 0], [1.8, 2.3, 1], [0.2, 0.7, 0],
        [1.9, 1.4, 1], [0.8, 0.6, 0]
    ])

    # Features (X) and Target (y)
    X = data[:, :2]
    y = data[:, 2]

    # Standardize features (mean = 0, std = 1)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std

    # Add a bias column (intercept term) to X
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Test with multiple learning rates
    learning_rates = [0.1]
    iterations = 10

    # Loop through different learning rates
    for lr in learning_rates:
        print(f"\nTesting with Learning Rate: {lr}")
        weights, losses = logistic_regression(X, y, learning_rate=lr, iterations=iterations)

        # Predict and Evaluate
        y_pred = predict(X, weights)
        accuracy = evaluate(y, y_pred)
        print(f"Learning Rate: {lr}, Model Accuracy: {accuracy:.2f}%")

        # Visualize Loss
        plt.plot(losses, label=f"LR: {lr}")

    # Plot loss for different learning rates
    plt.title("Loss Over Iterations for Different Learning Rates")
    plt.xlabel("Iterations")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.show()

    # Visualize Decision Boundary for the last learning rate
    x1_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x2_range = -(weights[0] + weights[1] * x1_range) / weights[2]
    plt.scatter(X[y == 0, 1], X[y == 0, 2], color='red', label='Class 0')
    plt.scatter(X[y == 1, 1], X[y == 1, 2], color='blue', label='Class 1')
    plt.plot(x1_range, x2_range, color='green', label='Decision Boundary')
    plt.legend()
    plt.xlabel("X1 (Standardized)")
    plt.ylabel("X2 (Standardized)")
    plt.title("Decision Boundary")
    plt.show()
main()