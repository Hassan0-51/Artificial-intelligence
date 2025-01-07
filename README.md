# AI
# Subject Overview

This document provides a concise overview of key topics in Artificial Intelligence (AI), including **Logistic Regression**, **Random Forest**, **Traveling Salesperson Problem (TSP)**, and **Perceptron**. Each section explains the concept, its working principles, applications, and related algorithms.

---

## Logistic Regression

### **Overview**
Logistic Regression is a supervised machine learning algorithm used for binary classification problems. Despite its name, it is a classification algorithm, not a regression algorithm.

### **How It Works**
1. **Equation**: Logistic Regression uses a sigmoid function to map predicted values to probabilities:
   
   \[
   P(y=1|X) = \frac{1}{1 + e^{-z}}\]
   
   where \(z = w \cdot X + b\), and \(P(y=1|X)\) is the probability of the positive class.

2. **Decision Boundary**: A threshold (commonly 0.5) is used to classify predictions into two classes.

3. **Optimization**:
   - Loss function: Binary cross-entropy loss.
   - Gradient descent is used to minimize the loss and update weights.

### **Applications**
- Spam detection
- Disease diagnosis (e.g., predicting diabetes)
- Customer churn prediction

---

## Random Forest

### **Overview**
Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve classification and regression performance. It reduces overfitting and enhances generalization.

### **How It Works**
1. **Bootstrapping**:
   - Creates multiple subsets of the training data by sampling with replacement.
2. **Random Feature Selection**:
   - At each split, a random subset of features is considered.
3. **Tree Aggregation**:
   - For classification: Majority voting is used.
   - For regression: Averages the predictions of individual trees.

### **Applications**
- Fraud detection
- Medical diagnosis
- Stock market prediction

---

## Traveling Salesperson Problem (TSP)

### **Overview**
The Traveling Salesperson Problem (TSP) is an optimization problem where a salesperson must visit a set of cities exactly once and return to the starting point, minimizing the total travel distance or cost.

### **Approaches to Solve TSP**
1. **Exact Methods**:
   - Dynamic Programming
   - Branch and Bound
2. **Heuristic Methods**:
   - Nearest Neighbor
   - Minimum Spanning Tree
3. **Metaheuristic Methods**:
   - Genetic Algorithms
   - Simulated Annealing

### **Applications**
- Route optimization for delivery services
- Circuit design optimization
- Network routing

---

## Perceptron

### **Overview**
The Perceptron is one of the simplest types of artificial neural networks. It is primarily used for binary classification of linearly separable data.

### **How It Works**
1. **Structure**:
   - Input layer: Takes features as input.
   - Weights and bias: Assigns importance to each feature.

2. **Activation Function**:
   - A step function determines the output (0 or 1) based on a weighted sum:
     \[
     z = w \cdot X + b\]
     If \(z > 0\), the output is 1; otherwise, 0.

3. **Learning Rule**:
   - Adjust weights iteratively to minimize misclassification using:
     \[
     w := w + \alpha(y - \hat{y})X
     \]
     where \(\alpha\) is the learning rate.

### **Applications**
- Basic binary classification
- Initial building block for modern neural networks

---

## Summary Table
| **Algorithm**            | **Type**               | **Purpose**                      | **Key Features**                     |
|--------------------------|------------------------|----------------------------------|---------------------------------------|
| Logistic Regression      | Supervised Learning   | Binary Classification            | Sigmoid function, Cross-Entropy Loss |
| Random Forest            | Ensemble Learning     | Classification, Regression       | Bootstrapping, Aggregation           |
| Traveling Salesperson    | Optimization Problem  | Minimize travel cost/distance    | Exact and heuristic solutions        |
| Perceptron               | Neural Network        | Binary Classification            | Linear decision boundary             |

---

This document serves as a foundation for understanding these core AI concepts and their applications. For detailed implementations and examples, refer to the associated code or documentation provided in your course material.

