import math
from collections import Counter
import random

def calculate_entropy(data, target_col):
    target_values = [row[target_col] for row in data]
    freq = Counter(target_values)
    total = len(data)
    return -sum((count / total) * math.log2(count / total) for count in freq.values())

def calculate_information_gain(data, attribute, target_col):
    total_entropy = calculate_entropy(data, target_col)
    attr_values = [row[attribute] for row in data]
    subsets = {value: [row for row in data if row[attribute] == value] for value in set(attr_values)}
    weighted_entropy = sum((len(subset) / len(data)) * calculate_entropy(subset, target_col) for subset in subsets.values())
    return total_entropy - weighted_entropy

def build_tree(data, attributes, target_col, depth=0, max_depth=3):
    target_values = [row[target_col] for row in data]
    if len(set(target_values)) == 1 or depth == max_depth or not attributes:
        return Counter(target_values).most_common(1)[0][0]
    best_attr = max(attributes, key=lambda attr: calculate_information_gain(data, attr, target_col))
    tree = {best_attr: {}}
    attr_values = set(row[best_attr] for row in data)
    remaining_attrs = [attr for attr in attributes if attr != best_attr]
    for value in attr_values:
        subset = [row for row in data if row[best_attr] == value]
        tree[best_attr][value] = build_tree(subset, remaining_attrs, target_col, depth + 1, max_depth)
    return tree

def predict(tree, data_point):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    value = data_point[attr]
    return predict(tree[attr][value], data_point)

def build_random_forest(data, attributes, target_col, n_trees=2):
    trees = []
    for _ in range(n_trees):
        sampled_data = random.choices(data, k=len(data))
        tree = build_tree(sampled_data, attributes, target_col)
        trees.append(tree)
    return trees

def predict_forest(forest, data_point):
    predictions = [predict(tree, data_point) for tree in forest]
    return Counter(predictions).most_common(1)[0][0]

data = [
    {"Weather": "Sunny", "Temperature": "Hot", "Play?": "No"},
    {"Weather": "Overcast", "Temperature": "Hot", "Play?": "Yes"},
    {"Weather": "Rainy", "Temperature": "Mild", "Play?": "Yes"},
    {"Weather": "Sunny", "Temperature": "Mild", "Play?": "No"},
    {"Weather": "Overcast", "Temperature": "Mild", "Play?": "Yes"},
    {"Weather": "Rainy", "Temperature": "Hot", "Play?": "No"}
]

attributes = ["Weather", "Temperature"]
target_col = "Play?"
forest = build_random_forest(data, attributes, target_col)
test_point = {"Weather": "Sunny", "Temperature": "Mild"}
prediction = predict_forest(forest, test_point)
print(f"Prediction for {test_point}: {prediction}")
