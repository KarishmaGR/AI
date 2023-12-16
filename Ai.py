#++++++++++++++++++++++++++++++++++++++++1st practical  =   Breadth First Search ++++++++++++++++++++++++++++++++++++++====
from queue import Queue

romaniaMap = {
    'Arad': ['Sibiu', 'Zerind', 'Timisoara'],
    'Zerind': ['Arad', 'Oradea'],
    'Oradea': ['Zerind', 'Sibiu'],
    'Sibiu': ['Arad', 'Oradea', 'Fagaras', 'Rimnicu'],
    'Timisoara': ['Arad', 'Lugoj'],
    'Lugoj': ['Timisoara', 'Mehadia'],
    'Mehadia': ['Lugoj', 'Drobeta'],
    'Drobeta': ['Mehadia', 'Craiova'],
    'Craiova': ['Drobeta', 'Rimnicu', 'Pitesti'],
    'Rimnicu': ['Sibiu', 'Craiova', 'Pitesti'],
    'Fagaras': ['Sibiu', 'Bucharest'],
    'Pitesti': ['Rimnicu', 'Craiova', 'Bucharest'],
    'Bucharest': ['Fagaras', 'Pitesti', 'Giurgiu', 'Urziceni'],
    'Giurgiu': ['Bucharest'],
    'Urziceni': ['Bucharest', 'Vaslui', 'Hirsova'],
    'Hirsova': ['Urziceni', 'Eforie'],
    'Eforie': ['Hirsova'],
    'Vaslui': ['Iasi', 'Urziceni'],
    'Iasi': ['Vaslui', 'Neamt'],
    'Neamt': ['Iasi']
}

def bfs(startingNode, destinationNode):
    visited = {}
    distance = {}
    parent = {}
    bfs_traversal_output = []
    queue = Queue()

    for city in romaniaMap.keys():
        visited[city] = False
        parent[city] = None
        distance[city] = -1

    startingCity = startingNode
    visited[startingCity] = True
    distance[startingCity] = 0
    queue.put(startingCity)

    while not queue.empty():
        u = queue.get()
        bfs_traversal_output.append(u)

        for v in romaniaMap[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                distance[v] = distance[u] + 1
                queue.put(v)

    g = destinationNode
    path = []
    while g is not None:
        path.append(g)
        g = parent[g]

    path.reverse()
    return path

# Starting City & Destination City
path_to_bucharest = bfs('Arad', 'Bucharest')
print(path_to_bucharest)






#++++++++++++++++++++++++++++++++++++++++++++ 2nd practical   =  Recursive Best First Search +++++++++++++++++++++++++++++++++++++
import sys

romania_map = {
    'Arad': {'Zerind': 75, 'Timisoara': 118, 'Sibiu': 140},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Rimnicu Vilcea': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
    'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
    'Drobeta': {'Mehadia': 75, 'Craiova': 120},
    'Craiova': {'Drobeta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101}
}

heuristics = {
    'Arad': 366,
    'Zerind': 374,
    'Timisoara': 329,
    'Sibiu': 253,
    'Oradea': 380,
    'Lugoj': 244,
    'Fagaras': 176,
    'Rimnicu Vilcea': 193,
    'Mehadia': 241,
    'Drobeta': 242,
    'Craiova': 160,
    'Pitesti': 100,
    'Bucharest': 0
}

def rbfs_search(start, goal, path, f_limit):
    if start == goal:
        return path

    successors = romania_map[start]

    if len(successors) == 0:
        return None

    sorted_successors = sorted(successors, key=lambda x: successors[x] + heuristics[x])
    for city in sorted_successors:
        new_path = path + [city]
        f_value = successors[city] + heuristics[city]
        if f_value > f_limit:
            return None
        result = rbfs_search(city, goal, new_path, min(f_limit, f_value))
        if result is not None:
            return result
    return None

def recursive_best_first_search(start, goal):
    f_limit = sys.maxsize
    path = [start]
    while True:
        result = rbfs_search(start, goal, path, f_limit)
        if result is not None:
            return result
        f_limit = sys.maxsize

# Example usage:
start_city = 'Arad'
goal_city = 'Bucharest'
path = recursive_best_first_search(start_city, goal_city)
if path is None:
    print("Path not found!")
else:
    print("Path:", path)
    print("Cost:", sum(romania_map[path[i]][path[i + 1]] for i in range(len(path) - 1)))







#+++++++++++++++++++++++++++++++++++++++++++++++++++ 3rd practical   == A* Search Algorithm ++++++++++++++++++++++++++++++++++++++++++++
romania_graph = {
    'Arad': [('Zerind', 75), ('Timisoara', 118), ('Sibiu', 140)],
    'Zerind': [('Arad', 75), ('Oradea', 71)],
    'Timisoara': [('Arad', 118), ('Lugoj', 111)],
    'Sibiu': [('Arad', 140), ('Oradea', 151), ('Fagaras', 99), ('Rimnicu Vilcea', 80)],
    'Oradea': [('Zerind', 71), ('Sibiu', 151)],
    'Lugoj': [('Timisoara', 111), ('Mehadia', 70)],
    'Fagaras': [('Sibiu', 99), ('Bucharest', 211)],
    'Rimnicu Vilcea': [('Sibiu', 80), ('Craiova', 146), ('Pitesti', 97)],
    'Mehadia': [('Lugoj', 70), ('Drobeta', 75)],
    'Drobeta': [('Mehadia', 75), ('Craiova', 120)],
    'Craiova': [('Drobeta', 120), ('Rimnicu Vilcea', 146), ('Pitesti', 138)],
    'Pitesti': [('Rimnicu Vilcea', 97), ('Craiova', 138), ('Bucharest', 101)],
    'Bucharest': [('Fagaras', 211), ('Pitesti', 101)],
}

heuristic = {
    'Arad': 366,
    'Bucharest': 0,
    'Craiova': 160,
    'Drobeta': 242,
    'Fagaras': 178,
    'Lugoj': 244,
    'Mehadia': 241,
    'Oradea': 380,
    'Pitesti': 98,
    'Rimnicu Vilcea': 193,
    'Sibiu': 253,
    'Timisoara': 329,
    'Zerind': 374,
}

def a_star(start, goal):
    open_set = [(0 + heuristic[start], start)]  # Priority queue (f_score, node)
    came_from = {}  # Keep track of the optimal path
    g_score = {city: float('inf') for city in romania_graph}
    g_score[start] = 0

    while open_set:
        current_f_score, current_city = min(open_set)
        open_set.remove((current_f_score, current_city))

        if current_city == goal:
            path = [goal]
            while current_city in came_from:
                current_city = came_from[current_city]
                path.append(current_city)
            return path[::-1]

        for neighbor, cost in romania_graph[current_city]:
            tentative_g_score = g_score[current_city] + cost

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_city
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic[neighbor]
                open_set.append((f_score, neighbor))

    return None  # Path not found

# Example usage:
start_city = 'Arad'
goal_city = 'Bucharest'
path = a_star(start_city, goal_city)

if path is None:
    print("Path not found!")
else:
    print("Optimal Path:", path)




# ++++++++++++++++++++++++++++++++++++++++++4rth practical  = Iterative Depth First Search++++++++++++++++++++++++++==========================
romania_map = {
    'Arad': ['Zerind', 'Sibiu', 'Timisoara'],
    'Zerind': ['Arad', 'Oradea'],
    'Oradea': ['Zerind', 'Sibiu'],
    'Sibiu': ['Arad', 'Oradea', 'Fagaras', 'Rimnicu Vilcea'],
    'Timisoara': ['Arad', 'Lugoj'],
    'Lugoj': ['Timisoara', 'Mehadia'],
    'Mehadia': ['Lugoj', 'Drobeta'],
    'Drobeta': ['Mehadia', 'Craiova'],
    'Craiova': ['Drobeta', 'Rimnicu Vilcea', 'Pitesti'],
    'Rimnicu Vilcea': ['Sibiu', 'Craiova', 'Pitesti'],
    'Fagaras': ['Sibiu', 'Bucharest'],
    'Pitesti': ['Rimnicu Vilcea', 'Craiova', 'Bucharest'],
    'Bucharest': ['Fagaras', 'Pitesti', 'Giurgiu', 'Urziceni'],
    'Giurgiu': ['Bucharest'],
    'Urziceni': ['Bucharest', 'Hirsova', 'Vaslui'],
    'Hirsova': ['Urziceni', 'Eforie'],
    'Eforie': ['Hirsova'],
    'Vaslui': ['Urziceni', 'Iasi'],
    'Iasi': ['Vaslui', 'Neamt'],
    'Neamt': ['Iasi']
}

def iterative_deepening_dfs(start, goal, depth_limit):
    for depth in range(depth_limit + 1):
        visited = set()
        result = depth_limited_dfs(start, goal, depth, visited)
        if result is not None:
            return result
    return None

def depth_limited_dfs(node, goal, depth, visited):
    if node == goal:
        return [node]
    if depth == 0:
        return None
    visited.add(node)
    for neighbor in romania_map[node]:
        if neighbor not in visited:
            result = depth_limited_dfs(neighbor, goal, depth - 1, visited)
            if result is not None:
                return [node] + result
    return None

start_city = 'Arad'
goal_city = 'Bucharest'
depth_limit = 10
path = iterative_deepening_dfs(start_city, goal_city, depth_limit)

if path is not None:
    print("Path found:", ' -> '.join(path))
else:
    print("Path not found within depth limit.")



#+++++++++++++++++++++++++++++++++++++++++++++ 5th practical   == Adaboost Ensemble Learning ++++++++++++++++++++++++++++++++++++++++++++
import pandas
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)


array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]


seed = 7
num_trees = 30
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)


results = model_selection.cross_val_score(model, X, Y)

print("Mean Accuracy:", results.mean())

# Theory:
# • AdaBoost algorithm, short for Adaptive Boosting, is a boosting technique used as
# an Ensemble Method in machine learning.
# • It is called Adaptive Boosting as the weights are re-assigned to each instance,
# with higher weights assigned to incorrectly classified instances.
# • Boosting is used to reduce bias as well as variance for supervised learning.
# • It works on the principle of learners growing sequentially. Except for the first,
# each subsequent learner is grown from previously grown learners.
# • Weak learners are converted into strong ones. The AdaBoost algorithm works on
# the same principle as boosting with a slight difference.



# +++++++++++++++++++++++++++++++++++++Practical 6 ==  Decision Tree Learning Algorihtm++++++++++++++++++++++++++++++++
import numpy as np
import pandas as pd
import os

PlayTennis = pd.read_csv("C:\\Users\\karis\\Downloads\\PlayTennis.csv")
PlayTennis

from sklearn.preprocessing import LabelEncoder
Le =LabelEncoder()
PlayTennis["Outlook"] = Le.fit_transform(PlayTennis["Outlook"])
PlayTennis["Temperature"] = Le.fit_transform(PlayTennis["Temperature"])
PlayTennis["Humidity"] = Le.fit_transform(PlayTennis["Humidity"])
PlayTennis["Wind"] = Le.fit_transform(PlayTennis["Wind"])
PlayTennis["PlayTennis"] = Le.fit_transform(PlayTennis["PlayTennis"])
PlayTennis

from sklearn import tree
import matplotlib.pyplot as plt

y = PlayTennis['PlayTennis']
X = PlayTennis.drop(['PlayTennis'], axis=1)
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, y)
tree.plot_tree(clf)

import graphviz
dot_data = tree.export_graphviz(clf,out_file=None)
graph = graphviz.Source(dot_data)
graph

x_pred = clf.predict(X)
x_pred==y



# Theory:
# • A Decision Tree is a supervised machine learning algorithm used for both
# classification and regression tasks.
# • It's a tree-like model that makes decisions based on a series of conditions or
# rules learned from the training data.
# • Each internal node in the tree represents a decision rule based on a feature,
# and each leaf node represents a class label (in classification) or a predicted
# value (in regression).
# • Decision trees are characterized by their simplicity, interpretability, and ability to
# handle both categorical and numerical data.
# Decision trees work in the following way
# 1. Root Node: At the root of the tree, the algorithm selects the feature that best
# separates the data based on a criterion (e.g., Gini impurity or entropy for classification,
# mean squared error for regression). This feature becomes the root node.
# 2. Internal Nodes: The algorithm recursively selects features to split the data into
# subsets at each internal node. These splits are determined to minimize impurity (for
# classification) or reduce error (for regression).
# 3. Leaf Nodes: The process continues until a stopping criterion is met, such as a
# maximum depth of the tree or a minimum number of data points in a node. The final
# nodes are called leaf nodes and represent the predicted class or value.



#+++++++++++++++++++++++++++++++=== Practical  7   Knn_classification   +++++++++++++++++++=================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('Iris.csv')
X = df.drop(['variety'], axis=1).values
y = df['variety'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
k = 3
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
classification_accuracy = accuracy_score(y_test, y_pred)
print("Classification Accuracy:", classification_accuracy)


# Theory:
# K-Nearest Neighbors (K-NN) is a simple but powerful machine learning algorithm used
# for both classification and regression tasks.
# 1. Intuition: K-NN is based on the idea that objects (data points) that are close to each
# other in a feature space are more likely to belong to the same class or have similar
# values (For regression).
# 2. How it works:
# Classification: Given a new data point, K-NN finds the K nearest data points in the
# training dataset and assigns the class label that is most common among those K
# neighbors to the new point.
# Regression: For regression tasks, K-NN calculates the average (or another
# aggregation) of the target values of the K nearest neighbors and assigns this value to
# the new data point.
# 3. Hyperparameter K: The choice of the hyperparameter K (the number of neighbors to
# consider) is critical. A small K may lead to a noisy model (sensitive to outliers), while a
# large K may lead to a biased model (smoothing over variations in the data). K is
# typically an odd number to avoid ties in voting.
# 4. Distance Metric: K-NN uses a distance metric (e.g., Euclidean distance, Manhattan
# distance, etc.) to measure the similarity between data points. The choice of distance
# metric should be appropriate for your data and problem.
# 5. Scaling Features: It's important to scale features before applying K-NN, especially
# when using distance-based metrics, to ensure that all features have equal influence on
# the results.
# 6. Pros:
# • Simple and easy to understand.
# • No assumptions about the data distribution.
# • Works well for both classification and regression tasks.
# • Non-parametric (does not make assumptions about the functional form of
# relationships).
# 7. Cons:
# • Can be computationally expensive, especially for large datasets.
# • Sensitive to the choice of K and the distance metric.
# • Requires a sufficient amount of training data.
# • May not perform well when the feature space is high-dimensional.






#++++++++++++++++++++++++++++++++++++++++++++++++ 8Th Practical  = Feed Forward Back Propagation Neural Network +++++++++++++++++++++++++++++

import numpy as np

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with random values
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))

    def forward(self, inputs):
        # Forward propagation
        self.hidden_input = np.dot(inputs, self.weights_input_hidden)
        self.hidden_output = sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.predicted_output = sigmoid(self.output_input)
        return self.predicted_output

    def backward(self, inputs, target, learning_rate):
        # Backpropagation
        error = target - self.predicted_output
        delta_output = error * sigmoid_derivative(self.predicted_output)
        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * sigmoid_derivative(self.hidden_output)
        
        # Update weights
        self.weights_hidden_output += np.outer(self.hidden_output, delta_output) * learning_rate
        self.weights_input_hidden += np.outer(inputs, delta_hidden) * learning_rate

    def train(self, training_data, targets, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(len(training_data)):
                inputs = training_data[i]
                target = targets[i]
                self.forward(inputs)
                self.backward(inputs, target, learning_rate)

    def predict(self, inputs):
        return self.forward(inputs)

# Define XOR dataset
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Create and train the neural network
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1
epochs = 10000

nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(training_data, targets, epochs, learning_rate)

# Test the trained network
for i in range(len(training_data)):
    inputs = training_data[i]
    prediction = nn.predict(inputs)
    print(f"Input: {inputs}, Predicted Output: {prediction}")



# Theory:
# • Feedforward neural networks, also known as feedforward neural networks
# (FNNs) or multilayer perceptrons (MLPs), are a fundamental type of artificial
# neural network used in machine learning and deep learning.
# • They are designed to model complex relationships between inputs and outputs
# by stacking multiple layers of interconnected neurons (also called nodes or
# units).
# Here are the key characteristics of feedforward neural networks:
# 1. Feedforward Structure: FNNs have a strict feedforward structure, meaning
# information flows in one direction, from the input layer through one or more hidden
# layers to the output layer. There are no feedback loops or recurrent connections in this
# architecture.
# 2. Layers: An FNN typically consists of three main types of layers:
# Input Layer: This layer contains neurons that represent the features or input data. Each
# neuron corresponds to a specific input feature.
# Hidden Layers: These intermediate layers, which can be one or more, perform
# complex transformations on the input data. Each neuron in a hidden layer is connected
# to all neurons in the previous layer and feeds its output to the next layer.
# Output Layer: The final layer produces the network's predictions or outputs. The
# number of neurons in the output layer depends on the problem; for regression tasks, it
# may be one neuron, while for classification tasks, it can be one neuron per class.
# Feedforward neural networks are a foundational component of deep learning, allowing
# machines to learn and make predictions from data by forming increasingly abstract and
# hierarchical representations through multiple layers of interconnected neuron




#+++++++++++++++++++++++++++++++++++++++++++++ Practical 9  Implement the SVM algorithm for binary classification.++++++++++++++++++++++++++++
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('Iris.csv')

X = data.drop('variety', axis=1)
y = data['variety']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_classifier = SVC(kernel='linear')  

svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# Theory:
# Support Vector Machines (SVM) are a powerful and versatile class of supervised
# machine learning algorithms used for classification and regression tasks. Here's some
# essential theory about SVMs:
# 1. Linear Separability: SVMs are primarily designed for binary classification problems.
# They work by finding the optimal hyperplane that best separates two classes in the
# feature space. This hyperplane is chosen to maximize the margin between the two
# classes. When the classes are linearly separable, SVMs can find the hyperplane with
# the maximum margin.
# 2. Margin: The margin is the distance between the hyperplane and the nearest data
# point from either class. SVM aims to maximize this margin because a larger margin
# generally indicates a better separation and better generalization to unseen data.
# 3. Support Vectors: Support vectors are the data points that are closest to the
# hyperplane and directly influence its position and orientation. These are the critical data
# points that determine the margin. The SVM algorithm focuses on these support vectors
# during training.
# 4. Kernel Trick: SVMs can be extended to handle non-linearly separable data by using a
# kernel function. A kernel function transforms the original feature space into a higherdimensional space, where the data may become linearly separable. Common kernel
# functions include the linear, polynomial, radial basis function (RBF/Gaussian), and
# sigmoid kernels.
# 5. C Parameter: The C parameter is a regularization parameter in SVM that balances
# the trade-off between maximizing the margin and minimizing classification errors. A
# smaller C value results in a larger margin but may allow some training points to be
# misclassified, while a larger C value tries to classify all training points correctly but may
# result in a smaller margin.
