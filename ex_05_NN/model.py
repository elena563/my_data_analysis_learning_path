import numpy as np
from typing import List
from .utils import normalize, softmax

class NN():
    def __init__(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, activation: str, num_labels: int, architecture: List[int]):
        self.X = normalize(X)
        assert np.all((self.X >= 0) | (self.X <= 1))

        self.X, self.X_test = X.copy(), X_test.copy()
        self.y, self.y_test = y.copy(), y_test.copy()
        self.layers = {}    # results of activation
        self.architecture = architecture    # size of hidden layers
        self.activation = activation
        self.parameters = {}
        self.num_input_features = X.shape[0]
        self.num_labels = num_labels
        self.m = X.shape[1]
        self.architecture.append(self.num_input_features)
        self.architecture.append(self.num_labels)
        self.L = len(architecture)
        assert self.X.shape == (self.num_input_features, self.m)
        assert self.y.shape == (self.num_labels, self.m)


    # parameters initialization: for each deep layer, add to par dict weights and biases
    # weights are randomly picked from normal sample
    # biases are vectors of zeros

    def initialize_parameters(self):
        for i in range(1, self.L):
            print(f"Layer: {i}")
            self.parameters['w'+str(i)] = np.random.randn(self.architecture[i], self.architecture[i-1])*0.01
            self.parameters['b'+str(i)] = np.zeros((self.architecture[i], 1))
        

    # forward propagation method: no inputs, cost value as output

    def forward(self):
        params = self.parameters
        self.layers['a0'] = self.X
        for l in range(1, self.L-1):
            self.layers['z' + str(l)] = np.dot(params['w' + str(l)], self.layers['a' + str(l-1)]) + params['b' + str(l)]
            self.layers['a' + str(l)] = eval(self.activation)(self.layers['z' + str(l)])
            assert self.layers['a' + str(l)].shape == (self.architecture[l], self.m)
        self.layers['z' + str(self.L-1)] = np.dot(params['w' + str(self.L-1)], self.layers['a' + str(self.L-2)]) + params['b' + str(self.L-1)]
        self.layers['a' + str(self.L-1)] = softmax(self.layers['z' + str(self.L-1)])
        self.output = self.layers['a' + str(self.L-1)]
        assert self.output.shape == (self.num_labels, self.m)
        assert all([s for s in np.sum(self.output, axis=1)])

        cost = - np.sum(self.y * np.log(self.output + 0.000000001))
        return cost, self.layers


# backpropagation
# training
#cost function
# predict
# metrics