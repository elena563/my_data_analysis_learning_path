import numpy as np
from typing import List
from .utils import normalize

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
        

# forward propagation method
# backpropagation
# training
#cost function
# predict
# metrics