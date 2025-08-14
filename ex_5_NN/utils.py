import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0/(1.0 + np.exp(-z))

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def leaky_relu(z: np.ndarray) -> np.ndarray:
    return np.where(z>0, z, z*0.01)

def softmax(z: np.ndarray) -> np.ndarray:
    e=np.exp(z - np.max(z))
    return e/np.sum(e, axis=0)

# preprocessing
def normalize(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x) / np.max(x) - np.min(x))

def one_hot_encode(x: np.ndarray, num_labels: int) -> np.ndarray:
    return np.eye(num_labels)[x]


# derivative to perform gradient descent
def derivative(function_name: str, z:np.ndarray) -> np.ndarray:
    match function_name:
        case 'sigmoid':
            return sigmoid(z) * (1-sigmoid(z))
        case 'tanh':
            return 1 - np.square(tanh(z))
        case 'relu':
            y = (z > 0) * 1
            return y
        case 'leaky_relu':
            return np.where(z > 0, 1, 0.01)
        case _:
            'no such activation'