from sklearn.datasets import fetch_openml
from .utils import one_hot_encode
from .model import NN
mnist = fetch_openml(name="mnist_784")

print(mnist.keys())

#dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])

data = mnist.data
labels = mnist.target

train_test_split_no = 60000
X_train = data.values[:train_test_split_no].T
y_train = labels[:train_test_split_no].values.astype(int)
y_train = one_hot_encode(y_train, 10).T
X_test = data.values[train_test_split_no:].T
y_test = labels[train_test_split_no:].values.astype(int)
y_test = one_hot_encode(y_test, 10).T
X_train.shape, X_test.shape
((784, 60000), (784, 10000))

PARAMS = [X_train, y_train, X_test, y_test, "relu", 10, [128, 32]]
nn_relu = NN(*PARAMS)
epochs_relu = 200
lr_relu = 0.003
nn_relu.fit(X_train, y_train, lr=lr_relu, epochs=epochs_relu)