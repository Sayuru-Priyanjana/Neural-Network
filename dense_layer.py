import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01    # transpose of the weights matrix
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# X, Y = spiral_data(samples=100, classes=3)
# densel = Layer_Dense(2, 3)
# densel.forward(X)
# print(densel.output[:5])