import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
from activations import Activation_ReLU, Activation_Softmax
from dense_layer import Layer_Dense

X, Y = spiral_data(samples=100, classes=3)
densel1 = Layer_Dense(2,3)  # 2 inputs, 3 neurons
densel1.forward(X) # Forward pass through the first layer. contain result in output


activation1 = Activation_ReLU() # ReLU activation function
activation1.forward(densel1.output)  # Forward pass through activation functions.
# Activation function result is in 'activation1.output'.

densel2 = Layer_Dense(3,3)  # 3 inputs, 3 neurons
densel2.forward(activation1.output)  # Forward pass through the second layer.

activation2 = Activation_Softmax()  # Softmax activation function
activation2.forward(densel2.output)

print(activation2.output[:5])  # Print the first 5 output values of the softmax activation function.

