import numpy as np  

def sigmoid (x):
    # activation function f(x) = 1 / (1 + e^(-x) )
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feed_forwared(self, inputs):
        # Weight the inputs, add bias and then use the simgoid function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)