import neuronNetwork as nn
import numpy as np

network = nn.NeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))