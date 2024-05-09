import numpy as np  
import ctypes

my_lib = ctypes.CDLL('./functions.so')

sigmoid = my_lib.sigmoid
sigmoid.my_c_function.argytypes = [ctypes.c_double]
sigmoid.my_c_function.restype = ctypes.c_double

deriv_sigmoid = my_lib.deriv_sigmoid
deriv_sigmoid.my_c_funcion.argytypes = [ctypes._double]
deriv_sigmoid.my_c_function.restype = ctypes.c_double

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feed_forwared(self, inputs):
        # Weight the inputs, add bias and then use the simgoid function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
    
class NeuralNetwork:
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
    
    def train(self, data, all_y_trues):
         learn_rate = 0.1
         epochs = 1000 #time that the loop goes through the entire dataset

         for epoch in range(epochs):
             for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w4 * x[0] + self.w5 * x[1] + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1
                 
                d_ypred = -2 (y_true - y_pred)

                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)
                    
                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)
 
                self.w1 -= learn_rate * d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_ypred * d_ypred_d_h1 * d_h1_d_b1

                self.w3 -= learn_rate * d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_ypred * d_ypred_d_h2 * d_h2_d_b2

                self.w5 -= learn_rate * d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_ypred * d_ypred_d_b3  
            