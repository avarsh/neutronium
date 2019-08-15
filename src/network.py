"""
network.py
==========

Defines a neural network.
"""

import numpy as np
import random

from matplotlib import pyplot as plt

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))

class Network:

    class Layer:
        def __init__(self, n, prev):
            self.n = n
            self.prev = prev
            self.next = None

            self.b = None 
            self.w = None
            self.z = None
            self.a = None

            if not self.is_input():
                self.prev.next = self
                # Each layer owns the weights coming into it!
                self.w = np.random.randn(self.n, self.prev.n)
                self.b = np.random.randn(self.n, 1)
        
            self.delta_err = None
            
        def is_input(self):
            return (self.prev == None)

        def is_output(self):
            return (self.next == None)

        def feedforward(self, x):
            if self.is_input():
                self.a = x
                return self.next.feedforward(x)
            
            self.z = np.dot(self.w, x) + self.b
            self.a = sigmoid(self.z)

            if self.is_output():
                return self.a
            else:
                return self.next.feedforward(self.a)
        
        def backprop(self, cost_derivative, y, del_C_b, del_C_w):
            if self.is_input():
                return None
            
            del_C_b.insert(0, np.zeros(self.b.shape))
            del_C_w.insert(0, np.zeros(self.w.shape))

            sig_prime = sigmoid_prime(self.z)

            if self.is_output():
                self.delta_err = np.multiply(cost_derivative(self.a, y),
                                             sig_prime)
            else:
                nabla_C = np.dot(self.next.w.transpose(), self.next.delta_err)
                self.delta_err = nabla_C * sig_prime
            
            del_C_b[0] = self.delta_err
            del_C_w[0] = np.dot(self.delta_err, self.prev.a.transpose())

            self.prev.backprop(cost_derivative, y, del_C_b, del_C_w)


    def __init__(self, sizes):
        self.layers = []
        self.num_layers = len(sizes)
        prev = None
        for size in sizes:
            layer = self.Layer(size, prev)
            self.layers.append(layer)
            prev = layer
        
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]
    
    def get_output(self, x):
        return self.input_layer.feedforward(x)
    
    def grad_desc(self, training_data, epochs, learning_rate):
        """
        Trains the NN by using gradient descent, modifying weights after
        each training example.

        Test data is a list of tuples, mapping training inputs to
        the correct output.
        """

        m = len(training_data)

        epoch_list = []
        cost_list = []

        for j in range(epochs):
            del_b = []
            del_w = []

            for x, y in training_data:
                self.input_layer.feedforward(x)
                self.output_layer.backprop(self.cost_derivative, y, del_b, del_w)

                for l in range(1, self.num_layers):
                    self.layers[l].b = self.layers[l].b - \
                                        (learning_rate / m) * del_b[l-1]
                    self.layers[l].w = self.layers[l].w - \
                                        (learning_rate / m) * del_w[l-1]
            
            # Calculate average cost
            epoch_list.append(j)
            cost_list.append(self.cost(training_data))

        plt.plot(epoch_list, cost_list)
        plt.show()

    def cost(self, eval_data):
        total = 0
        for x, y in eval_data:
            total = total + pow((y - self.get_output(x)), 2)
        
        total = total / 2
        return total[0]
    
    def cost_derivative(self, actual, expected):
        return (actual - expected)


if __name__ == '__main__':
    network = Network([2, 4, 1])
    xor_data = [
        (np.array([[1], [1]]), np.array([0])),
        (np.array([[1], [0]]), np.array([1])),
        (np.array([[0], [1]]), np.array([1])),
        (np.array([[0], [0]]), np.array([0]))
    ]
    network.grad_desc(xor_data, 10000, 0.5)

    print("1 xor 1 =", network.get_output(np.array([[1], [1]])))
    print("0 xor 1 =", network.get_output(np.array([[0], [1]])))
    print("1 xor 0 =", network.get_output(np.array([[1], [0]])))
    print("0 xor 0 =", network.get_output(np.array([[0], [0]])))