"""
network.py
==========

Defines a neural network.
"""

import numpy as np
import random

class Activations:

    @staticmethod
    def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))


class Network:

    def __init__(self, sizes):
        """
        Sizes is a list containing the number
        of neurons in each layer, e.g. [3, 2, 4].

        Biases and weights are initialised randomly for now.
        """

        self.num_layers = len(sizes)
        self.sizes = sizes
        # np.random.randn(y, 1) generates a y by 1 vector of
        # random numbers, using a normal distribution with mean 0
        # and s.d. 1
        # We start from the second layer since the input layer needs 
        # no biases.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # The weights are a y by x matrix, where y is the size of layer
        # l, and x is the size of layer l + 1.
        # Hence, y ranges from the input until the second to last layer
        # and x ranges from the second to the last layer.
        self.weights = [np.random.randn(y, x)
                for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        # a is a vector of inputs.
        # We compute the output for this set of inputs by computing
        # sigmoid(weight * a + bias) for each layer, where each
        # weight is a matrix and each bias is a vector.
        for bias, weight in zip(self.biases, self.weights):
            a = Activations.sigmoid(np.dot(weight, a) + bias)

        return a

    def stochastic_grad_desc(self, training_data, epochs, mini_batch_size,
            learning_rate, test_data=None):
        """
        This function trains the network using stochastic gradient descent
        on mini-batches.

        SGD using mini-batches runs the feedforward and backpropogation
        over a small number of batches, say 50, and averages to find
        the cost. It then adjusts the weights and biases. An advantage
        of doing this instead of training over all examples is that
        convergence occurs faster.

        Test data is a list of tuples, mapping training inputs to
        the correct output.

        After all training inputs are exhaused, an epoch is completed.
        If test data is given, the network will be tested against the data
        after each epoch, useful for showing progress, but otherwise slow.
        """

        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                    training_data[k:k + mini_batch_size]
                    for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

        def update_mini_batch(self, mini_batch, learning_rate):
            """
            Updates the weights and biases by using grad. descent
            on the mini batch.
            """

            pass
