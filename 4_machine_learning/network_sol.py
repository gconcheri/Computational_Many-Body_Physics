# Copyright (c) 2012-2018 M. Nielsen (under MIT license)
# modified version of network.py from http://neuralnetworksanddeeplearning.com
"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network:

    def __init__(self, sizes, standard_init=True):
        """
        The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a two-layer network (not counting
        input layer), with the input layer containing 2 neurons,
        the hidden layer 3 neurons, and the output layer 1 neuron.
        The biases and weights for the network are initialized randomly,
        using a Gaussian distribution with mean 0, and variance 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        Optional information:
            More precisely, the input should always be pre-
            processed with normalization.
            The standard init follows the Xavier initialization.
            See http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        if standard_init: #Xavier initialization
            self.biases = [np.random.randn(y, 1) * 1e-3 for y in sizes[1:]] 
            #Generates a matrix of shape (y, 1) with random values from a standard normal distribution (mean 0, variance 1)
            #biases are initialized for all layers except the input layer.
            self.weights = [np.random.randn(y, x) * np.sqrt(6/(x + y))
                            for x, y in zip(sizes[:-1], sizes[1:])]
            #zip() Pairs each layer size with the next one to determine the shape of the weight matrices: [(size[layer1],size[layer2]),(size[2],size[3])...]
            #self.weights = matrices of dimensions size[2]xsize[1] (because (y,x)) with randomized gaussian entries!! 
            # to understand why remember weights defined in class w_jk
        else:
            self.biases = [np.random.randn(y, 1)  for y in sizes[1:]]
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                eval_correct = self.evaluate(test_data)
                n_test = len(test_data)
                print("Epoch {0: 2d}: {1:8d} / {2} = {3:.1f}%".format(
                    j, eval_correct, n_test, 100*eval_correct/n_test))
                #j: current epoch number, eval_correct: num of correct predictions
                #n_test: total number of tests, last term is percentage of correct predictions
            else:
                print("Epoch {0: 2d} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #creo array vuoto per ogni layer, con tante entries quanti i b
        nabla_w = [np.zeros(w.shape) for w in self.weights] #creo matrice vuota per ogni layer, con tante entries quanti i w a quel layer
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    # implement the backpropagation for the network, solution from the book.
    # full explanation: http://neuralnetworksanddeeplearning.com/chap2.html
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


if __name__ == "__main__":
    # example usage: train a network to recognise digits using the mnist data
    # first load the data
    import data_loader
    training_data, _, test_data = data_loader.load_data_wrapper("mnist.pkl.gz")
    # then generate the neuronal network
    net = Network([784, 30, 10])
    # and train it for 15 epochs
    net.SGD(training_data, 15, 10, 0.5, test_data=test_data)
