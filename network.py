
from matplotlib import pyplot as plt
import numpy as np
import random

class Network(object):

    def __init__(self, sizes):

        self.num_layers=len(sizes)
        self.sizes = sizes
        #sizes[:-1] excludes tail
        #sizes[1:] excludes head

        #---guassian distributions with mean 0 and standard deviation 1---#

        #for each layer besides the head
        #  make a list of biases with quantity y, or number of neurons
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        #zip ends when one param is exhausted
        #(here first param is always smaller by 1)
        #zip returns a list w/ length/sizes of hidden layers
        #each list elem is tuple with (iNeurons, oNeurons)
        #so..
        #for each hidden layer
        #  make list of nIns by nOuts matrices, with |ws| = num of layer interfaces
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:] )]
        #weights[1] stores weights connecting 2nd and 3rd layers
        #w_jk is the weight connecting the kth input neuron and the jth output nueron

    def sigmoid(s):
        #this function maps onto s if its a vector
        return 1.0 / ( 1.0 + np.exp(-s))

    def feedforward(self, a):
        #a is a list of inputs
        #[[b1, w1]..[bk, wk]] where k = |sizes| - 1
        for b, w in zip(self.biases, self.weights):
            #a′=   σ   (w * a + b).
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        #training data is list of tuples (input, desired output)
        #epochs and minibatchsize are self explained
        #eta is learning rate (aka n)
        #test_data has network evaluate self and print after each epoch(slows but is useful)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        training_data = list(training_data)
        n = len(training_data)


        #for each epic
        for j in range(epochs):
            random.shuffle(training_data)

            #training data broken up into batches
            mini_batches = [
                #k goes from 0 -> n w/ m_b_size steps
                #pull from t_data k -> k + m_b_size
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                #this is where the shit happens
                #applys a single step of SGD to each mini_batch
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):

        #init these to zeros
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:

            #most of the work done here
            #figures out ∂C/∂b and ∂C/∂w
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            #update the weights and biases by adding the deltas
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #w - (n/m)(nw)
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        #init these shits to 0
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        # feedforwarding
        for b, w in zip(self.biases, self.weights):
            #z = (w * a) + b
            z = np.dot(w, activation) + b
            zs.append(z)
            #activate and toss into list
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])

        #last element ()
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

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
