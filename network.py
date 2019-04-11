import numpy as np
import functions as fn


class Layer:

    def __init__(self, size, function):
        self.nodes = np.zeros(size)
        self.function = function

    def __eq__(self, other):
        return other == self.nodes

    def __repr__(self):
        return str(self.nodes)

    def __len__(self):
        return len(self.nodes)

    def __add__(self, other):
        return self.nodes + other

    def __mul__(self, other):
        return self.nodes * other

    def __iadd__(self, other):
        self.nodes += other
        return self.nodes

    def propogate(self, x, w):
        self.nodes = self.function(np.matmul(np.transpose(w), x))

    def set(self, x):
        self.nodes = x


class Network:

    def __init__(self, loss_function=fn.loss, gamma=0.1):
        self.layers = list()
        self.functions = list()
        self.weights = list()
        self.loss_function = loss_function
        self.gamma = gamma

    def build(self):
        # Create the weight matrices
        for l in range(1, len(self.layers)):
            x = len(self.layers[l - 1])
            y = len(self.layers[l])
            self.weights.append(np.asmatrix(np.random.rand(x, y)))

    def add_layer(self, size, function='sigmoid'):
        function = fn.func(function)
        self.layers.append(np.zeros((size, 1)))
        self.functions.append(function)

    def forward_propogation(self, input):
        # activate the first layer
        self.layers[0] = input
        # set each layer as the product of the activation pattern of the previous layer and their weights
        for i in range(1, len(self.layers)):
            x = self.functions[i-1](self.layers[i-1])
            self.layers[i] = np.matmul(np.transpose(self.weights[i - 1]), x)

        return self.functions[-1](self.layers[-1])

    def back_propogation(self):
        return
