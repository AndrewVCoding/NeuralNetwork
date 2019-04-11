import numpy as np
import matplotlib
import network
import functions as fn

nn = network.Network(2, 1)

input = np.matrix([[1.0], [0.0]])

nn.add_layer(2, 'sigmoid')
nn.add_layer(20, 'sigmoid')
nn.add_layer(5, 'purelin')

nn.build()

print(nn.forward_propogation(input))
