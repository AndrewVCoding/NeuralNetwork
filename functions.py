import numpy as np


def relu(x):
    if x <= 0:
        return 0
    else:
        return x


def d_relu(x):
    if x <= 0:
        return 0
    else:
        return 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return np.exp(-x) / np.power(1 + np.exp(-x), 2)


def purelin(x):
    return x


def d_purelin(x):
    return 1


# loss functions
def loss(x, s):
    return (1 / len(x)) * sum(np.power(s - x, 2))


def linear(s, t):
    return np.transpose(t)


func_dict = {'relu': relu, 'd_relu': d_relu, 'sigmoid': sigmoid, 'd_sigmoid': d_sigmoid, 'purelin': purelin, 'd_purelin': d_purelin}


def func(f):
    return func_dict[f]
