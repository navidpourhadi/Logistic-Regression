"""
Developer: Navid Pourhadi
Date : 22 February 2021

This file contains implementations of forward and backward propagations
of sigmiod , relu , leaky relu , tanh activation functions

"""

import numpy as np


# forward propagations

def sigmoid(Z):
    A = 1/1+np.exp(-Z)
    cache = (Z , "sigmoid")

    return A, cache


def relu(Z):
    A = np.maximum(0,Z)
    cache = (Z , "relu")

    return A, cache


def leaky_relu(Z):
    A = np.maximum(0.1*Z , Z)
    cache = (Z , "leaky_relu")

    return A, cache    


def tanh(Z):
    A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))
    cache = (Z , "tanh")

    return A, cache




# backward propagations

def sigmoid_backward(dA , cache):
    
    Z = cache
    temp = 1/1+np.exp(-Z)
    dZ = dA * temp * (1-temp)

    return dZ


def relu_backward(dA , cache):

    Z = cache
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0
    return dZ


def leaky_relu_backward(dA , cache):

    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0.1

    return dZ 


def tanh_backward(dA, cache):

    Z = cache
    temp = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))
    dZ = dA * (1-np.power(temp,2))

    return dZ