import numpy as np
import matplotlib.pyplot as plt
from activation_utils import *


def parameters_initialization(layer_dimensions):
    L = len(layer_dimensions)
    parameters = {}

    for i in range(1 , L):
        parameters["W" + str(i)] = np.random.randn(layer_dimensions[i] , layer_dimensions[i-1])
        parameters["b" + str(i)] = np.random.randn(layer_dimensions[i] , 1)

    return parameters



def forward_prop_activation(A_prev, W, b, activation_type):
    

    linear_cache = (A_prev, W, b)
    Z = np.dot(W,A)+b
    
    if activation_type == "sigmoid":
        A, activation_cache = sigmoid(Z)    

    if activation_type == "relu":
        A, activation_cache = relu(Z)

    if activation_type == "leaky_relu":
        A, activation_cache = leaky_relu(Z)

    if activation_type == "tanh":        
        A, activation_cache = tanh(Z)
    
    cache = (linear_cache, activation_cache)

    return A, cache


def backward_prop_activation(dA, cache):

    linear_cache , activation_cache_data = cache
    activation_cache , activation_type = activation_cache_data
    if activation_type == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    if activation_type == "relu":
        dZ = relu_backward(dA, activation_cache)

    if activation_type == "leaky_relu":
        dZ = leaky_relu_backward(dA, activation_cache)

    if activation_type == "tanh":        
        dZ = tanh_backward(dA, activation_cache)

    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = (1/m)*np.dot(dZ,A_prev.T)
    db = (1/m)*np.sum(dZ, axis=1 , keepdims=True)
    dA_prev = np.dot(W.T,dZ)

    return dA_prev, dW, db


def cost(Y_predicted, Y, cost_type):

    m = Y.shape[1]

    if cost_type == "MSE":
        cost = (-1/m)*np.sum((Y_predicted-Y)**2)

    if cost_type == "cross_entropy":
        cost = -1 / m * np.sum(Y * np.log(Y_predicted) + (1-Y) * np.log(1-Y_predicted))

    cost = np.squeeze(cost)

    return cost



def model_forward(X , parameters):

    caches = []
    A = X
    L = len(parameters) // 2 

    for i in range(1, L):
        A_prev = A 
        A, cache = forward_prop_activation(A_prev, parameters['W'+str(i)], parameters['b'+str(i)], "relu")
        caches.append(cache)

    AL, cache = forward_prop_activation(A, parameters['W'+str(L)], parameters['b'+str(L)], "sigmoid")
    caches.append(cache)

    return AL , caches



def model_backward(AL , Y , caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)    

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    for l in reversed(range(L)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads



def update_parameters(parameters, grads, learning_rate):

    L = len(parameters)

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    
    return parameters        



def predict(X, y, parameters):

    m = X.shape[1]
    n = len(parameters) // 2 
    p = np.zeros((1,m))
    
    probas, caches = L_model_forward(X, parameters)
    
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p    