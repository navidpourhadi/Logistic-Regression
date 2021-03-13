import numpy as np
import matplotlib.pyplot as plt
from activation_utils import *


def parameters_initialization(layer_dimensions):
    L = len(layer_dimensions)
    parameters = {}

    for i in range(1 , L):
        parameters["W" + str(i)] = np.random.randn(layer_dimensions[i] , layer_dimensions[i-1]) / np.sqrt(layer_dimensions[i-1])
        parameters["b" + str(i)] = np.random.randn(layer_dimensions[i] , 1)
        
        print("W"+str(i)+".shape = "+str(parameters["W" + str(i)].shape))
        print("b"+str(i)+".shape = "+str(parameters["b" + str(i)].shape))

    return parameters



def forward_prop_activation(A_prev, W, b, activation_type):
    

    linear_cache = (A_prev, W, b)
    Z = np.dot(W,A_prev)+b
    
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


def backward_prop_activation(dA, cache, lambd):

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

    dW = (1./m)*np.dot(dZ,A_prev.T)+lambd/m*W
    db = (1./m)*np.sum(dZ, axis=1 , keepdims=True)
    dA_prev = np.dot(W.T,dZ)

    return dA_prev, dW, db


def cost(Y_predicted, Y,parameters, lambd, cost_type):

    m = Y.shape[1]
    L = len(parameters)//2

    if cost_type == "MSE":
        cost = (-1./m)*np.sum((Y_predicted-Y)**2)

    if cost_type == "cross_entropy":
        cost = (1./m) * (-np.dot(Y,np.log(Y_predicted).T) - np.dot(1-Y, np.log(1-Y_predicted).T))
        
    cost = np.squeeze(cost)
    
    L2_regularization = 0

    for i in range(L):
        L2_regularization += np.sum(parameters["W"+str(i+1)])
        
    L2_regularization = 1./m * lambd / 2 * L2_regularization



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



def model_backward(AL , Y , caches , lambd):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)    

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = backward_prop_activation(dAL, current_cache, lambd)

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = backward_prop_activation(grads["dA"+str(l+1)], current_cache, lambd)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads



def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    
    return parameters        



def predict(X, y, parameters):

    m = X.shape[1]
    n = len(parameters) // 2 
    p = np.zeros((1,m))
    
    probas, caches = model_forward(X, parameters)
    
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p    