from layers_utils import *
from activation_utils import *
from sklearn.model_selection import train_test_split
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image


# import h5 dataset
def dataset(address, input_name, output_name, class_name):
    dataset = h5py.File(address , 'r')
    dataset_X = np.array(dataset[input_name][:]) 
    dataset_Y = np.array(dataset[output_name][:])
    
    print(dataset_Y.shape)


    train_data, test_data, train_label, test_label = train_test_split(dataset_X, dataset_Y, test_size=0.2)

    train_label = train_label.reshape((1, train_label.shape[0]))
    test_label = test_label.reshape((1, test_label.shape[0]))


    classes = np.array(dataset[class_name][:]) 

    return train_data, test_data, train_label, test_label, classes


def flatten_dataset(dataset):
    dataset = dataset.reshape(dataset.shape[0] , -1).T
    dataset = dataset/255.
    return dataset


def L_layer_model(X, Y, layer_dimensions,  lr=0.0025, epoch=1000,lambd=0.1, print_cost=False):

    costs = []

    parameters = parameters_initialization(layer_dimensions)

    for i in range(0, epoch):

        Y_pred , caches = model_forward(X , parameters)

        compute_cost = cost(Y_pred , Y ,parameters,lambd, "cross_entropy")

        gradients = model_backward(Y_pred , Y , caches, lambd)

        parameters = update_parameters(parameters , gradients , lr)

        if print_cost and i % 50 == 0:
            print ("Cost after epoch %i: %f" %(i, compute_cost))
        if print_cost and i % 50 == 0:
            costs.append(compute_cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('epochs (per tens)')
    plt.title("Learning rate =" + str(lr))
    plt.show()
    plt.savefig('results.png',dpi=200)

    return parameters





