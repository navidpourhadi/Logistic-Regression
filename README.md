# LogisticRegression

In this repo , I tried to implement Logisitc regression from scratch in python. Logistic regression is a statistical model . It is used in various applications such as medical predictions , machine learning , etc. 
Logistic Regression is a simple form of Neural Networks which classifies data in some categories. It takes an input and passes that through some functions in different layers and finally pass to a sigmoid function in order to predict . 

## Deep neural network
lots of sciences use neural networks in order to solve their problem . The only thing you need is define random weights and biases for every different features of your data and try to tune your parameters.
I implement a simple and required utils which needed to tune your parameters . I also use L2 regularization in order to prevent overfitting .
The neural network structure is implemented in n-layer with 'relu' activation function and a 'sigmoid' activation at the end of the network.


## Experiment

I tested 'cat vs non-cat' dataset on my code and get below results:

![download](https://user-images.githubusercontent.com/40823648/111034300-3a81a580-842a-11eb-8825-5f9d12fecc1d.png)

you can change the code in 'model.py' and 'n_layer_model.ipynb' based on your dataset and change 'layer_dimensions' of the code to make neural network bigger or smaller , then run the code and try to tune the parameters.



