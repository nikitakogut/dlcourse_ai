import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels, reg=0):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        self.reg = reg
        # TODO Create necessary layers
        assert input_shape[0] % 4 == 0 & input_shape[1] % 4 == 0, "Invalid input_shape value"
        self.layers = [ConvolutionalLayer(input_shape[2], conv1_channels, 3, 0), 
                       ReLULayer(), 
                       MaxPoolingLayer(4, 4),
                       ConvolutionalLayer(conv1_channels, conv2_channels, 3, 0), 
                       ReLULayer(), 
                       MaxPoolingLayer(4, 4),
                       Flattener(),
                       FullyConnectedLayer(4 * conv2_channels, n_output_classes)
                      ]
        

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        for layer in self.layers:
            if {'W', 'B'} <= set(layer.params()):
                layer.W.grad = np.zeros(layer.W.value.shape)
                layer.B.grad = np.zeros(layer.B.value.shape)
        forward_val = X
        for layer in self.layers:
            forward_val = layer.forward(forward_val)
        loss, backward_val = softmax_with_cross_entropy(forward_val, y)
        for layer in self.layers[::-1]:
            backward_val = layer.backward(backward_val)
        for layer in self.layers:
            for param_name, param in layer.params().items():
                    loss_reg, grad_reg = l2_regularization(param.value, self.reg)
                    loss += loss_reg
                    param.grad += grad_reg
        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        pred = np.zeros(X.shape[0], np.int)
        forward_val = X
        for layer in self.layers:
            forward_val = layer.forward(forward_val)
        pred = np.argmax(softmax(forward_val), axis=1)
        return pred

    def params(self):
        result = {}
        # TODO: Aggregate all the params from all the layers
        # which have parameters
        for ind, layer in enumerate(self.layers):
            for param in layer.params().items():
                result['layer_' + str(ind/2+1) + '_' + param[0]] = param[1]
        return result
