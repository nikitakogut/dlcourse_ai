import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        hidden_layer_size, int - number of neurons in the hidden layer
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        #raise Exception("Not implemented!")
        self.layers = [FullyConnectedLayer(n_input, hidden_layer_size), ReLULayer(),
                       FullyConnectedLayer(hidden_layer_size, n_output)]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        #raise Exception("Not implemented!")
        for layer in self.layers:
            if {'W', 'B'} <= set(layer.params()):
                layer.W.grad = np.zeros(layer.W.value.shape)
                layer.B.grad = np.zeros(layer.B.value.shape)
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        forward_val = X
        for layer in self.layers:
            forward_val = layer.forward(forward_val)
        loss, backward_val = softmax_with_cross_entropy(forward_val, y)
        for layer in self.layers[::-1]:
            backward_val = layer.backward(backward_val)
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        #raise Exception("Not implemented!")
        for layer in self.layers:
            for param_name, param in layer.params().items():
                    loss_reg, grad_reg = l2_regularization(param.value, self.reg)
                    loss += loss_reg
                    param.grad += grad_reg
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        #raise Exception("Not implemented!")
        forward_val = X
        for layer in self.layers:
            forward_val = layer.forward(forward_val)
        pred = np.argmax(softmax(forward_val), axis=1)
        return pred

    def params(self):
        result = {}
        # TODO Implement aggregating all of the params
        #raise Exception("Not implemented!")
        for ind, layer in enumerate(self.layers):
            for param in layer.params().items():
                result['layer_' + str(ind/2+1) + '_' + param[0]] = param[1]
        return result
