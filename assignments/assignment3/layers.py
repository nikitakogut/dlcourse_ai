import numpy as np
from copy import deepcopy


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    predictions = predictions - np.max(predictions)
    if predictions.ndim == 1:
        return (np.exp(predictions) / np.sum(np.exp(predictions[np.newaxis, :]), axis=1).reshape(-1, 1))[0]
    elif predictions.ndim == 2:
        return np.exp(predictions) / np.sum(np.exp(predictions), axis=1).reshape(-1, 1)
    raise Exception("DimensionError")
    
    
def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if probs.ndim == 1:
        return -np.log(probs[target_index])
    elif probs.ndim == 2:
        return np.mean(-np.log(probs[range(probs.shape[0]), target_index]))
    raise Exception("DimensionError")
    

def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    loss = cross_entropy_loss(softmax(predictions), target_index)
    if predictions.ndim == 1:
        dprediction = np.exp(predictions)/np.sum(np.exp(predictions))
        dprediction[target_index] -= 1
    elif predictions.ndim == 2:
        dprediction = 1/predictions.shape[0]*np.exp(predictions)/np.sum(np.exp(predictions), axis=1).reshape(-1, 1)
        dprediction[range(predictions.shape[0]), target_index] -= 1/predictions.shape[0]
    else:
        raise Exception("DimensionError")   
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = deepcopy(X)
        return X * (X > 0) 

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_result = d_out * (self.X > 0) 
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = deepcopy(X)
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0)[np.newaxis, :]
        d_result = np.dot(d_out, self.W.value.T)
        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros((1, out_channels)))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = height - self.filter_size + 1 + self.padding * 2
        out_width = width - self.filter_size + 1 + self.padding * 2
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        if self.padding > 0:
            X_padding = np.zeros((batch_size, height+self.padding * 2, width+self.padding * 2, channels))
            X_padding[:, self.padding:self.padding+height, self.padding:self.padding+width, :] = X
            X = X_padding
        self.X = deepcopy(X)
        conv_result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                region = X[:, y:y+self.filter_size, x:x+self.filter_size, :]
                conv_result[:, y, x, :] = region.reshape((batch_size, -1)) @ \
                                          self.W.value.reshape((-1, self.out_channels)) + \
                                          self.B.value
        return conv_result


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        d_result = np.zeros_like(self.X)
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                region_reshaped = self.X[:, y:y+self.filter_size, x:x+self.filter_size, :].reshape((batch_size, -1))
                W_reshaped = self.W.value.reshape((-1, self.out_channels))
                
                self.W.grad += np.dot(region_reshaped.T, d_out[:, y, x, :]).reshape(self.W.value.shape)
                self.B.grad += np.sum(d_out[:, y, x, :], axis=0)[np.newaxis, :]
                
                region_shape = (batch_size, self.filter_size, self.filter_size, channels)
                d_result[:, y:y+self.filter_size, x:x+self.filter_size, :] += np.dot(d_out[:, y, x, :], W_reshaped.T) \
                                                                              .reshape(region_shape)
        if self.padding > 0:
            d_result = d_result[:, self.padding:self.padding+height-self.padding*2, \
                                   self.padding:self.padding+width-self.padding*2, :]
        return d_result

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool
        
        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        out_height = (height-self.pool_size)//self.stride+((height-self.pool_size)%self.stride>0)+1
        out_width = (width-self.pool_size)//self.stride+((height-self.pool_size)%self.stride>0)+1
        
        self.X = deepcopy(X)
        pooling_result = np.zeros((batch_size, out_height, out_width, channels))
        for y in range(out_height):
            for x in range(out_width):
                stride = np.array([y, x]) * self.stride
                if y == out_height - 1:
                    stride[0] = height - self.pool_size
                if x == out_width - 1:
                    stride[1] = width - self.pool_size
                region = X[:, stride[0]:stride[0]+self.pool_size, stride[1]:stride[1]+self.pool_size, :]
                pooling_result[:, y, x, :] = region.max(axis=1).max(axis=1)
        return pooling_result

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        d_result = np.zeros_like(self.X)
        for y in range(out_height):
            for x in range(out_width):
                stride = np.array([y, x]) * self.stride
                if y == out_height - 1:
                    stride[0] = height - self.pool_size
                if x == out_width - 1:
                    stride[1] = width - self.pool_size
                region = self.X[:, stride[0]:stride[0]+self.pool_size, stride[1]:stride[1]+self.pool_size, :]
                d_result[:, stride[0]:stride[0]+self.pool_size, stride[1]:stride[1]+self.pool_size, :] += \
                np.equal(np.ones_like(region) * region.max(axis=1).max(axis=1).reshape((batch_size, 1, 1, channels)), region)*\
                d_out[:, y, x, :].reshape((batch_size, 1, 1, channels))
        return d_result

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        
        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        self.X_shape = X.shape
        return X.reshape((batch_size, -1)) 

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)
                       
    def params(self):
        # No params!
        return {}
