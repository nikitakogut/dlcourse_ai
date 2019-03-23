import numpy as np
from tqdm import tqdm_notebook


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
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
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
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    if probs.ndim == 1:
        return -np.log(probs[target_index])
    elif probs.ndim == 2:
        return np.mean(-np.log(probs[range(probs.shape[0]), target_index]))
    raise Exception("DimensionError")


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
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

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch, 1) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    loss = cross_entropy_loss(softmax(predictions), target_index)
    dW = np.zeros(W.shape)
    dW += 1/X.shape[0] * np.dot((np.exp(predictions)/np.exp(predictions).sum(axis=1).reshape(-1, 1)).T, X).T
    I = np.zeros((X.shape[0], W.shape[1]))
    I[range(len(target_index)), target_index] = 1
    dW -= 1/X.shape[0] * np.dot(X.T, I)
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1, early_stop=1e-4, verbose=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
          early_stop, float - minimal difference between epoch loss
          verbose, float - percent of epoch to be displayed
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in tqdm_notebook(range(epochs)):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            #raise Exception("Not implemented!")
            loss = 0
            for i in range(len(batches_indices)):
                ind = batches_indices[i]
                loss += linear_softmax(X[ind], self.W, y[ind])[0] + l2_regularization(self.W, reg)[0]
                grad = linear_softmax(X[ind], self.W, y[ind])[1] + l2_regularization(self.W, reg)[1]
                self.W -= learning_rate * grad
            loss_history.append(loss / len(batches_indices))
            if(len(loss_history) > 1 and np.abs(loss_history[-1]-loss_history[-2]) < early_stop):
                print("Epoch %i, loss: %f" % (epoch, loss_history[-1]))
                print('Early stopped.')
                break
            # end
            if(epoch == 0 or  epoch == epochs - 1 or int(verbose * epoch) > int(verbose * (epoch - 1))):
                print("Epoch %i, loss: %f" % (epoch, loss_history[-1]))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        #raise Exception("Not implemented!")
        y_pred = np.argmax(softmax(np.dot(X, self.W)), axis=1)
        
        return y_pred



                
                                                          

            

                
