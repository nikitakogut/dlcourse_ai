import numpy as np
import torch
from torch.utils.data.sampler import Sampler

def display_history(train_history, val_history, plt):
    plt.figure(figsize=(14, 5))
    plt.subplot('121')
    plt.title("Loss")
    plt.plot(train_history['loss'], label='train')
    plt.plot(val_history['loss'], label='val')
    plt.legend()
    plt.subplot('122')
    plt.title("Train/Validation accuracy")
    plt.plot(train_history['score'], label='train')
    plt.plot(val_history['score'], label='val')
    plt.legend();

class SubsetSampler(Sampler):
    r"""Samples elements with given indices sequentially

    Arguments:
        indices (ndarray): indices of the samples to take
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
    

def evaluate_model(model, device, batch_size, dataset, indices):
    """
    Computes predictions and ground truth labels for the indices of the dataset
    
    Returns: 
    predictions: np array of ints - model predictions
    grount_truth: np array of ints - actual labels of the dataset
    """
    model.eval()
    sampler = SubsetSampler(indices)
    batch_size=12
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    samples_pred = np.zeros_like(indices)
    samples_gt = np.zeros_like(indices)
    for i_step, (x, y, _) in enumerate(loader):
        x_gpu = x.to(device)
        y_gpu = y.to(device)
        
        prediction = model(x_gpu)    
        _, indices = torch.max(prediction, 1)
        samples_pred[i_step*batch_size:(i_step+1)*batch_size] = indices.cpu()
        samples_gt[i_step*batch_size:(i_step+1)*batch_size] = y.cpu()
    return samples_pred, samples_gt

def visualize_confusion_matrix(predictions, ground_truth, plt, size=10):
    """
    Visualizes confusion matrix
    
    confusion_matrix: np array of ints, x axis - predicted class, y axis - actual class
                      [i][j] should have the count of samples that were predicted to be class i,
                      but have j in the ground truth
                     
    """
    confusion_matrix = build_confusion_matrix(predictions, ground_truth)
    assert confusion_matrix.shape[0] == confusion_matrix.shape[1]
    size = confusion_matrix.shape[0]
    fig = plt.figure(figsize=(size,size))
    plt.title("Confusion matrix")
    plt.ylabel("predicted")
    plt.xlabel("ground truth")
    res = plt.imshow(confusion_matrix, cmap='GnBu', interpolation='nearest')
    cb = fig.colorbar(res)
    plt.xticks(np.arange(size))
    plt.yticks(np.arange(size))
    for i, row in enumerate(confusion_matrix):
        for j, count in enumerate(row):
            plt.text(j, i, count, fontsize=14, horizontalalignment='center', verticalalignment='center')
    
def build_confusion_matrix(predictions, ground_truth):
    """
    Builds confusion matrix from predictions and ground truth

    predictions: np array of ints, model predictions for all validation samples
    ground_truth: np array of ints, ground truth for all validation samples
    
    Returns:
    np array of ints, (10,10), counts of samples for predicted/ground_truth classes
    """    
    n = len(set(predictions))
    confusion_matrix = np.zeros((n,n), np.int)
    data = np.vstack((predictions, ground_truth)).T
    for i in range(n):
        for j in range(n):
            confusion_matrix[i][j] = len(data[(data[:, 0] == i) & (data[:, 1] == j)])
    return confusion_matrix
