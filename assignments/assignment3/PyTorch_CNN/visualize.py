import numpy as np
import torch
from torch.utils.data.sampler import Sampler

def display_history(loss_history, train_history, val_history, plt):
    plt.figure(figsize=(14, 5))
    plt.subplot('121')
    plt.title("Train/Validation accuracy")
    plt.plot(train_history, label='train')
    plt.plot(val_history, label='val')
    plt.legend()
    plt.subplot('122')
    plt.title("Loss")
    plt.plot(loss_history, label='loss')
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
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    predictions = []
    ground_truth = []
    for i_step, (x, y) in enumerate(loader):
        x_gpu = x.to(device)
        y_gpu = y.to(device)
        
        prediction = model(x_gpu)    
        _, ind = torch.max(prediction, 1)
        predictions += list(ind.cpu().data.numpy())
        ground_truth += list(y.data.numpy())
    return np.array(predictions), np.array(ground_truth)

def visualize_confusion_matrix(confusion_matrix, plt):
    """
    Visualizes confusion matrix
    
    confusion_matrix: np array of ints, x axis - predicted class, y axis - actual class
                      [i][j] should have the count of samples that were predicted to be class i,
                      but have j in the ground truth
                     
    """
    assert confusion_matrix.shape[0] == confusion_matrix.shape[1]
    size = confusion_matrix.shape[0]
    fig = plt.figure(figsize=(10,10))
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
    confusion_matrix = np.zeros((10,10), np.int)
    data = np.vstack((predictions, ground_truth)).T
    for i in range(10):
        for j in range(10):
            confusion_matrix[i][j] = len(data[(data[:, 0] == i) & (data[:, 1] == j)])
    return confusion_matrix

def visualize_images(indices, data, plt, title='', max_num=10):
    """
    Visualizes several images from the dataset
 
    indices: array of indices to visualize
    data: torch Dataset with the images
    title: string, title of the plot
    max_num: int, max number of images to display
    """
    to_show = min(len(indices), max_num)
    ind_show = np.random.choice(indices, to_show, replace=False)
    fig = plt.figure(figsize=(10,1.5))
    fig.suptitle(title)
    for i, index in enumerate(ind_show):
        plt.subplot(1,to_show, i+1)
        plt.axis('off')
        sample = data[index][0]
        plt.imshow(sample)
        
def visualize_predicted_actual(predicted_class, gt_class, predictions, 
			       ground_truth, val_indices, val_data, plt):
    """
    Visualizes images of a ground truth class which were predicted as the other class 
    
    predicted: int 0-9, index of the predicted class
    gt_class: int 0-9, index of the ground truth class
    predictions: np array of ints, model predictions for all validation samples
    ground_truth: np array of ints, ground truth for all validation samples
    val_indices: np array of ints, indices of validation samples
    """
    data = np.vstack((val_indices, predictions, ground_truth)).T
    indices = data[(data[:, 1] == predicted_class) & (data[:, 2] == gt_class)][:,0]
    title = f'Failing samples. Predicted: {predicted_class}, Actual: {gt_class}'
    visualize_images(indices, val_data, plt, title=title, max_num=10)

