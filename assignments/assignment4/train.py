import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

def tqdm_enumerate(iter):
    i = 0
    for y in tqdm(iter):
        yield i, y
        i += 1
        
def train_model(model, device, train_loader, val_loader, loss, optimizer, num_epochs, scheduler=None):  
    train_loss_history = []
    val_loss_history = []
    train_score_history = []
    val_score_history = []
    batch_size = train_loader.batch_size
    for epoch in range(num_epochs):
        model.train()
        loss_accum = 0
        samples_pred = np.zeros(len(train_loader.sampler))
        samples_gt = np.zeros(len(train_loader.sampler))
        
        for i_step, (x, y, _) in tqdm_enumerate(train_loader):
            x_gpu = x.to(device)
            y_gpu = y.to(device)
            prediction = model(x_gpu)    
            loss_value = loss(prediction, y_gpu)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            _, indices = torch.max(prediction, 1)
            samples_pred[i_step*batch_size:(i_step+1)*batch_size] = indices.cpu()
            samples_gt[i_step*batch_size:(i_step+1)*batch_size] = y.cpu()
            
            loss_accum += loss_value.detach()
            
        ave_loss = loss_accum / i_step
        train_accuracy = f1_score(samples_gt, samples_pred)
        val_loss, val_accuracy = compute_accuracy(model, device, val_loader, loss)
        
        train_loss_history.append(float(ave_loss))
        val_loss_history.append(val_loss)
        train_score_history.append(train_accuracy)
        val_score_history.append(val_accuracy)
        
        print("Epoch: %d, Train loss: %f, Train F1 score: %f, Val loss: %f, Val F1 score: %f" % 
              (epoch + 1, ave_loss, train_accuracy, val_loss, val_accuracy))
        
        if scheduler is not None:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
    return {'loss': train_loss_history, 'score': train_score_history},\
           {'loss': val_loss_history, 'score': val_score_history}
        
def compute_accuracy(model, device, loader, loss):
    """
    Computes accuracy on the dataset wrapped in a loader
    
    Returns: accuracy as a float value between 0 and 1
    """
    model.eval()
    samples_pred = np.zeros(len(loader.sampler))
    samples_gt = np.zeros(len(loader.sampler))
    loss_accum = 0
    batch_size = loader.batch_size
    for i_step, (x, y, _) in enumerate(loader):
        x_gpu = x.to(device)
        y_gpu = y.to(device)
        prediction = model(x_gpu)    
        loss_accum += loss(prediction, y_gpu).detach()
        
        _, indices = torch.max(prediction, 1)
        samples_pred[i_step*batch_size:(i_step+1)*batch_size] = indices.cpu()
        samples_gt[i_step*batch_size:(i_step+1)*batch_size] = y.cpu()
    ave_loss = loss_accum / i_step
    return ave_loss, f1_score(samples_gt, samples_pred)
