import numpy as np
import torch
from tqdm.auto import tqdm

def tqdm_enumerate(iter):
    i = 0
    for y in tqdm(iter):
        yield i, y
        i += 1

def train_model(model, device, dataset, loss, optimizer, scheduler, num_epochs):
    '''
    Trains plain word2vec using cross-entropy loss and regenerating dataset every epoch
    
    Returns:
    loss_history, train_history
    '''
    loss_history = []
    train_history = []
    for epoch in range(num_epochs):
        model.train()
        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        
        del dataset.samples
        dataset.samples = []
        dataset.generate_dataset()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=20)
        
        for i_step, (x, y) in tqdm_enumerate(train_loader):
            x_gpu = x.to(device)
            y_gpu = y.to(device)
            prediction = model(x_gpu)   
            loss_value = loss(prediction, y_gpu)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            _, indices = torch.max(prediction, 1)
            correct_samples += torch.sum(indices == y_gpu)
            
            total_samples += y.shape[0]
            
            loss_accum += loss_value.detach()

        ave_loss = loss_accum / i_step
        train_accuracy = float(correct_samples) / total_samples
        
        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        
        print("Epoch %i, Average loss: %f, Train accuracy: %f" % (epoch+1, ave_loss, train_accuracy))
        if scheduler is not None:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(ave_loss)
            else:
                scheduler.step()
        
    return loss_history, train_history
        
def train_neg_sample(model, device, dataset, loss, optimizer, scheduler, num_epochs):    
    '''
    Trains word2vec with negative samples on and regenerating dataset every epoch
    
    Returns:
    loss_history, train_history
    '''
    loss_history = []
    train_history = []
    for epoch in range(num_epochs):
        model.train()
        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        
        del dataset.samples
        dataset.samples = []
        dataset.generate_dataset()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=20)
        
        for i_step, (x, y, z) in tqdm_enumerate(train_loader):
            x_gpu = x.to(device)
            y_gpu = y.to(device)
            z_gpu = z.to(device)
            
            prediction = model(x_gpu, y_gpu)  
            loss_value = loss(prediction, z_gpu)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            loss_accum += loss_value.detach()
            correct_samples += sum(1 / (1+np.exp(-prediction[:, 0].detach().cpu().numpy())) > .5)
            total_samples += y.shape[0]  
            
        ave_loss = loss_accum / i_step
        train_accuracy = float(correct_samples) / total_samples
        
        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        
        print("Epoch %i, Average loss: %f, Train accuracy: %f" % (epoch+1, ave_loss, train_accuracy))
        if scheduler is not None:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(ave_loss)
            else:
                scheduler.step()
        
    return loss_history, train_history
