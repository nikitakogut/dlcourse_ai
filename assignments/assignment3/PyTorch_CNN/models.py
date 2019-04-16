import torch
import torch.nn as nn
from torchvision import models

class Flattener(nn.Module):
    def forward(self, x):
        batch_size, *_ = x.shape
        return x.view(batch_size, -1)

def Regular(device):
	model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),   
 
            Flattener(),

            nn.Linear(64*2*2, 10),
          )
	model.type(torch.cuda.FloatTensor)
	model.to(device)
	return model

def LeNet(device):
    model = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            Flattener(),

            nn.Linear(400, 120),
            nn.ReLU(inplace=True),  
      
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),  
      
            nn.Linear(84, 10),
              )

    model.type(torch.cuda.FloatTensor)
    model.to(device)
    return model

def MyModel(device):
    model = nn.Sequential(              
            nn.Conv2d(3, 48, 3, padding=2),#34
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(48), 
            nn.MaxPool2d(2),#17
            nn.Dropout2d(.2),
               
            nn.Conv2d(48, 128, 4, padding=2),#18
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),#9
            nn.Dropout2d(.2),
               
            nn.Conv2d(128, 192, 4, padding=2),#10
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(2),#5   
            nn.Dropout2d(.2),
               
            nn.Conv2d(192, 192, 4, padding=2),#6
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(2),#3
            nn.Dropout2d(.2),
               
            Flattener(),
            
			nn.Linear(3*3*192, 1024),
            nn.ReLU(inplace=True),        
            nn.BatchNorm1d(1024),
            nn.Dropout2d(.2),
               
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True), 
            nn.BatchNorm1d(1024),
            nn.Dropout2d(.2),

            nn.Linear(1024, 10),
              )

    model.type(torch.cuda.FloatTensor)
    model.to(device)
    return model

def ResNet(device):
    model = models.resnet18(num_classes=10)
    model = model.to(device)
    return model
