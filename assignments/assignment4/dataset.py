import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class HotdogOrNotDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.transform = transform
        self.folder = folder
        
        file_names = sorted([f for f in os.listdir(self.folder) if 
                            os.path.isfile(os.path.join(self.folder, f))])
        labels = ['frankfurter' in  name or 
                  'chili-dog' in name or 
                  'hotdog' in name 
                  for name in file_names]
        self.labels_frame = pd.DataFrame(np.vstack((file_names, labels)).T, 
                                         columns=['name', 'label'])
        
        
    def __len__(self):
        return len(self.labels_frame)
    
    def __getitem__(self, index):        
        img_name = os.path.join(self.folder, self.labels_frame.iloc[index, 0])
        img = Image.open(img_name)
        y = torch.tensor(1 if self.labels_frame.iloc[index, 1] == 'True' else 0)
        img_id = self.labels_frame.iloc[index, 0]
        
        if self.transform:
            img = self.transform(img)
        
        return img, y, img_id

def visualize_samples(dataset, indices, plt, title=None, count=10):
    plt.figure(figsize=(count*3,4))
    display_indices = np.random.choice(indices, count, False)
    if title:
        plt.suptitle("%s %s/%s" % (title, len(display_indices), len(indices)))        
    for i, index in enumerate(display_indices):    
        x, y, _ = dataset[index]
        plt.subplot(1,count,i+1)
        plt.title("Name: %s\nLabel: %s" % (_, int(y)))
        plt.imshow(x)
        plt.grid(False)
        plt.axis('off')   
    
