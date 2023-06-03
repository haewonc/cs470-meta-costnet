import torch 
import numpy as np
from torch.utils.data import Dataset

class AutoEncoderDataset(Dataset):
    def __init__(self, data):
        self.x = torch.tensor(data)
    
    def __getitem__(self, index):
        return self.x[index].float(), self.x[index].float()
    
    def __len__(self):
        return self.x.size(0)
    
class CoSTNetDataset(Dataset):
    def __init__(self, data):
        self.x = torch.tensor(data['x'])
        self.y = torch.tensor(data['y'])
        self.ext_x = torch.tensor(data['ext_x'])

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float(), self.ext_x[index].float()
    
    def __len__(self):
        return self.y.size(0)