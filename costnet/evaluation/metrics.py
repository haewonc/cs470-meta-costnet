import numpy as np
import torch.nn as nn 
import torch 
from scipy import stats

class MSE(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, preds, labels):
        loss = torch.pow(preds - labels, 2)
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

class RMSE(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, preds, labels):
        loss = torch.pow(preds - labels, 2)
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.sqrt(torch.mean(loss))

class MAPE(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, preds, labels):
        loss = torch.abs(torch.divide(torch.subtract(preds, labels), labels))
        loss = torch.nan_to_num(loss)
        return torch.mean(loss)

class PCC():
    def __init__(self):
        pass
    
    def __call__(self, preds, labels):
        x = preds.reshape(preds.size(0)*preds.size(1),-1).numpy()
        y = labels.reshape(labels.size(0)*labels.size(1),-1).numpy()
        p = 0.0
        for i in range(x.shape[0]):
            p += stats.pearsonr(x[i],y[i])[0]
        return p/x.shape[0]