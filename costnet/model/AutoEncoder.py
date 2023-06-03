'''
Implementation of heterogeneous autoencoder
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv2d(config.num_channels, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 14, 3, 1, 1)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x 
        
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tconv1 = nn.ConvTranspose2d(14, 16, 3, 1, 1)
        self.tconv2 = nn.ConvTranspose2d(16, config.num_channels, 3, 1, 1)
        torch.nn.init.xavier_uniform_(self.tconv1.weight)
        torch.nn.init.xavier_uniform_(self.tconv2.weight)

    def forward(self, x):
        x = self.tconv1(x)
        x = F.relu(x)
        x = self.tconv2(x)
        return x 

class AutoEncoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x