import torch
import torch.nn as nn

class STResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(STResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)
        out += self.conv3(residual)
        out = self.relu(out)
        return out

class STResNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.init = nn.Linear(
            self.config.num_nodes * self.config.num_nodes * self.config.num_modes, 
            self.config.num_features
        )
        
        self.layers = nn.Sequential()
        for _ in range(self.config.num_layers):
            self.layers.add_module(
                "linear", 
                nn.Linear(self.config.num_features, self.config.num_features)
            )
            self.layers.add_module("relu", nn.ReLU())

        self.conv_out = nn.Linear(
            self.config.num_features, 
            self.config.num_nodes * self.config.num_nodes * self.config.num_modes
        )

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.init.weight)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.xavier_uniform_(self.conv_out.weight)

    def forward(self, x):
        out = self.init(x)
        out = self.layers(out)
        out = self.conv_out(out)
        return out