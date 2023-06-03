import torch
import torch.nn as nn 

class MLPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init = nn.Linear(self.config.num_nodes * self.config.num_nodes * self.config.num_modes, self.config.num_features)
        lins = []
        lins.append(nn.ReLU())
        for i in range(self.config.num_layers):
            lin = nn.Linear(self.config.num_features, self.config.num_features) 
            torch.nn.init.xavier_uniform_(lin.weight)
            lins.append(lin)
            lins.append(nn.ReLU())
        self.lins = nn.Sequential(*lins)
        self.final = nn.Linear(self.config.num_features, self.config.num_nodes * self.config.num_nodes * self.config.num_modes)
        torch.nn.init.xavier_uniform_(self.init.weight)
        torch.nn.init.xavier_uniform_(self.final.weight)

    def forward(self, x):
        x = self.init(x)
        x = self.lins(x)
        x = self.final(x)
        return x