import torch
import torch.nn as nn

class OursModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(input_size=config.num_modes * config.num_features * config.num_nodes * config.num_nodes,
                            hidden_size=config.num_modes * config.lstm_hidden_size,
                            num_layers=self.config.num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(0.5)
        
        self.linear = nn.Linear(in_features=config.num_modes * config.lstm_hidden_size + config.ext_features,
                                out_features=config.num_modes * config.num_features * config.num_nodes * config.num_nodes)
        
        torch.nn.init.xavier_uniform_(self.linear.weight)
        
    def forward(self, enc, ext):
        out, _ = self.lstm(enc)
        out = self.dropout(out)
        out = self.linear(torch.cat([out, ext], dim=-1))
        out = out.reshape(out.size(0), self.config.num_pred, -1, self.config.num_modes)
        
        return out
