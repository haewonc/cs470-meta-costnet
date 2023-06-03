import os 
from config.base_config import BaseConfig

class CoSTNet_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name)

        self.num_his = 12 
        self.num_pred = 12
        self.batch_size = 8
        self.num_nodes = 20
        self.num_modes = 4
        self.num_channels = 1
        self.num_layers = 1
        self.modes_name = ['bike_pickup', 'bike_drop', 'taxi_pickup', 'taxi_drop']
        self.num_features = 14
        self.ext_features = 7
        self.lstm_hidden_size = 196
        self.encoder_ckpts = ['../results/saved_models/AEN_1/{}.pth'.format(i) for i in self.modes_name]

        self.loss = 'MSE'
        self.metrics = ['RMSE', 'MSE']
        
        self.optimizer = 'AdamW'
        self.learning_rate = 1e-4
        self.scheduler = 'OneCycleLR'
        self.start_epoch = 0
        self.total_epoch = 48
        self.scheduler_args = {
            'max_lr': self.learning_rate,
            'epochs':self.total_epoch,
            'steps_per_epoch': 462
        }
        self.valid_every_epoch = 4
        
