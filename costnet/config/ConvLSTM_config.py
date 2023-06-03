import os 
from config.base_config import BaseConfig

class ConvLSTM_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name)

        self.num_his = 12 
        self.num_pred = 12
        self.batch_size = 16 
        self.num_nodes = 20
        self.num_modes = 4
        self.num_channels = 1
        self.num_layers = 2
        self.modes_name = ['bike_pickup', 'bike_drop', 'taxi_pickup', 'taxi_drop']
        self.num_features = 14
        self.ext_features = 7

        self.loss = 'MSE'
        self.metrics = ['RMSE']
        
        self.optimizer = 'AdamW'
        self.learning_rate = 1e-5
        self.scheduler = 'OneCycleLR'
        self.start_epoch = 0
        self.total_epoch = 64
        self.scheduler_args = {
            'max_lr': self.learning_rate,
            'epochs':self.total_epoch,
            'steps_per_epoch': 231
        }
        self.valid_every_epoch = 1
        
