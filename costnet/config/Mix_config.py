import os 
from config.base_config import BaseConfig

class Mix_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name)

        self.batch_size = 32
        self.num_modes = 4

        # Train 
        self.optimizer = 'RMSprop'
        self.learning_rate = 3e-5 
        self.num_channels = 4
        
        self.loss = 'MSE'
        self.metrics = ['RMSE']
        self.total_epoch = 48 
        self.valid_every_epoch = 4 

        self.scheduler = 'OneCycleLR'
        self.scheduler_args = {
            'max_lr':3e-5,
            'total_steps':self.total_epoch,
            'steps_per_epoch': 578
        }
