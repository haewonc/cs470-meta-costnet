import os 
from config.base_config import BaseConfig

class MLAE_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name):
        # Data, Train
        super().__init__(device, dataset_dir, dataset_name)

        self.batch_size = 32
        self.num_modes = 4

        # Train 
        self.optimizer = 'Adam'
        self.learning_rate = 1e-3
        self.num_channels = 1
        
        self.loss = 'MSE'
        self.metrics = ['RMSE']
        self.total_epoch = 64
        self.valid_every_epoch = 4 
        self.adpation_epochs = 16
        self.test_batch_size = 512

        self.scheduler = 'MultiStepLR'
        self.scheduler_args = {
            'milestones': [4, 8, 16, 32],
            'gamma': 0.5
        }