import os 
from config.base_config import BaseConfig

class STResNet_config(BaseConfig):
    def __init__(self, device, dataset_dir, dataset_name):

        super().__init__(device, dataset_dir, dataset_name)
        
        self.num_his = 8
        self.num_pred = 8
        self.batch_size = 16 
        self.num_nodes = 20
        self.num_modes = 4
        self.num_channels = 1
        self.num_layers = 4
        self.num_features = 512
        self.modes_name = ['bike_pickup', 'bike_drop', 'taxi_pickup', 'taxi_drop']
        self.ext_features = 7

        self.len_closeness = 5  # Number of time steps for closeness
        self.len_period = 3  # Number of time steps for period
        self.len_trend = 3  # Number of time steps for trend
        #self.external_dim = None  # Number of external dimensions (e.g., holiday, temperature, etc.)
        self.nb_flow = 2  # Number of flow channels (e.g., inflow and outflow)
        self.map_height = 20  # Height of the spatial grid
        self.map_width = 20 # Width of the spatial grid

        # Model settings
        self.nb_residual_unit = 4  # Number of residual units in the time network
        
        self.modes_name = ['bike_pickup', 'bike_drop', 'taxi_pickup', 'taxi_drop']
        self.num_features = 14
        self.num_residual_units = 2
        self.filter_sizes = [2, 2]

        self.loss = 'MSE'
        self.metrics = ['RMSE', 'MSE', 'PCC']

        self.optimizer = 'Adam'
        self.learning_rate = 1e-4
        self.scheduler = 'OneCycleLR'
        self.start_epoch = 0
        self.total_epoch = 64
        self.scheduler_args = {
            'max_lr': self.learning_rate,
            'epochs':self.total_epoch,
            'steps_per_epoch': 231
        }
        self.valid_every_epoch = 4