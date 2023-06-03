class BaseConfig: 
    def __init__(self, device, dataset_dir, dataset_name):
        # Device
        self.device = device 

        # Data
        self.test_batch_size = 128
        self.dataset_dir = dataset_dir 
        self.dataset_name = dataset_name
        