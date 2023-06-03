import time
import math
import torch 
import torch.nn as nn 
import numpy as np 
from data.utils import *
from data.datasets import AutoEncoderDataset
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer
from util.logging import * 
import pickle
import json

'''
AutoEncoderTrainer()
'''

class AutoEncoderTrainer(BaseTrainer):
    def __init__(self, cls, config, args):
        super().__init__()
        self.config = config
        self.device = self.config.device
        self.cls = cls
        self.setup_save(args)
        self.best_model = None 
        self.best_metric = 1000

    def setup_model(self):
        self.model = self.cls(self.config).to(self.device)

    def load_dataset(self):
        with open('{}/{}.pkl'.format(self.config.dataset_dir, self.config.dataset_name), 'rb') as file:
            data = pickle.load(file)
        self.data_min = data['min']
        self.data_max = data['max']
        return data


    def compose_dataset(self):
        datasets = self.load_dataset()
        for category in ['train', 'test']:
            datasets[category] = AutoEncoderDataset(datasets[category])
        self.num_train_iteration_per_epoch = math.ceil(len(datasets['train']) / self.config.batch_size)
        self.num_val_iteration_per_epoch = math.ceil(len(datasets['test']) / self.config.test_batch_size)
        self.num_test_iteration_per_epoch = math.ceil(len(datasets['test']) / self.config.test_batch_size)
        return datasets['train'], datasets['test']

    def compose_loader(self):
        self.train_dataset, self.test_dataset = self.compose_dataset()
        print(toGreen('Dataset loaded successfully ') + '| Train [{}] Test [{}]'.format(toGreen(len(self.train_dataset)), toGreen(len(self.test_dataset))))
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.test_batch_size, shuffle=False)
        self.val_loader = self.test_loader

    def train(self):
        data_names = ['BIKE_PICKUP', 'BIKE_DROP', 'TAXI_PICKUP', 'TAXI_DROP']
        for i in range(4):
            self.setup_model()
            self.setup_train()
            print(toGreen(f'\n{data_names[i]} TRAINING START'))
            for epoch in range(self.config.total_epoch):
                total_loss, total_metrics = self.train_epoch(epoch, i)
                avg_loss = total_loss / len(self.train_loader)
                avg_metrics = total_metrics / len(self.train_loader)
                self.history['train'].append({'loss': avg_loss, 'metrics': list(avg_metrics)})
                print_total('TRAIN', epoch, self.config.total_epoch, self.config.loss, avg_loss, self.config.metrics, avg_metrics)
                if epoch % self.config.valid_every_epoch == 0:
                    self.validate(epoch, i, is_test=False)
            print(toGreen(f'\n{data_names[i]} TRAINING END'))
            self.validate(epoch, i, is_test=True)
            print(toGreen(f'Saving the model with {self.config.metrics[0]} {round(self.best_metric,2)}'))
            torch.save(self.best_model, '../results/saved_models/{}/{}.pth'.format(self.save_name, data_names[i].lower()))
            with open('../results/saved_models/{}/history_{}.json'.format(self.save_name, data_names[i].lower()), 'w') as file:
                json.dump(self.history, file)
                self.history = {}
                self.history['train'] = []
                self.history['test'] = []
            self.best_metric = 1000

    def validate(self, epoch, i, is_test=False):
        total_loss, total_metrics = self.validate_epoch(epoch, i, is_test)
        avg_loss = total_loss / len(self.test_loader if is_test else self.val_loader)
        avg_metrics = total_metrics / len(self.test_loader if is_test else self.val_loader)
        if avg_metrics < self.best_metric:
            self.best_model = self.model.state_dict()
            self.best_metric = avg_metrics[0] 
        self.history['test'].append({'loss': avg_loss, 'metrics': list(avg_metrics)})
        print_total('TEST' if is_test else 'VALID', epoch, self.config.total_epoch, self.config.loss, avg_loss, self.config.metrics, avg_metrics)

    def train_epoch(self, epoch, i):
        self.model.train()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data[:, :, :, i].unsqueeze(1).to(self.device)  
            target = target[:, :, :, i].unsqueeze(1).to(self.device)  
            
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.loss(output, target) 
            loss.backward()

            self.optimizer.step()
            training_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()

            output = output.detach().cpu()
            target = target.detach().cpu()

            output *= self.data_max[i]
            target *= self.data_max[i]
            output += self.data_min[i]
            target += self.data_min[i]

            this_metrics = self._eval_metrics(output, target)
            total_metrics += this_metrics

            print_progress('TRAIN', epoch, self.config.total_epoch, batch_idx, self.num_train_iteration_per_epoch, training_time, self.config.loss, loss.item(), self.config.metrics, this_metrics)

        return total_loss, total_metrics

    def validate_epoch(self, epoch, i, is_test):
        self.model.eval()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target) in enumerate(self.test_loader if is_test else self.val_loader):
            data = data[:, :, :, i].unsqueeze(1).to(self.device)  
            target = target[:, :, :, i].unsqueeze(1).to(self.device)  

            with torch.no_grad():
                output = self.model(data)
                loss = self.loss(output, target) 

            valid_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()

            output = output.detach().cpu()
            target = target.detach().cpu()
            
            output *= self.data_max[i]
            target *= self.data_max[i]
            output += self.data_min[i]
            target += self.data_min[i]

            this_metrics = self._eval_metrics(output, target)
            total_metrics += this_metrics

            print_progress('TEST' if is_test else 'VALID', epoch, self.config.total_epoch, batch_idx, \
                self.num_test_iteration_per_epoch if is_test else self.num_val_iteration_per_epoch, valid_time, \
                self.config.loss, loss.item(), self.config.metrics, this_metrics)
        
        return total_loss, total_metrics
