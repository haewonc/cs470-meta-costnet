import time
import math
import torch 
import torch.nn as nn 
import numpy as np 
from data.utils import *
from data.datasets import CoSTNetDataset
from model.AutoEncoder import AutoEncoderModel
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer
from util.logging import * 
import json
import pickle

class ConvLSTMTrainer(BaseTrainer):
    def __init__(self, cls, config, args):
        super().__init__()
        self.config = config
        self.device = self.config.device
        self.use_modes = True
        self.best_metric = 1000
        self.best_model = None
        self.cls = cls
        self.setup_save(args)
    
    def setup_model(self):
        self.model = self.cls(self.config).to(self.device)
        
    def load_dataset(self):
        with open('{}/{}.pkl'.format(self.config.dataset_dir, self.config.dataset_name), 'rb') as file:
            data = pickle.load(file)
        datasets = {}
        self.data_min = data['min']
        self.data_max = data['max']
        for category in ['train', 'test']:
            demand = data[category]["data"]
            t, n, n, s = demand.shape
            x, y = seq2instance(np.reshape(demand, (t, n*n*s)), self.config.num_his, self.config.num_pred)
            x, y = np.reshape(x, (x.shape[0], self.config.num_his, n, n, s)), np.reshape(y, (y.shape[0], self.config.num_pred, n, n, s))
            ext_x, ext_y = seq2instance(data[category]["ext"], self.config.num_his, self.config.num_pred)
            datasets[category] = {
                "x": x, "y": y, "ext_x": ext_x, "ext_y": ext_y
            }
        return datasets, n

    def compose_dataset(self):
        datasets, num_nodes = self.load_dataset()
        self.config.num_nodes = num_nodes
        for category in ['train', 'test']:
            datasets[category] = CoSTNetDataset(datasets[category])
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
        if self.use_modes:
            super().train()
        else:
            data_names = ['BIKE_PICKUP', 'BIKE_DROP', 'TAXI_PICKUP', 'TAXI_DROP']
            for i in range(4):
                self.setup_model()
                self.setup_train()
                print(toGreen(f'\n{data_names[i]} TRAINING START'))
                self.config.num_modes = 1 
                self.config.modes_name = [data_names[i]]
                for epoch in range(self.config.total_epoch):
                    total_loss, total_metrics = self.train_epoch(epoch, mode=i)
                    avg_loss = total_loss / len(self.train_loader)
                    avg_metrics = total_metrics / len(self.train_loader)
                    self.history['train'].append({'loss': avg_loss, 'metrics': list(avg_metrics)})
                    print_total('TRAIN', epoch, self.config.total_epoch, self.config.loss, avg_loss, self.config.metrics, avg_metrics)
                    if epoch % self.config.valid_every_epoch == 0:
                        self.validate(epoch, is_test=False, mode=i)
                print(toGreen(f'\n{data_names[i]} TRAINING END'))
                self.validate(epoch, is_test=False, mode=i)
                print(toGreen(f'Saving the model with {self.config.metrics[0]} {round(self.best_metric,2)}'))
                torch.save(self.best_model, '../results/saved_models/{}/{}.pth'.format(self.save_name, data_names[i].lower()))
                with open('../results/saved_models/{}/history_{}.json'.format(self.save_name, data_names[i].lower()), 'w') as file:
                    json.dump(self.history, file)
                    self.history = {}
                    self.history['train'] = []
                    self.history['test'] = []
                self.best_metric = 1000

    def train_epoch(self, epoch, mode=None):
        self.model.train()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target, ext) in enumerate(self.train_loader):
            data, target, ext = data.to(self.device), target.to(self.device), ext.to(self.device)
            self.optimizer.zero_grad()

            if mode is not None:
                data = data[..., mode].unsqueeze(-1)
                target = target[..., mode].unsqueeze(-1)
            
            b, t, w, h, c = data.size() # (b, 12?, 20, 20, 4)
            output = self.model(data.permute(0, 1, 4, 2, 3).reshape(b, t*c, w, h))
            output = output.reshape(b, t, c, w, h).permute(0, 1, 3, 4, 2)
            loss = self.loss(output, target) 
            loss.backward()

            self.optimizer.step()
            training_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()

            for i in range(output.size(-1)):
                if mode is not None:
                    idx = mode 
                else:
                    idx = i
                output[...,i] *= self.data_max[idx]
                target[...,i] *= self.data_max[idx]

                output[...,i] += self.data_min[idx]
                target[...,i] += self.data_min[idx]

            output = output.detach().cpu()
            target = target.detach().cpu()

            this_metrics = self._eval_metrics(output, target)
            total_metrics += this_metrics

            print_progress('TRAIN', epoch, self.config.total_epoch, batch_idx, self.num_train_iteration_per_epoch, training_time, self.config.loss, loss.item(), self.config.metrics, this_metrics)

        return total_loss, total_metrics

    def validate_epoch(self, epoch, is_test, mode=None):
        self.model.eval()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.config.metrics)*self.config.num_modes)
        this_metrics = np.zeros(len(self.config.metrics)*self.config.num_modes)
        metric_names = [f'{i}_{j}'.upper() for j in self.config.metrics for i in self.config.modes_name]

        for batch_idx, (data, target, ext) in enumerate(self.test_loader if is_test else self.val_loader):
            data, target, ext = data.to(self.device), target.to(self.device), ext.to(self.device)

            with torch.no_grad():
                if mode is not None:
                    data = data[..., mode].unsqueeze(-1)
                    target = target[..., mode].unsqueeze(-1)
                b, t, w, h, c = data.size()
                output = self.model(data.permute(0, 1, 4, 2, 3).reshape(b, t*c, w, h))
                output = output.reshape(b, t, c, w, h).permute(0, 1, 3, 4, 2)
                loss = self.loss(output, target) 
            
            valid_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()

            output = output.detach().cpu()
            target = target.detach().cpu()

            for i in range(output.size(-1)):
                if mode is not None:
                    idx = mode 
                else:
                    idx = i
                output[...,i] *= self.data_max[idx]
                target[...,i] *= self.data_max[idx]

                output[...,i] += self.data_min[idx]
                target[...,i] += self.data_min[idx]
                
                if len(self.config.metrics) == 1:
                    this_metrics[i] = self._eval_metrics(output[...,i], target[...,i])[0]
                else:
                    this_metrics[i:i*len(self.config.metrics)] = self._eval_metrics(output[...,i], target[...,i])
            total_metrics += this_metrics

            print_progress('TEST' if is_test else 'VALID', epoch, self.config.total_epoch, batch_idx, \
                self.num_test_iteration_per_epoch if is_test else self.num_val_iteration_per_epoch, valid_time, \
                self.config.loss, loss.item(), metric_names, this_metrics)
        
        return total_loss, total_metrics
    
    def validate(self, epoch, is_test=False, mode=None):
        metric_names = [f'{i}_{j}'.upper() for j in self.config.metrics for i in self.config.modes_name]
        total_loss, total_metrics = self.validate_epoch(epoch, is_test, mode)
        avg_loss = total_loss / len(self.test_loader if is_test else self.val_loader)
        avg_metrics = total_metrics / len(self.test_loader if is_test else self.val_loader)
        if avg_metrics[0] < self.best_metric:
            self.best_model = self.model.state_dict()
            self.best_metric = avg_metrics[0] 
        self.history['test'].append({'loss': avg_loss, 'metrics': list(avg_metrics)})
        print_total('TEST' if is_test else 'VALID', epoch, self.config.total_epoch, self.config.loss, avg_loss, metric_names, avg_metrics)