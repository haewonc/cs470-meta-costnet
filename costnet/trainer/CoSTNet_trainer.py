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
import pickle

class CoSTNetTrainer(BaseTrainer):
    def __init__(self, cls, config, args):
        super().__init__()
        self.config = config
        self.device = self.config.device
        self.best_metric = 1000
        self.best_model = None
        self.cls = cls
        self.setup_save(args)

    def setup_model(self):
        self.model = self.cls(self.config).to(self.device)
        self.autoencoders = []
        for ckpt in self.config.encoder_ckpts:
            model = AutoEncoderModel(self.config)
            model.load_state_dict(torch.load(ckpt))
            self.autoencoders.append(model.to(self.config.device))

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

    def train_epoch(self, epoch):
        self.model.train()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target, ext) in enumerate(self.train_loader):
            data, target, ext = data.to(self.device), target.to(self.device), ext.to(self.device)
            self.optimizer.zero_grad()

            b, t, w, h, c = data.size()
            encodings = []
            for i in range(c):
                xt = data[...,i].view(b*t, 1, w, h)
                encodings.append(self.autoencoders[i].encoder(xt).reshape(b, t, -1))
            data = torch.concat(encodings, dim=2)

            output = self.model(data, ext)

            decodings = []
            for i in range(output.size(-1)):
                dt = output[...,i].reshape(b*t, self.config.num_features, self.config.num_nodes, self.config.num_nodes)
                decodings.append(self.autoencoders[i].decoder(dt))
            output = torch.stack(decodings, dim=4).view(b, t, w, h, c)

            loss = self.loss(output, target) 
            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()
            training_time = time.time() - start_time
            start_time = time.time()

            total_loss += loss.item()

            for i in range(output.size(-1)):
                output[...,i] *= self.data_max[i]
                target[...,i] *= self.data_max[i]

                output[...,i] += self.data_min[i]
                target[...,i] += self.data_min[i]

            output = output.detach().cpu()
            target = target.detach().cpu()

            this_metrics = self._eval_metrics(output, target)
            total_metrics += this_metrics

            print_progress('TRAIN', epoch, self.config.total_epoch, batch_idx, self.num_train_iteration_per_epoch, training_time, self.config.loss, loss.item(), self.config.metrics, this_metrics)

        return total_loss, total_metrics

    def validate_epoch(self, epoch=None, is_test=True):
        self.model.eval()
        total_loss = 0
        total_metrics = np.zeros(len(self.config.metrics)*self.config.num_modes)
        this_metrics = np.zeros(len(self.config.metrics)*self.config.num_modes)

        for batch_idx, (data, target, ext) in enumerate(self.test_loader if is_test else self.val_loader):
            data, target, ext = data.to(self.device), target.to(self.device), ext.to(self.device)

            with torch.no_grad():
                b, t, w, h, c = data.size()
                encodings = []
            
                for i in range(c):
                    xt = data[...,i].view(b*t, 1, w, h)
                    encodings.append(self.autoencoders[i].encoder(xt).reshape(b, t, -1))

                data = torch.concat(encodings, dim=2)

                output = self.model(data, ext)

                decodings = []
                for i in range(output.size(-1)):
                    dt = output[...,i].reshape(b*t, self.config.num_features, self.config.num_nodes, self.config.num_nodes)
                    decodings.append(self.autoencoders[i].decoder(dt))
                output = torch.stack(decodings, dim=4).view(b, t, w, h, c)
                
                loss = self.loss(output, target) 
                

            total_loss += loss.item()

            output = output.detach().cpu()
            target = target.detach().cpu()

            for i in range(output.size(-1)):
                output[...,i] *= self.data_max[i]
                target[...,i] *= self.data_max[i]
                
                output[...,i] += self.data_min[i]
                target[...,i] += self.data_min[i]
                if len(self.config.metrics) == 1:
                    this_metrics[i] = self._eval_metrics(output[...,i], target[...,i])[0]
                else:
                    this_metrics[i*len(self.config.metrics):(i+1)*len(self.config.metrics)] = self._eval_metrics(output[...,i], target[...,i])
            total_metrics += this_metrics

        return total_loss, total_metrics
    
    def validate(self, epoch=None, is_test=False):
        metric_names = [f'{i}_{j}'.upper() for i in self.config.modes_name for j in self.config.metrics]
        total_loss, total_metrics = self.validate_epoch(epoch, is_test)
        avg_loss = total_loss / len(self.test_loader if is_test else self.val_loader)
        avg_metrics = total_metrics / len(self.test_loader if is_test else self.val_loader)
        self.history['test'].append([avg_loss, avg_metrics])
        if avg_metrics[0] < self.best_metric:
            self.best_model = self.model.state_dict()
            self.best_metric = avg_metrics[0] 
        print_total('TEST' if is_test else 'VALID', epoch, self.config.total_epoch, self.config.loss, avg_loss, metric_names, avg_metrics)