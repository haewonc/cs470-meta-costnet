import time
import torch 
import numpy as np 
from data.utils import *
from trainer.CoSTNet_trainer import CoSTNetTrainer
from trainer.base_trainer import BaseTrainer
from util.logging import * 

class STResNetTrainer(CoSTNetTrainer):
    def __init__(self, cls, config, args):
        BaseTrainer.__init__(self)
        self.config = config
        self.device = self.config.device
        self.best_metric = 1000
        self.best_model = None
        self.cls = cls
        self.setup_save(args)

    def setup_model(self):
        self.model = self.cls(self.config).to(self.device)
        
    def train_epoch(self, epoch):
        self.model.train()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target, ext) in enumerate(self.train_loader):
            data, target, ext = data.to(self.device), target.to(self.device), ext.to(self.device)
            self.optimizer.zero_grad()

            b, t, w, h, c = data.size()
            data = data.reshape(b*t, w*h*c)
            output = self.model(data)
            output = output.reshape(b, t, w, h, c)

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

    def validate_epoch(self, epoch, is_test):
        self.model.eval()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.config.metrics)*self.config.num_modes)
        this_metrics = np.zeros(len(self.config.metrics)*self.config.num_modes)

        for batch_idx, (data, target, ext) in enumerate(self.test_loader if is_test else self.val_loader):
            data, target, ext = data.to(self.device), target.to(self.device), ext.to(self.device)

            with torch.no_grad():
                b, t, w, h, c = data.size()
                data = data.reshape(b*t, w*h*c)
                output = self.model(data)
                output = output.reshape(b, t, w, h, c)
                
                loss = self.loss(output, target) 
            
            valid_time = time.time() - start_time
            start_time = time.time()

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