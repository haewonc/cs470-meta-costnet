import os
import numpy as np
from abc import abstractmethod
from util.logging import * 
import torch
import torch.nn as nn
from data.utils import *
import importlib
import shutil
import json
from datetime import datetime

class BaseTrainer:
    '''
    Base class for all trainers
    '''
    def __init__(self):
        self.history = {}
        self.history['train'] = []
        self.history['test'] = []
        self.best_metric = 1000
        self.best_model = None
    
    def load_model(self, ckpt):
        self.model.load_state_dict(torch.load(ckpt))

    @abstractmethod 
    def compose_dataset(self, *inputs):
        raise NotImplementedError
    
    @abstractmethod 
    def compose_dataset(self, *inputs):
        raise NotImplementedError
    
    @abstractmethod 
    def compose_loader(self, *inputs):
        raise NotImplementedError

    @abstractmethod 
    def train_epoch(self, *inputs):
        raise NotImplementedError

    def validate(self, epoch, is_test=False):
        total_loss, total_metrics = self.validate_epoch(epoch, is_test)
        avg_loss = total_loss / len(self.test_loader if is_test else self.val_loader)
        avg_metrics = total_metrics / len(self.test_loader if is_test else self.val_loader)
        if avg_metrics[0] < self.best_metric:
            self.best_model = self.model.state_dict()
            self.best_metric = avg_metrics[0] 
        self.history['test'].append({'loss': avg_loss, 'metrics': list(avg_metrics)})
        print_total('TEST' if is_test else 'VALID', epoch, self.config.total_epoch, self.config.loss, avg_loss, self.config.metrics, avg_metrics)

    @abstractmethod 
    def validate_epoch(self, epoch, is_test):
        raise NotImplementedError
    
    def train(self):
        print(toGreen('\nSETUP TRAINING'))
        self.setup_train()
        print(toGreen('\nTRAINING START'))
        for epoch in range(self.config.start_epoch, self.config.total_epoch):
            total_loss, total_metrics = self.train_epoch(epoch)
            avg_loss = total_loss / len(self.train_loader)
            avg_metrics = total_metrics / len(self.train_loader)
            self.history['train'].append({'loss': avg_loss, 'metrics': list(avg_metrics)})
            print_total('TRAIN', epoch, self.config.total_epoch, self.config.loss, avg_loss, self.config.metrics, avg_metrics)
            if epoch % self.config.valid_every_epoch == 0:
                self.validate(epoch, is_test=False)
                torch.save(self.model.state_dict(), '../results/saved_models/{}/{}.pth'.format(self.save_name, epoch))
        print(toGreen('\nTRAINING END'))
        self.validate(epoch, is_test=True)
        print(toGreen(f'Saving the model with {self.config.metrics[0]} {round(self.best_metric,2)}'))
        torch.save(self.best_model, '../results/saved_models/{}/best.pth'.format(self.save_name))
    
    def test(self):
        print(toGreen('\nSETUP TEST'))
        self.setup_test()
        print(toGreen('\nTEST START'))
        self.config.total_epochs = 0
        self.validate(0)
        print(toGreen('\nTEST END'))
    
    def setup_test(self):
        # loss, metrics
        try:
            loss_class = getattr(importlib.import_module('evaluation.metrics'), self.config.loss)
            self.loss = loss_class()
            self.metrics = [getattr(importlib.import_module('evaluation.metrics'), met)() for met in self.config.metrics]        
        except:
            print(toRed('No such metric in evaluation/metrics.py'))
            raise 
    

    def setup_train(self):
        # loss, metrics, optimizer, scheduler
        try:
            loss_class = getattr(importlib.import_module('evaluation.metrics'), self.config.loss)
            self.loss = loss_class()
            self.metrics = [getattr(importlib.import_module('evaluation.metrics'), met)() for met in self.config.metrics]        
        except:
            print(toRed('No such metric in evaluation/metrics.py'))
            raise 

        try:
            # TODO Allow different types of optimizer like I did in scheduler 
            optim_class = getattr(importlib.import_module('torch.optim'), self.config.optimizer)
            self.optimizer = optim_class(self.model.parameters(), lr=self.config.learning_rate, weight_decay = 1e-6)
        except:
            print(toRed('Error loading optimizer: {}'.format(self.config.optimizer)))
            raise 

        try: 
            if self.config.scheduler is not None:
                scheduler_class = getattr(importlib.import_module('torch.optim.lr_scheduler'), self.config.scheduler)
                scheduler_args = self.config.scheduler_args 
                scheduler_args['optimizer'] = self.optimizer
                self.lr_scheduler = scheduler_class(**scheduler_args)
            else: 
                self.lr_scheduler = None
        except:
            print(toRed('Error loading scheduler: {}'.format(self.config.scheduler)))
            raise 

        print_setup(self.config.loss, self.config.metrics, self.config.optimizer, self.config.scheduler)

    def _eval_metrics(self, output, target):
        acc_metrics = []
        for metric in self.metrics:
            if isinstance(metric, nn.Module):
                with torch.no_grad():
                    acc_metrics.append(metric(output, target).numpy())
            else:
                acc_metrics.append(metric(output, target))

        return np.array(acc_metrics)

    def setup_save(self, args):
        self.save_name = '{}_{}'.format(args.model, datetime.now().strftime("%d_%H_%M_%S"))
        self.save_dir = '../results/saved_models/{}/'.format(self.save_name)
        os.makedirs(self.save_dir)
        shutil.copy('config/{}.py'.format(args.config), self.save_dir + 'config.txt')
        with open(self.save_dir + 'cmd_args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        