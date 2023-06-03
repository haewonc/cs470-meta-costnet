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
import learn2learn as l2l

class MLAETrainer(BaseTrainer):
    def __init__(self, cls, config, args):
        super().__init__()
        self.config = config
        self.device = self.config.device
        self.cls = cls
        self.setup_save(args)
        self.best_model = None 
        self.best_metric = 1000

    def setup_model(self):
        self.network = self.cls(self.config).to(self.device)
        self.model = l2l.algorithms.MAML(self.network, lr=self.config.learning_rate, first_order=False)

    def load_dataset(self):
        with open('{}/{}.pkl'.format(self.config.dataset_dir, self.config.dataset_name), 'rb') as file:
            data = pickle.load(file)
        self.data_min = data['min']
        self.data_max = data['max']
        return data

    def compose_dataset(self):
        datasets = self.load_dataset()
        datasets['val'] = datasets['train'][-3600:]
        datasets['train'] = datasets['train'][:-3600]
        for category in ['train', 'test', 'val']:
            datasets[category] = AutoEncoderDataset(datasets[category])
        self.num_train_iteration_per_epoch = math.ceil(len(datasets['train']) / self.config.batch_size)
        self.num_val_iteration_per_epoch = math.ceil(len(datasets['val']) / self.config.test_batch_size)
        self.num_test_iteration_per_epoch = math.ceil(len(datasets['test']) / self.config.test_batch_size)
        return datasets['train'], datasets['val'], datasets['test']

    def compose_loader(self):
        self.train_dataset, self.val_dataset, self.test_dataset = self.compose_dataset()
        print(toGreen('Dataset loaded successfully ') + '| Train [{}] Val [{}] Test [{}]'.format(toGreen(len(self.train_dataset)), toGreen(len(self.val_dataset)), toGreen(len(self.test_dataset))))
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.test_dataset, batch_size=self.config.test_batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.test_batch_size, shuffle=False)

    def train(self):
        data_names = ['BIKE_PICKUP', 'BIKE_DROP', 'TAXI_PICKUP', 'TAXI_DROP']
        self.setup_model()
        self.setup_train()

        print(toGreen('Pre-training using meta-learning'))
        
        for epoch in range(self.config.total_epoch):
            total_loss = 0.0
            total_metrics = np.zeros(self.config.num_modes)
            from tqdm import tqdm 
            for _ in tqdm(range(len(self.train_loader))):
                learner = self.model.clone()
                datas, targets = next(iter(self.train_loader))
                for i in range(4):
                    data = datas[..., i].unsqueeze(1).to(self.device)  
                    target = targets[..., i].unsqueeze(1).to(self.device)
                    
                    predictions = learner(data)
                    loss = self.loss(predictions, target)
                    learner.adapt(loss)

                    data, target = next(iter(self.val_loader))
                    data = data[..., i].unsqueeze(1).to(self.device)  
                    target = target[..., i].unsqueeze(1).to(self.device)
                    
                    predictions_val = learner(data)
                    loss_val = self.loss(predictions_val, target)
                    total_loss += loss_val.item()

                    loss_val.backward(retain_graph=True)

                    # Eval metrics
                    predictions_val = predictions_val.detach().cpu()
                    target = target.detach().cpu()

                    predictions_val *= self.data_max[i]
                    target *= self.data_max[i]
                    predictions_val += self.data_min[i]
                    target += self.data_min[i]

                    total_metrics[i] += self._eval_metrics(predictions_val, target)[0]

                for p in self.model.parameters():
                    p.grad.data.mul_(1.0 / self.config.num_modes)
                self.optimizer.step()
                self.lr_scheduler.step()

            total_loss = total_loss / len(self.train_loader)
            total_metrics /= len(self.train_loader)
            print_total('TRAIN', epoch, self.config.total_epoch, self.config.loss, total_loss, data_names, list(total_metrics))
            
            if epoch % self.config.valid_every_epoch == 0:
                avg_metrics = []
                for i in range(4):
                    total_loss, total_metrics = self.validate_epoch(epoch, i, False)
                    avg_loss =  total_loss / len(self.val_loader)
                    avg_metrics.append(total_metrics[0] / len(self.val_loader))
                print_total('VALID', epoch, self.config.total_epoch, self.config.loss, avg_loss, data_names, avg_metrics)
                
        avg_metrics_base = []
        for i in range(4):
            total_loss, total_metrics = self.validate_epoch(epoch, i, True)
            avg_loss =  total_loss / len(self.test_loader)
            avg_metrics_base.append(total_metrics[0] / len(self.test_loader))
        print_total('TEST', epoch, self.config.total_epoch, self.config.loss, avg_loss, data_names, avg_metrics_base)

        print(toGreen(f'Adapting the model'))
        for i in range(4):
            best_model = self.model.module.state_dict()
            best_metric = avg_metrics_base[i]
            fine = self.cls(self.config).to(self.device)
            fine.load_state_dict(best_model)
            optimizer = torch.optim.Adam(fine.parameters(), 1e-4)

            for epoch in range(self.config.adpation_epochs):
                for _, (data, target) in enumerate(self.train_loader):
                    optimizer.zero_grad()

                    data = data[..., i].unsqueeze(1).to(self.device)  
                    target = target[..., i].unsqueeze(1).to(self.device)
                    
                    predictions = fine(data)
                    loss = self.loss(predictions, target)
                    loss.backward()
                    optimizer.step()
                
                total_loss, total_metrics = self.validate_epoch(epoch, i, True, fine)
                avg_loss =  total_loss / len(self.test_loader)
                avg_metrics = total_metrics / len(self.val_loader)

                if avg_metrics[0] < best_metric:
                    best_metric = avg_metrics[0]
                    best_model = fine.state_dict()
                print_total(f'TEST {data_names[i]}', epoch, self.config.adpation_epochs, self.config.loss, avg_loss, self.config.metrics, avg_metrics)

            print(toGreen(f'Saving model with {self.config.metrics[0]} {round(avg_metrics[0],3)}'))
            torch.save(best_model, '../results/saved_models/{}/{}.pth'.format(self.save_name, data_names[i].lower()))


    def validate_epoch(self, epoch, i, is_test, model=None):
        if model is None:
            model = self.model
        model.eval()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target) in enumerate(self.test_loader if is_test else self.val_loader):
            data = data[:, :, :, i].unsqueeze(1).to(self.device)  
            target = target[:, :, :, i].unsqueeze(1).to(self.device)  

            with torch.no_grad():
                output = model(data)
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
