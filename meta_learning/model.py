import os
import numpy as np
from abc import ABC, abstractmethod

from typing import Optional, Tuple, List
import pytorch_lightning as pl
import torch 
import torch.nn as nn
import learn2learn as l2l
from utils.loss import HeatmapMSELoss
from utils.heatmaps import *
from utils.utils import *

from utils.evaluation import AverageMeter
import segmentation_models_pytorch as smp
# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/12-meta-learning.html

class BaseSystem(pl.LightningModule, ABC):
    def __init__(self, lr: float, adaptation_steps: int, fast_lr: float, 
                 std: float, first_order: bool, allow_nograd: bool, 
                 shot: int, way: int,
                 algorithm: str,
                 encoder_name: str, in_channels: int, classes: int, activation: str):
        super().__init__()
        
        self.lr = lr
        self.fast_lr = fast_lr
        self.std = std
        self.shot = shot
        self.way = way
        
        self.adaptation_steps = adaptation_steps
        self.loss = HeatmapMSELoss(use_target_weight=True)
        self.algorithm = algorithm


        self.allow_nograd = allow_nograd
        self.first_order = first_order
        self.adapt_transform = adapt_transform
        
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes
        self.activation = activation

        self.model = None
        self.img_size = None
        # self.save_hyperparameters = True

    def _get_model(self):
        model = smp.DeepLabV3Plus(
                encoder_name = self.encoder_name,
                in_channels = self.in_channels,
                classes = self.classes,
                activation = self.activation)

        if self.algorithm == 'MAML':
            model = l2l.algorithms.MAML(model, 
                                        lr=self.fast_lr, 
                                        first_order=self.first_order, 
                                        allow_nograd=self.allow_nograd)
        elif self.algorithm == 'GBML':
            model = l2l.algorithms.GBML(model)


        self.model = model
        # TODO: device 할당 수정
        self.model.to('cuda')
        
    def configure_optimizers(self):
        if self.model is None:
            self._get_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = None
        return [self.optimizer], []

    def fast_adapt(self, learner, adaptation_data, adaptation_labels_heatmap):
        for step in range(self.adaptation_steps):
            pred = learner(adaptation_data)
            adaptation_error = self.loss(pred, 
                                         adaptation_labels_heatmap, 
                                         target_weight=torch.tensor([[1,1,1,1,0.01]]).to(pred.device)
                                        )
            # print(pred[0], adaptation_labels_heatmap[0])
            print(f"{step}: {adaptation_error}")
            learner.adapt(adaptation_error)
            # losses.update(adapatation_error.item(), evaluation_data.size(0))
            

    @staticmethod
    def split_data(batch, labels):
        pass

    # TODO: Delete or update
    def run_model(self, local_model, output_weight, output_bias, imgs, labels):
        preds = local_model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float()
        return loss, preds, acc


    def outer_loop(self, batch, mode='train'):
        learner = self.model.clone()
        losses = AverageMeter()
        mean_distance_error = AverageMeter()    
        self.optimizer.zero_grad()

        data, labels = batch['data'], batch['label']
        self.img_size = data[0].shape[1:]

        ############################
        # Separate data into adaptation/evalutation sets
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        adaptation_indices[np.arange(self.shot*self.way)*2] = True
        adaptation_indices = torch.from_numpy(adaptation_indices)
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
        
        # make heatmap
        adaptation_labels_heatmap = render_gaussian_dot_f(
                    adaptation_labels.flip(dims=[2]), # xy 2 yx
                    torch.tensor([self.std]*2, dtype=torch.float32).to(data.device),
                    self.img_size, 
                ).to(torch.float)
        background = 1 - adaptation_labels_heatmap.sum(dim=1).unsqueeze(1).clip(0,1)
        adaptation_labels_heatmap = torch.concat((adaptation_labels_heatmap,background), 1)
        ############################


        ### Adapt the model ### 
        self.fast_adapt(learner, adaptation_data, adaptation_labels_heatmap)
        ### End of adaptation ###

        # 학습된 모델을 기반으로 평가 수행
        evaluation_labels_heatmap = render_gaussian_dot_f(
                    evaluation_labels.flip(dims=[2]), # xy 2 yx
                    torch.tensor([self.std, self.std], dtype=torch.float32).to(data.device),
                    self.img_size,
                ).to(torch.float)
        background = 1 - evaluation_labels_heatmap.sum(dim=1).unsqueeze(1).clip(0,1)
        # print(evaluation_labels.shape,evaluation_labels_heatmap.shape)
        evaluation_labels_heatmap = torch.concat((evaluation_labels_heatmap,background), 1)
        
        # Eval the model
        predictions = self.model(evaluation_data)
        evaluation_error = self.loss(predictions, evaluation_labels_heatmap, 
                                target_weight=torch.tensor([[1,1,1,1,0.01]]).to(predictions.device))
        losses.update(evaluation_error.item(), evaluation_data.size(0))

        evaluation_error.backward()
        self.optimizer.step()
            

        sample = {
            'data': evaluation_data, 
            'label': evaluation_labels
        }
        # TODO: prediction coor 찍어보기 
        mean_distance_error.update(np.mean(
                                distance_error(sample, heatmap2coor(predictions[:,:4,...]))
                                ), 
                        evaluation_data.size(0)
        )

        mode = 'val'
        # self.log("%s_loss" % mode, losses.avg)
        # self.log("%s_accuracy" % mode, mean_distance_error.avg)
        self.log("%s_loss" % mode, losses.avg)
        self.log("%s_accuracy" % mode, mean_distance_error.avg)
        print(losses.avg, mean_distance_error.avg)
    # def adapt(self, batch, mode='test'):
        ## adapt on new batch


    def training_step(self, batch, batch_idx):
        self.outer_loop(batch, mode='train')
        return None
        # Returning None means we skip the default training optimizer steps by PyTorch Lightning
    
    # def valdiation_step(self, batch, batch_idx):
    #     torch.set_grad_enabled(True)
    #     self.outer_loop(batch, mode="val")
    #     torch.set_grad_enabled(False)
    # def test_step(self, batch, batch_idx):
    #     torch.set_grad_enabled(True)
    #     self.adapt(batch, mode="test")
    #     torch.set_grad_enabled(False)