import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import learn2learn as l2l
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from learn2learn.nn import Scale
from learn2learn.optim.transforms.kronecker_transform import KroneckerTransform
from learn2learn.optim.transforms.metacurvature_transform import MetaCurvatureTransform
from learn2learn.optim.transforms.module_transform import (
    ModuleTransform,
    ReshapedTransform,
)
from learn2learn.optim.transforms.transform_dictionary import TransformDictionary
from transformers import SegformerForSemanticSegmentation, SegformerConfig

from utils.heatmaps import *
from utils.loss import HeatmapMSELoss
from utils.utils import *
from .evaluation import AverageMeter, distance_error, batch_spatial_angular_similarity

# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/12-meta-learning.html

# class Scale(torch.nn.Module):
#     def __init__(self, num_params):
#         super().__init__()
#         self.scale = torch.nn.Parameter(torch.ones(num_params, requires_grad=True))

#     def forward(self, model):
#         print(model)
#         for param, scale in zip(model.parameters(), self.scale):
#             param.data.mul_(scale)
#         return model
class BaseSystem(pl.LightningModule, ABC):
    def __init__(self, network: str, lr: float, adaptation_steps: int, fast_lr: float, 
                 std: float, first_order: bool, allow_nograd: bool, 
                 shot: int, way: int,
                 algorithm: str, allow_unused: bool,
                 encoder_name: str, in_channels: int, classes: int, activation: str,
                 device: str):
        super().__init__()
        self.network = network
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
        
        self.encoder_name = encoder_name
        self.in_channels = in_channels                                                                                                                                                                             
        self.classes = classes
        self.activation = activation
        self.allow_unused = allow_unused
        self.model = None
        self.img_size = None

    def _get_model(self):
        model = get_basemodel(self.network, self.encoder_name, self.in_channels, self.classes, self.activation)

        if self.algorithm == 'MAML':
            model = l2l.algorithms.MAML(model, 
                                        lr=self.fast_lr, 
                                        first_order=self.first_order, 
                                        allow_nograd=self.allow_nograd,
                                        allow_unused=self.allow_unused)
        elif self.algorithm == 'MetaSGD':
            model = l2l.algorithms.MetaSGD(model,
                                           lr=self.fast_lr,
                                           first_order=self.first_order,)
        elif self.algorithm == 'MetaCurvature':
            model = l2l.algorithms.GBML(model, 
                                        lr=self.fast_lr,
                                        transform=MetaCurvatureTransform,
                                        adapt_transform=False,
                                        first_order=self.first_order,
                                        )
        self.model = model

    def configure_optimizers(self):
        if self.model is None:
            self._get_model()
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = None
        return [self.optimizer], []

    def training_step(self, batch, batch_idx):
        # This should be implemented by subclasses
        return None

class MetaLearnerModule(pl.LightningModule, ABC):
    lr = 5e-3
    fast_lr = 0.03
    std = 7
    way = 3
    shot = 5

    adaptation_steps = 10

    def configure_optimizers(self):
        if self.network.lower() == 'segformer':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer], []
    
    def get_preds(self, model, data, shape):
        preds = model(data)
        preds = preds.logits if hasattr(preds, "logits") else preds

        # 필요 시 크기 보정
        if preds.shape[-2:] != shape[-2:]:
            preds = F.interpolate(preds, size=(shape[-2], shape[-1]), mode='bilinear', align_corners=False)
        return preds

    def training_step(self, batch, batch_idx):
        train_loss, train_mde, train_sas = self.meta_learn(
            batch, batch_idx, self.train_ways, self.train_shots, self.train_queries
        )
        self.log(
            "train_loss",
            train_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train_MDE",
            train_mde.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_SAS",
            train_sas.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        return train_loss
        
class GenericMetaLearner(MetaLearnerModule):
    def __init__(self, model, network: str, first_order: bool, allow_nograd: bool, 
                 algorithm: str, allow_unused: bool, 
                 device: str, **kwargs):
        super().__init__()
        self.model = model
        self.network = network
        
        self.lr = kwargs.get("lr", MetaLearnerModule.lr)
        self.fast_lr = kwargs.get("fast_lr", MetaLearnerModule.fast_lr)
        self.std = kwargs.get("std", MetaLearnerModule.std)
        self.shot = kwargs.get("shot", MetaLearnerModule.shot)
        self.way = kwargs.get("way", MetaLearnerModule.way)
        
        self.algorithm = algorithm
        self.adaptation_steps = kwargs.get("adaptation_steps", MetaLearnerModule.adaptation_steps)
        self.loss = HeatmapMSELoss(use_target_weight=True)

        self.allow_nograd = allow_nograd
        self.first_order = first_order       
        self.allow_unused = allow_unused
        
        self.img_size = None
        # self.save_hyperparameters = True

    def fast_adapt(self, learner, adaptation_data, adaptation_labels_heatmap):
        for step in range(self.adaptation_steps):
            pred = learner(adaptation_data)
            adaptation_error = self.loss(pred, 
                                        adaptation_labels_heatmap, 
                                        target_weight=torch.tensor([[1,1,1,1,0.01]]).to(pred.device)
                                        )
            print(f"{step}: {adaptation_error}")
            learner.adapt(adaptation_error)


        
    def outer_loop(self, batch, mode='train'):
        losses = AverageMeter()
        mean_distance_error = AverageMeter()    

        data, labels = batch['data'], batch['label']
        self.img_size = data[0].shape[1:]
        
        indices = np.zeros(data.size(0), dtype=bool)
        indices[np.arange(self.shot*self.way)*2] = True
        support_data  = data[indices]
        support_labels = labels[indices]
        query_data   = data[~indices]
        query_labels = labels[~indices]

        learner = l2l.clone_module(self.model)
        
        # Render heatmaps
        support_labels_hm = render_gaussian_dot_f(
            support_labels.flip(dims=[2]),
            torch.tensor([self.std, self.std], device=support_data.device),
            self.img_size).float()
        bg = 1 - support_labels_hm.sum(dim=1, keepdim=True).clamp(0,1)
        support_labels_hm = torch.cat((support_labels_hm, bg), 1)

        query_labels_hm = render_gaussian_dot_f(
            query_labels.flip(dims=[2]),
            torch.tensor([self.std, self.std], device=query_data.device),
            self.img_size).float()
        bg2 = 1 - query_labels_hm.sum(dim=1, keepdim=True).clamp(0,1)
        query_labels_hm = torch.cat((query_labels_hm, bg2), 1)
        
         # Fast adaptation
        for step in range(self.adaptation_steps):
            preds = self.get_preds(learner, support_data, support_data.shape)
            error = self.loss(
                preds,
                support_labels_hm,
                target_weight=torch.tensor([[1,1,1,1,0.01]], device=preds.device))
            learner.adapt(error)
            
        predictions = self.get_preds(learner, query_data, query_data.shape)
        # loss = self.loss(
        #     preds,
        #     query_labels_hm,
        #     target_weight=torch.tensor([[1,1,1,1,0.01]], device=preds.device))
        valid_error = self.loss(predictions, 
                                query_labels_hm,
                                target_weight=torch.tensor([[1,1,1,1,0.01]], device=preds.device))
        preds_coor = heatmap2coor(predictions[:,:4,...])
        query_sample = {'data': query_data, 'label': query_labels}
        valid_mde = distance_error(
            query_sample,
            preds_coor
            )
        valid_sas = batch_spatial_angular_similarity(
                                                     query_sample, 
                                                     preds_coor
                                                     )
        
        return valid_error, valid_mde, valid_sas


    def training_step(self, batch, batch_idx):
        self.outer_loop(batch, mode='train')
        return None
        # Returning None means we skip the default training optimizer steps by PyTorch Lightning
    
    # @staticmethod
    # def split_data(batch, labels):
    #     pass
    # def adapt_on_new_task(self, batch, mode='train'):
    #     adaptation_data, adaptation_labels = batch['data'], batch['label']
    #     adaptation_labels_heatmap = render_gaussian_dot_f(
    #                 adaptation_labels.flip(dims=[2]), # xy 2 yx 
    #                 torch.tensor([self.std, self.std], dtype=torch.float32).to(adaptation_data.device), 
    #                 self.img_size,
    #             ).to(torch.float)
    #     background = 1 - adaptation_labels_heatmap.sum(dim=1).unsqueeze(1).clip(0,1)
    #     adaptation_labels_heatmap = torch.concat((adaptation_labels_heatmap,background), 1)

    #     for _ in range(self.adaptation_steps):
    #         pred = self.model(adaptation_data)
    #         adaptation_error = self.loss(pred, 
    #                                      adaptation_labels_heatmap, 
    #                                      target_weight=torch.tensor([[1,1,1,1,0.01]]).to(pred.device)
    #                                     )
    #         self.optimizer.zero_grad()
    #         adaptation_error.backward()
    #         self.optimizer.step()
    #         print(f"{adaptation_error}")
    # def valdiation_step(self, batch, batch_idx):
    #     torch.set_grad_enabled(True)
    #     self.outer_loop(batch, mode="val")
    #     torch.set_grad_enabled(False)
    # def test_step(self, batch, batch_idx):
    #     torch.set_grad_enabled(True)
    #     self.adapt(batch, mode="test")
    #     torch.set_grad_enabled(False)
    
class DecoderHead(nn.Module):
    def __init__(self, decoder, seg_head):
        super().__init__()
        self.decoder = decoder
        self.seg_head = seg_head

    def forward(self, features_list):
        # features_list: List[Tensor], 그대로 *로 풀어서 decoder에 전달
        x = self.decoder(*features_list)
        return self.seg_head(x)



class ANILMetaLearner(MetaLearnerModule):
    def __init__(self, model, network: str, first_order: bool, 
                 algorithm: str, device: str, **kwargs):
        super().__init__()
        self.model = model
        self.network = network
        
        self.lr = kwargs.get("lr", MetaLearnerModule.lr)
        self.fast_lr = kwargs.get("fast_lr", MetaLearnerModule.fast_lr)
        self.std = kwargs.get("std", MetaLearnerModule.std)
        self.shot = kwargs.get("shot", MetaLearnerModule.shot)
        self.way = kwargs.get("way", MetaLearnerModule.way)
        
        self.algorithm = algorithm
        self.adaptation_steps = kwargs.get("adaptation_steps", MetaLearnerModule.adaptation_steps)
        self.loss = HeatmapMSELoss(use_target_weight=True)

        self.first_order = first_order
        self.img_size = None
        if self.network.lower() == 'segformer':
            self.feature = model.segformer
            self.classifier = model.decode_head
        else:
            self.feature = model.encoder
            self.classifier = DecoderHead(model.decoder, model.segmentation_head)

        # self.feature, self.classifier = get_model_for_ANIL(self.network, self.model)
        # self._get_model(model)
        
            # self.feature = nn.Sequential(*list(model.children())[:-1])
            # self.classifier = list(model.children())[-1]
    def on_save_checkpoint(self, checkpoint):
        ...
        # print(f"ANILMetaLearner: on_save_checkpoint hook called. Checkpoint keys: {checkpoint.keys()}")
        # print(f"Epoch: {checkpoint['epoch']}")
    # ANILMetaLearner 클래스 내부에 추가
    def on_train_epoch_end(self):
        print(f"ANILMetaLearner: Executed on_train_epoch_end for epoch {self.current_epoch}.")

    def on_fit_end(self):
        # 이 메소드는 trainer.fit()이 완전히 종료된 후 호출됩니다.
        print(f"ANILMetaLearner: Executed on_fit_end. Training is complete.")
        # last.ckpt가 실제로 저장되었는지 ModelCheckpoint 객체를 통해 확인
        from pytorch_lightning.callbacks import ModelCheckpoint # isinstance를 위해 추가

        for cb in self.trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                print(f"  ModelCheckpoint's last_model_path: {cb.last_model_path}")
                if cb.last_model_path and os.path.exists(cb.last_model_path):
                    print(f"  Confirmed: last.ckpt exists at {cb.last_model_path}")
                else:
                    print(f"  Warning: last.ckpt does NOT exist or path is incorrect according to ModelCheckpoint ({cb.last_model_path}).")

    @torch.enable_grad()
    def outer_loop(self, batch, mode='train'):
        """One meta‐training iteration for ANIL."""
        self.feature.train()
        learner = l2l.clone_module(self.classifier)
        learner = l2l.algorithms.MAML(learner, lr=self.fast_lr, first_order=self.first_order)
        learner.train()
        
        data, labels = batch['data'].to(self.device), batch['label'].to(self.device)
        self.img_size = data.shape[-2:]

        # Split data
        indices = np.zeros(data.size(0), dtype=bool)
        indices[np.arange(self.shot*self.way)*2] = True
        support_data  = data[indices]
        support_labels = labels[indices]
        query_data   = data[~indices]
        query_labels = labels[~indices]

        # Render heatmaps
        support_labels_hm = render_gaussian_dot_f(
            support_labels.flip(dims=[2]),
            torch.tensor([self.std, self.std], device=support_data.device),
            self.img_size).float()
        bg = 1 - support_labels_hm.sum(dim=1, keepdim=True).clamp(0,1)
        support_labels_hm = torch.cat((support_labels_hm, bg), 1)

        query_labels_hm = render_gaussian_dot_f(
            query_labels.flip(dims=[2]),
            torch.tensor([self.std, self.std], device=query_data.device),
            self.img_size).float()
        bg2 = 1 - query_labels_hm.sum(dim=1, keepdim=True).clamp(0,1)
        query_labels_hm = torch.cat((query_labels_hm, bg2), 1)

        ## extracting feature 
        if self.network == 'segformer':
            support_feats = self.feature(support_data, output_hidden_states=True).hidden_states
            query_feats = self.feature(query_data, output_hidden_states=True).hidden_states
        elif self.network.lower() == 'deeplabv3plus':
            support_feats = self.feature(support_data)
            query_feats = self.feature(query_data)

        for step in range(self.adaptation_steps):
            preds = self.get_preds(learner, support_feats, support_data.shape)
            error = self.loss(
                preds,
                support_labels_hm,
                target_weight=torch.tensor([[1,1,1,1,0.01]], device=preds.device))
            learner.adapt(error)

        # Evaluate on query data
        predictions = self.get_preds(learner, query_feats, query_data.shape)
        valid_error = self.loss(
            predictions,
            query_labels_hm,
            target_weight=torch.tensor([[1,1,1,1,0.01]], device=predictions.device))
        preds_coor = heatmap2coor(predictions[:,:4,...])
        valid_mde = distance_error(
            {'data': query_data, 'label': query_labels},
            preds_coor
            )
        valid_sas = batch_spatial_angular_similarity(
                                                     {'data': query_data, 'label': query_labels}, 
                                                     preds_coor
                                                     )
        # self.log(f"{mode}_loss", valid_error.item(), prog_bar=True)
        # self.log(f"{mode}_mde", valid_mde.mean(), prog_bar=True)
        # self.log(f"{mode}_sas", valid_sas.mean(), prog_bar=True)
        return valid_error, valid_mde, valid_sas


    def training_step(self, batch, batch_idx):
        mode='train'
        train_loss, train_mde, train_sas = self.outer_loop(batch, mode=mode)
        self.log(f"{mode}_loss", train_loss.item(), prog_bar=True)
        self.log(f"{mode}_mde", train_mde.mean(), prog_bar=True)
        self.log(f"{mode}_sas", train_sas.mean(), prog_bar=True)

        return train_loss
        
def get_basemodel(network: str = 'Unet', 
                  encoder_name: str = 'tu-efficientnet_b0', 
                  in_channels: int = 3, 
                  classes: int = 5, 
                  activation: str = 'softmax') -> nn.Module:
    if network.lower() == 'Unet'.lower():
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            
        )
    elif network.lower() == 'DeepLabV3Plus'.lower():
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    elif network.lower() == 'segformer'.lower():
        arch = 'b0'
        model_name = f"nvidia/segformer-{arch}-finetuned-ade-512-512"
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, num_labels=5, ignore_mismatched_sizes=True)

    else:
        raise ValueError(f"Unsupported network type: {network}")

    return model


def get_meta_model(CFG):
    network = CFG['network'].lower()
    if network == 'DeepLabV3Plus'.lower():   
        model = smp.DeepLabV3Plus(
            encoder_name=CFG['backbone'],
            in_channels=3,
            classes=5,
            activation='softmax'
        )
    elif network == 'Unet'.lower():
        model = smp.Unet(
            encoder_name=CFG['backbone'],
            in_channels=3,
            classes=5,
            activation='softmax'
        )
    elif network == 'segformer':
        arch = 'b0'
        
        model_name = f"nvidia/segformer-{arch}-finetuned-ade-512-512"
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, num_labels=5, ignore_mismatched_sizes=True)

    else:
        raise ValueError(f"Unsupported network type: {network}")

    model = l2l.algorithms.MAML(model, lr=CFG['fast_lr'], first_order=True, allow_nograd=True)
    model.to(CFG['device'])
    return model

def get_model(CFG):
    model = get_basemodel(CFG['network'], CFG['backbone'], CFG['channels'])

    if CFG['algorithm'] == 'MAML':
        model = l2l.algorithms.MAML(model, 
                                    lr=CFG['fast_lr'], 
                                    first_order=CFG['first_order'], 
                                    allow_nograd=CFG['allow_nograd'])
    elif CFG['algorithm'] == 'MetaSGD':
        model = l2l.algorithms.MetaSGD(model,
                                       lr=CFG['fast_lr'], 
                                       first_order=CFG['first_order'],)
                                    #    allow_unused=self.allow_unused)
    elif CFG['algorithm'] == 'MetaCurvature':
        model = l2l.algorithms.GBML(model, 
                                    lr=CFG['fast_lr'], 
                                    transform=MetaCurvatureTransform,
                                    adapt_transform=False,
                                    first_order=CFG['first_order'],
                                    )
    elif CFG['algorithm'] == 'ANIL':
        if CFG['network'].lower() == 'segformer'.lower():
            encoder = model.segformer.encoder
            head = model.decode_head
            # head = list(model.children())[-1]
            model = {
                'features': encoder.to(CFG['device']),
                'head': head.to(CFG['device'])
            }
        else:
            features = list(model.children())[:-1]
            head = list(model.children())[-1]
            model = {
                'features': features.to(CFG['device']),
                'head': head.to(CFG['device'])
            }
    return model


def get_model_for_ANIL(network, model):
    if network.lower() == 'segformer':
        feature = model.segformer
        classifier = model.decode_head
    else:
        feature = model.encoder
        classifier = DecoderHead(model.decoder, model.segmentation_head)
    return feature, classifier