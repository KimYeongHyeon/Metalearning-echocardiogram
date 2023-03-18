import pytorch_lightning as pl
from typing import Optional, Callable, Sequence, Tuple

import os
import pandas as pd
import numpy as np
import glob
import collections
import cv2
from operator import index

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import utils
from utils.heatmaps import *
from utils.meta_dataset import EchoDataset_Meta_heatmap


class MinMaxNormalize(ImageOnlyTransform):
    """
    Min-max normalization
    """

    def apply(self, img, **param):
        # minmax normalize
        # img = (img - img.min()) / (img.max() - img.min())
        img = img / 255.
        return img
class DataModule(pl.LightningDataModule):
    def __init__(self, root_dir: str, batch_size: int, img_size: Sequence[int],
                 augment: bool, num_channels: int, num_workers: int,
                 tasks, target, num_tasks, shot: int, way:int):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.tasks = tasks
        self.target = target
        self.num_tasks = num_tasks
        self.shot = shot
        self.way = way
        self.num_channels = num_channels
        self.num_workers = num_workers
        self.augment = augment
        self.prepare_data_per_node = False


        self._train_dataset = None
        self._val_dataset = None
    def _log_hyperparams(self):
        pass
    @staticmethod
    def get_transform(img_size:Tuple[int], is_train: bool):
        if is_train:
            transforms = A.Compose([
                A.Resize(*img_size),
                A.SafeRotate(limit=30),
                MinMaxNormalize(p=1),
                ToTensorV2(),
            ],
                keypoint_params=A.KeypointParams(format='xy')
            )
        else:
            transforms = A.Compose([
                A.Resize(*img_size),
                MinMaxNormalize(p=1),
                ToTensorV2(),
            ],
                keypoint_params=A.KeypointParams(format='xy')
            )
        return transforms

    @property
    def train_dataset(self):
        # if self._train_dataset is None:
        transforms = self.get_transform(self.img_size, is_train=True)
        self._train_dataset = EchoDataset_Meta_heatmap(self.root_dir, self.tasks, 
                                                    transforms=transforms, 
                                                    split='train+val', 
                                                    shot=self.shot*2, 
                                                    num_channels=self.num_channels)
        return self._train_dataset
    
    @property
    def val_dataset(self):
        if self._val_dataset is None:
            transforms = self.get_transform(self.img_size, is_train=False)
            self._val_dataset = EchoDataset_Meta_heatmap(self.root_dir, self.tasks, 
                                                        transforms=transforms, 
                                                        split='val', 
                                                        shot=self.shot, 
                                                        num_channels=self.num_channels)
        return self._val_dataset

    @property
    def test_dataset(self):
        transforms = self.get_transform(self.img_size, is_train=False)
        self._test_dataset = EchoDataset_heatmap(self.root_dir, 
                                                split=str(CFG['shot']),
                                                shot=CFG['shot'],
                                                transforms=transforms, 
                                                num_channels=self.num_channels,
                                                task_list = [CFG['target']])
    
    def train_dataloader(self):       
        return DataLoader(self.train_dataset, batch_size=self.shot*self.way*2, 
                          shuffle=False, num_workers=self.num_workers)#, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, 
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, 
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)