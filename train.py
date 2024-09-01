import os

import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import copy
import gc
import warnings
import argparse
import random
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize
import learn2learn
import learn2learn as l2l
import segmentation_models_pytorch as smp


import wandb

import hydra
from hydra.core.hydra_config import HydraConfig

import logging
from datetime import datetime
from pytz import timezone
import ssl

from colorama import Fore, Back, Style

from collections import defaultdict

from omegaconf import DictConfig, open_dict

from tqdm import tqdm

from utils.loss import HeatmapMSELoss
from utils.meta_dataset import EchoDataset_Meta_heatmap
from utils.train import *
from utils.evaluation import *
from utils.optimizer import *
from utils.utils import *
from utils.heatmaps import *
from utils.train_metalearning import *
from utils.model import get_meta_model, BaseSystem
import albumentations as A
os.environ['WANDB_MODE']='offline'

ssl._create_default_https_context = ssl._create_unverified_context
c_ = Fore.GREEN
sr_ = Style.RESET_ALL

warnings.filterwarnings("ignore")

class TqdmLoggingHandler(logging.StreamHandler):

    """Avoid tqdm progress bar interruption by logger's output to console"""
    # see logging.StreamHandler.eval method:
    # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
    # and tqdm.write method:
    # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

def log(config):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    def timetz(*args):
        return datetime.now(tz).timetuple()
    tz = timezone('Asia/Seoul')
    logging.Formatter.converter = timetz

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # log 출력
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # log를 파일에 출력
    file_handler = logging.FileHandler(os.path.join(config['save_dir'], f"{config['target']}.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # logger.addHandler(TqdmLoggingHandler())

    logger.info(f"python {' '.join(sys.argv)}")
    logger.info("Timezone: " + str(tz))
    logger.info(f"Training Start")
    return logger


def seed_everything(seed: int=0):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
seed_everything()



# def get_model(config):
#     import learn2learn as l2l
#     model = smp.DeepLabV3Plus(
#         encoder_name=config['backbone'],
#         in_channels=3,
#         classes=5,
#         activation='softmax'
#     )

#     model = l2l.algorithms.MAML(model, 
#     lr=config['fast_lr'], first_order=True, allow_nograd=True)
#     model.to(config['device'])
#     return model

@hydra.main(config_path='configs', config_name='main', version_base='1.1')
def main(config: DictConfig):
    # 태스크 정의
    with open_dict(config):
        config.data.tasks = list(set(['PLAX', 'PSAX', '2CH', '4CH']) - set([config.data.target]))

    # logger = log(config)
    # config['trace_func'] = logger.info
    # trace_func = config['trace_func']
    trace_func = print

    model: BaseSystem = hydra.utils.instantiate(config.model)
    
    ## Load Dataset
    from utils.DataModule import DataModule
    datadmoule: DataModule = hydra.utils.instantiate(config.data)
    checkpoint = ModelCheckpoint(monitor='val_mde', mode='min', save_top_k=3, save_last=True,
                                 filename='{epoch}-{val_loss:.6f}-{val_mde:.4f}')
    
    cwd = HydraConfig.get().runtime.output_dir if config.ckpt_path is None else \
        str(Path(config.ckpt_path).parents[1].absolute())
    device = config.model.device
    trainer = hydra.utils.instantiate(config.trainer, 
                                    #   logger=TensorBoardLogger(cwd, '', '.'),
                                      accelerator=device, devices=1,#strategy="dp",
                                      callbacks=[checkpoint],
                                      reload_dataloaders_every_n_epochs=1
                                    )
    trainer.fit(model, datamodule=datadmoule, ckpt_path=config.ckpt_path)
    # train_ds = EchoDataset_Meta_heatmap(root=config['dataset_dir'], 
    #                                     split=config['data_type'],
    #                                     shot=config['shot']*2,
    #                                     transforms=train_ts, 
    #                                     num_channels=3,
    #                                     task_list = config['task'])
    # config['way'] = len(config['task'])
    # trace_func(f"Task: {config['task']}, Target: {config['target']}")


    ## Get model
    # model = get_model(config)
    # run_training(model, train_ds, optimizer=optimizer, criterion=criterion, config=config, scheduler='None')

def test(config):
    print(config)


if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--task', required=True, action='append', nargs='+', help="Choice 'PLAX', 'PSAX', '4CH', '2CH'")
#     parser.add_argument('--target', required=True, choices=['PLAX', 'PSAX', '4CH', '2CH'], help="One of views except for meta train task")
#     parser.add_argument('--shot', required=True, type=int, default=5, help="shot")
#     parser.add_argument('--num_tasks', type=int, default=50, help='the number of meta training tasks (epoch)')
#     parser.add_argument('--adaptation_steps', type=int, default=25)
#     parser.add_argument('--fast_lr', type=float, default=0.05)
#     parser.add_argument('--way', type=int, default=3)

#     parser.add_argument('--dataset_dir', default='meta_dataset', required=True, help='directory to dataset')
#     parser.add_argument('--data_type', default='train_100', choices=['train_100', 'train', 'train+val'], help='data type')
#     parser.add_argument('--width', default=256, type=int, help='width for model input')
#     parser.add_argument('--height', default=256, type=int, help='height for model input')
#     parser.add_argument('--std', default=7, type=int, help='std for heatmap')
#     parser.add_argument('--lr', default=5e-3, type=float)
#     parser.add_argument('--device', default=0)
#     parser.add_argument('--epochs', default=50, type=int)
#     parser.add_argument('--patience', default=10, type=int)
    
    
#     parser.add_argument('--network', default='DeepLabV3Plus')
#     parser.add_argument('--backbone', default='resnet50')
#     parser.add_argument('--version', required=True)
#     parser.add_argument('--use_wandb', default=False)
#     args = parser.parse_args()
#     # print(args['task'])
    
    
#     args.save_dir = os.path.join('saved_model', f"Meta_{args.target}_shot_{args.shot}_version_{args.version}")

#     if os.path.exists(args.save_dir):
#         print("Already exists version")
#         exit()
#     os.makedirs(args.save_dir, exist_ok=True)
    
#     args = vars(args)
#     args['task'] = set(['PLAX', 'PSAX', '2CH', '4CH']) - set([args['target']])
    main()
    