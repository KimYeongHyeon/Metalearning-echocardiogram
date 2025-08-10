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


@hydra.main(config_path='configs', config_name='main', version_base='1.1')
def main(config: DictConfig):
    # 태스크 정의
    with open_dict(config):
        config.data.tasks = list(set(['PLAX', 'PSAX', '2CH', '4CH']) - set([config.data.target]))

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

def test(config):
    print(config)


if __name__ == '__main__':
    main()
    