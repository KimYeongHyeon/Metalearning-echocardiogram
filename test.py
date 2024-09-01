import os

import glob
import numpy as np
import matplotlib.pyplot as plt
import random
os.environ['WANDB_MODE']='offline'
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from albumentations.pytorch import ToTensorV2

import albumentations as A

from colorama import Fore, Back, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

import time
from collections import defaultdict
import copy
import gc

from utils.loss import HeatmapMSELoss
from utils.meta_dataset import EchoDataset_Meta_heatmap
from utils.train import *
from utils.evaluation import *
from utils.optimizer import *
from utils.utils import *
from utils.heatmaps import *
from utils.train_metalearning import *

import wandb
import warnings
import learn2learn as l2l
from learn2learn.optim.transforms.metacurvature_transform import MetaCurvatureTransform

warnings.filterwarnings("ignore")

import argparse

### Logging part
import logging
from datetime import datetime
from pytz import timezone
import sys

import hydra
from hydra.core.hydra_config import HydraConfig

import random


from tqdm import tqdm

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

def log(CFG):
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
    file_handler = logging.FileHandler(os.path.join(CFG['save_dir'], f"{CFG['target']}.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # logger.addHandler(TqdmLoggingHandler())

    logger.info(f"python {' '.join(sys.argv)}")
    logger.info("Timezone: " + str(tz))
    logger.info(f"Finetuning Start")
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


from albumentations.core.transforms_interface import ImageOnlyTransform
class MinMaxNormalize(ImageOnlyTransform):
    """
    Min-max normalization
    """

    def apply(self, img, **param):
        # minmax normalize
        # img = (img - img.min()) / (img.max() - img.min())
        img = img / 255.
        return img

def get_model(CFG):
    import segmentation_models_pytorch as smp
    import learn2learn as l2l
    from utils.model import get_basemodel
    model = get_basemodel(CFG['network'], CFG['backbone'], CFG['channels'])
    # model = smp.DeepLabV3Plus(
    #                 encoder_name = CFG['backbone'],
    #                 in_channels = CFG['channels'],
    #                 classes = 5,
    #                 activation = 'softmax')

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

    return model

def main(CFG):
    if torch.cuda.is_available():
        try:
            CFG['device'] = torch.device(f'cuda:{CFG["device"]}')
        except Exception as e:
            print(e)
    else:
        CFG['device'] = torch.device('cpu')

    # logger = log(CFG)
    # CFG['trace_func'] = logger.info
    CFG['trace_func'] = print
    # trace_func = print#CFG['trace_func']

    train_ts = A.Compose([
        A.Resize(CFG['height'], CFG['width']),
        A.SafeRotate(limit=30),
        MinMaxNormalize(p=1),
        ToTensorV2(),
    ],
        keypoint_params=A.KeypointParams(format='xy')
    )
    test_ts = A.Compose([
        A.Resize(CFG['height'], CFG['width']),
        MinMaxNormalize(p=1),
        ToTensorV2(),
    ],
        keypoint_params=A.KeypointParams(format='xy')
    )


    ## Load Dataset
    from utils.meta_dataset import EchoDataset_Meta_heatmap

    test_ds = EchoDataset_Meta_heatmap(root_dir=CFG['dataset_dir'], 
                                  split=str(CFG['shot']),
                                  shot=CFG['shot'],
                                  transforms=test_ts, 
                                  num_channels=CFG['channels'],
                                  task_list = [CFG['target']])

    test_dl = DataLoader(test_ds, batch_size=CFG['shot'], shuffle=False)
    CFG['trace_func'](f"Task: {CFG['task']}, Target: {CFG['target']}")

    from utils.dataset_new import EchoDataset_heatmap
    unseen_test_ds = EchoDataset_heatmap(root=os.path.join(CFG['dataset_dir'], CFG['target']), 
                                         transforms=test_ts, 
                                         split='test', 
                                         num_channels=CFG['channels'])

    unseen_test_dl = DataLoader(unseen_test_ds, shuffle=False, batch_size=1)


    ## Get model
    # model: BaseSystem = hydra.utils.instantiate(config.model)
    model = get_model(CFG)
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG['lr'])
    criterion = HeatmapMSELoss(use_target_weight=True)
    
    model_dir = f"outputs/{CFG['algorithm']}_FO_{CFG['first_order']}_AN_{CFG['allow_nograd']}/{CFG['target']}/{CFG['shot']}/lightning_logs/version_{CFG['version']}/checkpoints"

    model_state_dict = torch.load(os.path.join(model_dir, 'last.ckpt'))['state_dict']
    model_state_dict = {k.replace('model.module.', 'module.'): v for k, v in model_state_dict.items()}
    model_state_dict = {k.replace('model.lrs.', 'lrs.'): v for k, v in model_state_dict.items()}
    model_state_dict = {k.replace('model.compute_update.', 'compute_update.'): v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    model.to(CFG['device'])
    
    evaluation_error, mean_distance_error = adapt_on_new_dataset(test_dl.__iter__().__next__(), model, criterion, optimizer, CFG)
    from tqdm import tqdm
    mean_distance_error = AverageMeter()
    pred_list = []
    all_distance_error_list = []
    model.eval()
    
    with torch.no_grad():
        for sample in tqdm(unseen_test_dl):
            sample['data'] = sample['data'].to(CFG['device'])
            sample['label'] = sample['label'].to(CFG['device'])
            x=sample['data']
            y=sample['label']
            
            heatmap = render_gaussian_dot_f(
                y.flip(dims=[2]), # xy 2 yx
                torch.tensor([CFG['std'], CFG['std']], dtype=torch.float32).to(CFG['device']),
                [CFG['height'], CFG['width']],
                # mul=255.
            ).to(torch.float)
            background = 1 - heatmap.sum(dim=1).unsqueeze(1).clip(0,1)
            heatmap = torch.concat((heatmap,background), 1)
            preds = model(x)
            
            metric = distance_error(sample, heatmap2coor(preds[:,:-1,...]))
            mean_distance_error.update(np.mean(
                                                metric
                                                ), 
                                    x.size(0)
                                    )
            pred_list.extend(preds)#.detach().cpu().numpy())
            distance_error_list = distance_error(sample, heatmap2coor(preds[:,:-1,...]))
            all_distance_error_list.append(distance_error_list)
        
    CFG['trace_func'](f"Mean Disatnce Error (point): {np.mean(all_distance_error_list, axis=0)}")
    CFG['trace_func'](f"std (point): {np.std(all_distance_error_list, axis=0)}")

    CFG['trace_func'](f"Mean Disatnce Error (all): {np.mean(all_distance_error_list)}")
    CFG['trace_func'](f"std (all): {np.std(all_distance_error_list)}")
    
    CFG['trace_func'](f"{np.mean(all_distance_error_list, axis=0)} {np.mean(all_distance_error_list)} {np.std(all_distance_error_list, axis=0)} {np.std(all_distance_error_list)}")

    if CFG.get('output_dir') is not None:
        os.makedirs(os.path.join(CFG['output_dir'], CFG['algorithm']), exist_ok=True)
        pd.DataFrame(all_distance_error_list).to_excel(os.path.join(CFG['output_dir'], CFG['algorithm'], f"target_{CFG['target']}-shot_{CFG['shot']}-{np.mean(all_distance_error_list):.4f}.xlsx"))
    else:
        pd.DataFrame(all_distance_error_list).to_excel(os.path.join(model_dir, f"target_{CFG['target']}-shot_{CFG['shot']}-{np.mean(all_distance_error_list):.4f}.xlsx"))
    model_name = os.path.join(model_dir, f"finetuned_model-{np.mean(all_distance_error_list):.4f}.pth")
    torch.save({
                'state_dict': model.state_dict(),
                'optimizer_states':optimizer.state_dict(),
                # 'loss': evaluation_error, 
                # 'metric': mean_distance_error

                }, 
                model_name)
                # os.path.join(CFG['save_dir'], model_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--task', required=True, action='append', nargs='+', help="Choice 'PLAX', 'PSAX', '4CH', '2CH'")
    parser.add_argument('--target', required=True, choices=['PLAX', 'PSAX', '4CH', '2CH'], help="One of views except for meta train task")
    parser.add_argument('--shot', required=True, type=int, default=5, help="shot")
    parser.add_argument('--num_tasks', type=int, default=50, help='the number of meta training tasks (epoch)')
    parser.add_argument('--adaptation_steps', type=int, default=100)
    parser.add_argument('--fast_lr', type=float, default=0.05)
    parser.add_argument('--way', type=int, default=3)
    parser.add_argument('--allow_nograd', type=bool, default=True)
    parser.add_argument('--first_order', type=bool, default=True)
    parser.add_argument('--algorithm', type=str, default='MAML', choices=['MAML', 'MetaSGD', 'MetaCurvature'])
    
    parser.add_argument('--dataset_dir', default='dataset',  help='directory to dataset')
    parser.add_argument('--width', default=256, type=int, help='width for model input')
    parser.add_argument('--height', default=256, type=int, help='height for model input')
    parser.add_argument('--channels', default=3, type=int)
    parser.add_argument('--std', default=7, type=int, help='std for heatmap')
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--device', default=0)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--patience', default=10, type=int)
    
    # parser.add_argument('--type', required=True, choices=['loss','metric','last'], help='model type')
    
    parser.add_argument('--network', default='DeepLabV3Plus')
    parser.add_argument('--backbone', default='tu-efficientnet_b0')
    parser.add_argument('--version', required=True)
    parser.add_argument('--use_wandb', default=False)
    parser.add_argument('--output_dir', default='result_excel')
    args = parser.parse_args()
    
    args = vars(args)

    args['save_dir'] = os.path.join('saved_model', f"Meta_{args['target']}_shot_{args['shot']}_version_{args['version']}")

    # if os.path.exists(os.path.join('saved_model', args.save_dir)):
    #     print("Already exists version")
    #     exit()
    # if not os.path.exists(args['save_dir']):
    #     print("Wrong! Check target, shot, version!")
    #     exit()
    
    # if args['type'] in ['loss', 'metric']:
    #     args['model_path'] = os.path.join(args['save_dir'], f"best_{args['type']}_epoch-{args['target']}.pth")
    #     if not os.path.exists(args['model_path']):
    #         print("Wrong model load! Check target, type!")
    #         exit()
        
            
    # if args['type'] in ['last']:
    #     args['model_path'] = os.path.join(args['save_dir'], f"last_epoch-{args['target']}.pth")
    #     if not os.path.exists(args['model_path']):
    #         print("Wrong model load! Check target, type!")
    #         exit()
    
    # print(args['model_path'])
    args['task'] = set(['PLAX', 'PSAX', '2CH', '4CH']) - set([args['target']])
    args['img_size'] = [args['height'], args['width']]
    main(args)
    