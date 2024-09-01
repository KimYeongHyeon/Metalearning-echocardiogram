import torch

from sklearn.metrics import mean_squared_error

from tqdm import tqdm

from .evaluation import AverageMeter# ,distance_error, 
from .utils import *
from .heatmaps import *

def train_one_epoch(model, dataloader, optimizer, scheduler, device, criterion, CFG):
    model.train()

    losses = AverageMeter()
    mean_distance_error = AverageMeter()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, sample in pbar:
        optimizer.zero_grad()
        sample['data'] = sample['data'].to(device)
        sample['label'] = sample['label'].to(device)
        
        x = sample['data']
        y = sample['label']
        heatmap = render_gaussian_dot_f(
            y.flip(dims=[2]), # xy 2 yx
            torch.tensor([CFG['std'], CFG['std']], dtype=torch.float32).to(CFG['device']),
            [CFG['height'], CFG['width']],
            # mul=255.
        ).to(torch.float)
        background = 1 - heatmap.sum(dim=1).unsqueeze(1).clip(0,1)
        heatmap = torch.concat((heatmap,background), 1)

        preds = model(x)
        
        loss = criterion(preds*255., heatmap*255.)  
        
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), x.size(0))
        
        metric = distance_error(sample, heatmap2argmax(preds[:,:-1,...]))
        mean_distance_error.update(np.mean(
                                            metric
                                            ), 
                                   x.size(0)
                                   )
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{losses.avg:.4f}',
                         train_MDE=f'{mean_distance_error.avg:.4f}',
                         lr=f'{current_lr:.5f}',
                         )
    return losses.avg, mean_distance_error.avg



@torch.no_grad()
def valid_one_epoch(model, dataloader, device, criterion, CFG):
    model.eval()
    
    losses = AverageMeter()
    mean_distance_error = AverageMeter()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')    

    for step, sample in pbar:
        sample['data'] = sample['data'].to(device)
        sample['label'] = sample['label'].to(device)
        
        x = sample['data']
        y = sample['label']
        
        heatmap = render_gaussian_dot_f(
            y.flip(dims=[2]), # xy 2 yx
            torch.tensor([CFG['std'], CFG['std']], dtype=torch.float32).to(CFG['device']),
            [CFG['height'], CFG['width']],
            # mul=255.
        ).to(torch.float)
        background = 1 - heatmap.sum(dim=1).unsqueeze(1).clip(0,1)
        heatmap = torch.concat((heatmap,background), 1)
        preds = model(x)
        
        loss = criterion(preds*255., heatmap*255.)
        
        losses.update(loss.item(), x.size(0))
        metric = distance_error(sample, heatmap2argmax(preds[:,:-1,...]))
        mean_distance_error.update(np.mean(
                                            metric
                                            ), 
                                   x.size(0)
                                   )
        

        pbar.set_postfix(valid_loss=f'{losses.avg:.4f}',
                         valid_MDE=f'{mean_distance_error.avg:.4f}'
                        )

    return losses.avg, mean_distance_error.avg
