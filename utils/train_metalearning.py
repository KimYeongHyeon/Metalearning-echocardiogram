from torch.utils.data import DataLoader
import torch.nn as nn
import os
from .heatmaps import *
from .utils import *
from .evaluation import AverageMeter
from collections import defaultdict
import gc
import tqdm
def fast_adapt(batch, learner, loss, CFG):
    data, labels = batch['data'], batch['label']
    data, labels = data.to(CFG['device']), labels.to(CFG['device'])
    losses = AverageMeter()
    mean_distance_error = AverageMeter()    

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(CFG['shot']*CFG['way'])*2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
    
    adaptation_labels_heatmap = render_gaussian_dot_f(
                adaptation_labels.flip(dims=[2]), # xy 2 yx
                torch.tensor([CFG['std'], CFG['std']], dtype=torch.float32).to(CFG['device']),
                [CFG['height'], CFG['width']],
            ).to(torch.float)

    background = 1 - adaptation_labels_heatmap.sum(dim=1).unsqueeze(1).clip(0,1)
    adaptation_labels_heatmap = torch.concat((adaptation_labels_heatmap,background), 1)
    for step in range(CFG['adaptation_steps']):
        pred = learner(adaptation_data)
        pred = F.interpolate(pred, size = (256, 256), mode = 'bilinear', align_corners = False)

        adaptation_error = loss(pred, adaptation_labels_heatmap, target_weight=torch.tensor([[1,1,1,1,0.01]]).to(CFG['device']))
        # print(pred[0], adaptation_labels_heatmap[0])
        # print(f"{step}: {adaptation_error}")
        learner.adapt(adaptation_error)
        # losses.update(adapatation_error.item(), evaluation_data.size(0))

    evaluation_labels_heatmap = render_gaussian_dot_f(
                evaluation_labels.flip(dims=[2]), # xy 2 yx
                torch.tensor([CFG['std'], CFG['std']], dtype=torch.float32).to(CFG['device']),
                [CFG['height'], CFG['width']],
            ).to(torch.float)
    background = 1 - evaluation_labels_heatmap.sum(dim=1).unsqueeze(1).clip(0,1)
    # print(evaluation_labels.shape,evaluation_labels_heatmap.shape)
    evaluation_labels_heatmap = torch.concat((evaluation_labels_heatmap,background), 1)
    
    # Adapt the model
    predictions = learner(evaluation_data)
    # if segformer
    # predictions = F.interpolate(predictions, size = (256, 256), mode = 'bilinear', align_corners = False)

    evaluation_error = loss(predictions, evaluation_labels_heatmap, target_weight=torch.tensor([[1,1,1,1,0.01]]).to(CFG['device']))
    losses.update(evaluation_error.item(), evaluation_data.size(0))

    sample = {
        'data': evaluation_data, 
        'label': evaluation_labels
    }
    metric = distance_error(sample, heatmap2coor(predictions[:,:4,...]))
    mean_distance_error.update(np.mean(
                            metric
                            ), 
                    evaluation_data.size(0)
    )

    return evaluation_error, mean_distance_error.avg

import time

def run_training(model, train_ds, optimizer, scheduler, criterion, CFG, wandb=None):
    """
    모델, 데이터를 받아 학습/검증 수행후 결과에 대해 출력 및 모델 저장
    """
    trace_func = CFG['trace_func']

    start = time.time()
    early_stopping = EarlyStopping(patience=CFG['patience'], verbose=True, trace_func=trace_func)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mde = np.inf
    best_metric_epoch = -1

    best_loss = np.inf
    best_loss_epoch = -1

    history = defaultdict(list)
    # 에폭만큼 학습 수행
    
    # train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
    # model_ = model
    
    pbar = tqdm.tqdm(range(1, CFG['epochs']+1))
    for epoch in pbar:
        optimizer.zero_grad()
        train_ds.resample()
        
        model_ = model.clone()
        
        train_dl = DataLoader(train_ds, batch_size=CFG['shot']*CFG['way']*2, shuffle=False)
        batch = train_dl.__iter__().__next__()

        
        evaluation_error, mean_distance_error = fast_adapt(batch, learner=model_, loss=criterion, CFG=CFG)
        evaluation_error.backward()
#         early_stopping(evaluation_error, model)
        # for p in model.parameters():
        #     p.grad.data.mul_(1.0 / num_epochs)
        optimizer.step()
        val_mde = mean_distance_error
        val_loss = evaluation_error
        
        pbar.set_description(f"Epoch {epoch}/{CFG['epochs']}")
        trace_func(f"Epoch {epoch}/{CFG['epochs']}   Val MDE: {val_mde:.6f}, Val Loss: {val_loss.detach().cpu().numpy():.6f}")

        # Val MDE가 개선된 경우
        if val_mde <= best_mde:
            trace_func(f"Valid Score Improved ({best_mde:.6f} ---> {val_mde:.6f})")
            best_mde = val_mde
            best_metric_epoch = epoch
            
            if CFG['use_wandb']:
                wandb.run.summary['Best MDE'] = best_mde
                wandb.run.summary['Best Metric Epoch'] = best_metric_epoch

        #   model_name = f"best_metric_epoch-{CFG['view']}-{CFG['dataset']}-{CFG['backbone']}.pth"
            model_name = f"best_metric_epoch-{CFG['target']}.pth"

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss': val_loss,
                'metric': val_mde
                }, 
                os.path.join(CFG['save_dir'], model_name))

        # Loss 개선되었을때
        if val_loss <= best_loss:
            trace_func(f"Valid Loss Improved ({best_loss:.6f} ---> {val_loss:.6f})")
            best_loss = val_loss
            best_loss_epoch = epoch
            
            if CFG['use_wandb']:
                wandb.run.summary['Best MDE'] = best_mde
                wandb.run.summary['Best Loss Epoch'] = best_loss_epoch

            model_name = f"best_loss_epoch-{CFG['target']}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss': val_loss,
                'metric': val_mde
                }, 
                os.path.join(CFG['save_dir'], model_name))

        del train_dl
        gc.collect()
        
#         if early_stopping.early_stop:
#             trace_func("Early stopping")
#             break

    model_name = f"last_epoch-{CFG['target']}.pth"
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss': evaluation_error, 
                'metric': mean_distance_error

                }, 
                os.path.join(CFG['save_dir'], model_name))
    
    end = time.time()
    time_elapsed = end-start
    trace_func("Trianing complete in {:.0f}h {:.0f}m {:.0f}s".format(time_elapsed//3600, (time_elapsed%3600)//60, (time_elapsed%3600)%60))
    trace_func("Best Val Loss: {:.8f}".format(best_loss))
    trace_func("Best Val MDE: {:.4f}".format(best_mde))

    # model.load_state_dict(best_metric_model_wts)

    return model, history


from utils.heatmaps import *
from utils.utils import *

def adapt_on_new_dataset(batch, learner, loss, optimizer, CFG):
    losses = AverageMeter()
    mean_distance_error = AverageMeter()    

    data, labels = batch['data'], batch['label']
    adaptation_data, adaptation_labels = data.to(CFG['device']), labels.to(CFG['device'])

    adaptation_labels_heatmap = render_gaussian_dot_f(
                adaptation_labels.flip(dims=[2]), # xy 2 yx
                torch.tensor([CFG['std'], CFG['std']], dtype=torch.float32).to(CFG['device']),
                [CFG['height'], CFG['width']],
            ).to(torch.float)
    background = 1 - adaptation_labels_heatmap.sum(dim=1).unsqueeze(1).clip(0,1)
    adaptation_labels_heatmap = torch.concat((adaptation_labels_heatmap,background), 1)

    sample = {
        'data': adaptation_data, 
        'label': adaptation_labels
    }

    for step in range(CFG['adaptation_steps']):
        optimizer.zero_grad()

        predictions = learner(adaptation_data)
        adaptation_error = loss(predictions, adaptation_labels_heatmap, target_weight=torch.tensor([[1,1,1,1,0.01]]).to(CFG['device']))
        # print(pred[0], adaptation_labels_heatmap[0])
        CFG['trace_func'](f"step {step}: Loss {adaptation_error:.8f}")
        # learner.adapt(adaptation_error)

        adaptation_error.backward()
        optimizer.step()
        metric = distance_error(sample, heatmap2coor(predictions[:,:-1,...]))

        losses.update(adaptation_error.item(), adaptation_data.size(0))
        mean_distance_error.update(np.mean(
                                metric
                                ), 
                        adaptation_data.size(0)
        )
        CFG['trace_func'](f"step {step}: MDE {mean_distance_error.avg:.4f}")
    # fig, ax = plt.subplots(1, 5, figsize=(15,10))
    # for i, prediction in enumerate(predictions[0]):
    #     ax[i].imshow(prediction.detach().cpu().numpy())
    # plt.show() 
    # fig, ax = plt.subplots(1, 5, figsize=(15,10))
    # for i, prediction in enumerate(adaptation_labels_heatmap[0]):
    #     ax[i].imshow(prediction.detach().cpu().numpy())
    # plt.show() 
    

    return losses, mean_distance_error.avg