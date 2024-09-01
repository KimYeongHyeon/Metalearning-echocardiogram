
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}))#.  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def restore(sample: dict, pred, order='xyxy') -> list:
    '''restore predicted value to original coordinates per batch

    Inputs:
        sample(dict)
        preds(list)
        order(str): 'xxyy' or 'xyxy'
    Returns:
        restored_coors(list)    
    '''
    assert order in ['xxyy', 'xyxy']
    # if isinstance(pred, torch.Tensor):  pred = pred.detach().cpu().numpy()

    ori_width = sample['width']
    ori_height = sample['height']
    batch_size, _, resized_height, resized_width = sample['data'].shape
    ratio_w = ori_width / resized_width
    ratio_h = ori_height / resized_height
    ratio_w = ratio_w.reshape(-1, 1)
    ratio_h = ratio_h.reshape(-1, 1)
    new_pred = []
    pred_ = pred.reshape(batch_size, -1, 4).clone()
    
    if order == 'xyxy':
        pred_[:, :, 0] = pred_[:, :, 0] * ratio_w
        pred_[:, :, 1] = pred_[:, :, 1] * ratio_h
        pred_[:, :, 2] = pred_[:, :, 2] * ratio_w
        pred_[:, :, 3] = pred_[:, :, 3] * ratio_h
    elif order == 'xxyy':
        pred_[:, :, 0] = pred_[:, :, 0] * ratio_w
        pred_[:, :, 1] = pred_[:, :, 1] * ratio_w
        pred_[:, :, 2] = pred_[:, :, 2] * ratio_h
        pred_[:, :, 3] = pred_[:, :, 3] * ratio_h        
    # for pred in pred_:
    #     # arrange xyxy -> xxyy 
    #     if order == 'xyxy':
    #         pred = np.array(pred)[[0,2,1,3]]
        
    #     pred[:2] = pred[:2] * (ori_width/resized_width)
    #     pred[2:] = pred[2:] * (ori_height/resized_height)
    #     new_pred.extend(pred)

    return pred_.to(torch.int)

def distance_error(sample: dict, preds, order='xyxy'):
    '''Caculate distance error bewteen the answers and predicted values per batch
    
    '''
    assert order=='xyxy', NotImplemented

    batch_size, _, _ = sample['label'].shape
    error = sample['label'].detach().cpu().numpy() - preds.detach().cpu().numpy()
    
    # 좌표 2개씩 묶음
    distance_error_per_coor_type_per_sample = np.mean(error.reshape(-1, 2)**2, axis=1)
    # 같은 좌표끼리 묶음
    distance_error_per_coor_type = np.mean(distance_error_per_coor_type_per_sample.reshape(batch_size, -1), axis=0)
    
    return np.sqrt(distance_error_per_coor_type)

def heatmap2argmax(heatmap, scale=False):
    def _scale(p, s): return 2 * (p / s) - 1

    N, C, H, W = heatmap.shape
    index = heatmap.view(N,C,1,-1).argmax(dim=-1).to('cpu')
    pts = torch.cat([index%W, index//W], dim=2)
    pts.to(heatmap.device)
    if scale:
        scale = torch.tensor([W,H], device=heatmap.device)
        pts = _scale(pts, scale)

    return pts
    
def show_lvh(sample, pred = None):
    image = sample['data'].squeeze().permute(1,2,0).clone()
    
    coors = sample['label'] if pred == None else pred
    plt.figure(figsize=(15,10))
    plt.imshow(image, cmap='gray')
    for coor in coors.reshape(-1, 4):
        coor = coor.squeeze()
        plt.plot((coor[0], coor[2]),
                (coor[1], coor[3]), linewidth=3)
        plt.scatter((coor[0], coor[2]), (coor[1], coor[3]), )
    plt.title(f"id: {sample['id']} \
                shape: {sample['data'].shape}")
    plt.axis('off')
    plt.show()