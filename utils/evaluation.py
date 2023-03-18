import numpy as np       
from .heatmaps import *
from .utils import *

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def test_on_model(batch, learner, loss, optimizer, CFG, vis=True):
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

    for step in range(CFG['adaptation_steps']*2):
        optimizer.zero_grad()

        predictions = learner(adaptation_data)
        adaptation_error = loss(predictions, adaptation_labels_heatmap, target_weight=torch.tensor([[1,1,1,1,0.01]]).to(CFG['device']))
        # print(pred[0], adaptation_labels_heatmap[0])
        print(f"{step}: {adaptation_error}")
        # learner.adapt(adaptation_error)

        adaptation_error.backward()
        optimizer.step()
        metric = distance_error(sample, heatmap2coor(predictions[:,:4,...]))

        losses.update(adaptation_error.item(), adaptation_data.size(0))
        mean_distance_error.update(np.mean(
                                metric
                                ), 
                        adaptation_data.size(0)
        )
    if vis:
        fig, ax = plt.subplots(1, 5, figsize=(15,10))
        for i, prediction in enumerate(predictions[0]):
            ax[i].imshow(prediction.detach().cpu().numpy())
        plt.show() 
        fig, ax = plt.subplots(1, 5, figsize=(15,10))
        for i, prediction in enumerate(adaptation_labels_heatmap[0]):
            ax[i].imshow(prediction.detach().cpu().numpy())
        plt.show()     

    return losses, mean_distance_error.avg