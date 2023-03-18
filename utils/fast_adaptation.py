from utils.heatmaps import *
from utils.utils import *
from utils.evaluation import AverageMeter

def fast_adapt(batch, learner, loss, CFG, vis=True):
    target_weights = torch.tensor([[1,1,1,1,0.01]]).to(CFG['device'])
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
        # import cv2
        # i = cv2.addWeighted(adaptation_data[0,0,...].numpy(),1,adaptation_labels_heatmap[0].squeeze().sum(0).numpy(),.5,0)

        # plt.figure(figsize=(15,10))
        # plt.axis('off')
        # plt.imshow(i, cmap='gray')
        # plt.show()
        pred = learner(adaptation_data)
        adaptation_error = loss(pred, adaptation_labels_heatmap, target_weight=target_weights)
        # print(pred[0], adaptation_labels_heatmap[0])
        print(f"{step}: {adaptation_error}")
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
    if vis:
        fig, ax = plt.subplots(1, 5, figsize=(15,10))

        for i, prediction in enumerate(evaluation_labels_heatmap[0]):
            ax[i].imshow(prediction.detach().cpu().numpy())
        plt.show() 

        fig, ax = plt.subplots(1, 5, figsize=(15,10))
        for i, prediction in enumerate(predictions[0]):
            ax[i].imshow(prediction.detach().cpu().numpy())
        plt.show() 
    evaluation_error = loss(predictions, evaluation_labels_heatmap, target_weight=target_weights)
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