import numpy as np
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
        
        
from numpy.linalg import norm
def distance_error(output, target):
    diffs = output-target    
    result = [norm(diff,2) for diffs_ in diffs for diff in diffs_]
    result = np.mean(result) 
    return result