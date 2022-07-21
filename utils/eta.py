from argparse import ArgumentError
import numpy as np


class ETA(object):
    """An estimator of ETA."""
    def __init__(self, num_step_per_epoch, num_epochs, max_hist=10):
        super().__init__()
        self.num_step_per_epoch = num_step_per_epoch
        self.num_epochs = num_epochs
        self.max_hist = max_hist
        self.toc_step_hist = []
        self.toc_epoch_hist = []
    
    def __call__(self, step=None, epoch=None, toc_step=None, toc_epoch=None):
        if toc_epoch is not None and toc_step is None:
            self.toc_epoch_hist.append(toc_epoch)
            if len(self.toc_epoch_hist) > self.max_hist:
                self.toc_epoch_hist.pop(0)
            toc_epoch = np.mean(self.toc_epoch_hist)

            eta = toc_epoch * (self.num_epochs - epoch)
            return eta
        elif toc_epoch is None and toc_step is not None:
            self.toc_step_hist.append(toc_step)
            if len(self.toc_step_hist) > self.max_hist:
                self.toc_step_hist.pop(0)
            toc_step = np.mean(self.toc_step_hist)

            eta = toc_step * (self.num_step_per_epoch * (self.num_epochs - (epoch-1)) - step)
            return eta
        elif toc_epoch is None and toc_step is None:
            raise ArgumentError('Cannot estimate ETA when both toc_epoch and toc_step are None.')
        else:
            raise ArgumentError('Cannot estimate ETA when both toc_epoch and toc_step are not None.')


        