from automl_workflow.api import DataLoader

from Auto_Image.skeleton.data import FixedSizeDataLoader

import torch


class MyDataLoader(object):
    """Default PyTorch dataloader for train, validation, test, etc.
    
    Note that this definition of data loader is different to that of PyTorch.
    This defines `f` instead of `f(x)` if `f` is the transformation and `x` is
    the dataset.
    """
    
    def __call__(self, dataset, train=True):
        # pt_dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
        #                             shuffle=train, num_workers=2)
        # return pt_dataloader
        return FixedSizeDataLoader(dataset)