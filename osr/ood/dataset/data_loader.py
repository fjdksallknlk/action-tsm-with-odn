# original code is from https://github.com/aaron-xichen/pytorch-playground
# modified by Kimin Lee
# modified by Shu Yu
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from dataset import dataset_config

import sys
sys.path.append('../../')
from ops.transforms import *
from ops.dataset import TSNDataSet
# from ops import dataset_config



def getDataSet(data_type, dataset, modality, batch_size, num_segments, transform, workers, dense_sample, twice_sample):
    num_class, train_list, val_list, root_path, prefix = dataset_config.return_dataset(dataset, modality)

    if modality == 'RGB':
        data_length = 1
    elif modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(root_path, train_list, num_segments=num_segments,
                   new_length=data_length,
                   modality=modality,
                   image_tmpl=prefix,
                   transform=transform,
                   dense_sample=dense_sample,
                   twice_sample=twice_sample),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    test_loader = torch.utils.data.DataLoader(
        TSNDataSet(root_path, val_list, num_segments=num_segments,
                   new_length=data_length,
                   modality=modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=transform, dense_sample=dense_sample, twice_sample=twice_sample),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    
    if data_type == 'inData':
        return train_loader, test_loader
    return test_loader
