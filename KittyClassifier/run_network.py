import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import torchvision, os, h5py, copy


traindata = torchvision.datasets.Cityscapes('/mnt/d/MLAIprojects/data/CityScapes', \
            split='train', mode='coarse', target_type='semantic')

testdata = torchvision.datasets.Cityscapes('/mnt/d/MLAIprojects/data/CityScapes', \
            split='val', mode='coarse', target_type='semantic')




