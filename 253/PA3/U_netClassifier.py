import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import torchvision, os, h5py, copy
from my_utils import *


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, stride=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, filter_size, stride=stride, \
                        padding=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=3, stride=2):
        super(UpConv, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, padding=0,\
                kernel_size=filter_size, stride=stride, bias=True, output_padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class ReductionBlock(nn.Module):
    def __init__(self, in_channels, l1_channels, out_channels, filter_size=3, padding=0):
        super(ReductionBlock, self).__init__()
        self.conv1 = BasicConv(in_channels, l1_channels, filter_size)
        self.bn1 = nn.BatchNorm2d(l1_channels)
        self.conv2 = BasicConv(l1_channels, out_channels, filter_size)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        y = x
        x = self.maxpool(x)
        return x, y

class ExpansionBlock(nn.Module):
    def __init__(self, in_channels, l1_channels, out_channels, filter_size=3):
        super(ExpansionBlock, self).__init__()
        self.conv1 = BasicConv(in_channels, l1_channels, filter_size)
        self.bn1 = nn.BatchNorm2d(l1_channels)
        self.conv2 = BasicConv(l1_channels, out_channels, filter_size)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.upconv = UpConv(l1_channels, out_channels, filter_size=2, stride=2)
        self.bnup = nn.BatchNorm2d(out_channels)

    def forward(self, x, xcat):
        x = torch.cat((x,xcat), 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.upconv(x)
        x = self.bnup(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        '''
            define the reductions
        '''

        self.Reduce1 = ReductionBlock(in_channels, 32, 64)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.Reduce2 = ReductionBlock(64, 64, 128)
        self.Reduce3 = ReductionBlock(128, 128, 256)
        self.Reduce4 = ReductionBlock(256, 256, 512)

        '''
            Expansion Blocks
        '''
        self.max_bypass = nn.MaxPool2d(kernel_size=9, stride=1, padding=0)
        self.max_half = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Exp1 = ExpansionBlock(1024, 256, 256)
        self.Exp2 = ExpansionBlock(512, 128, 128)
        self.Exp3 = ExpansionBlock(256, 64, 64)
        # self.Exp4 = ExpansionBlock(128, 32, 32)
        ## final 1x1x1 convolution to get right number of outputs
        self.FinalConv = BasicConv(64, out_channels, filter_size=3, stride=1)

        #self.fcOutput = nn.Linear(5832, 1000)
        # self.output = nn.Softmax()



    def crop (self, x, bypass):
        dy = (bypass.size()[2] - x.size()[2])//2
        dx = (bypass.size()[3] - x.size()[3])//2
        y = F.pad(x, (dx, dx, dy, dy))
        return y
    def forward(self, x):
        '''
            contract
        '''
        x, catx1 = self.Reduce1(x)
        x , catx2 = self.Reduce2(x)
        x, catx3 = self.Reduce3(x)
        x, catx4 = self.Reduce4(x)
        '''
            expand
        '''
        x = self.crop(x, catx4)
        x = self.Exp1(x,catx4)
        x = self.crop(x, catx3)
        x = self.Exp2(x, catx3)
        x= self.crop(x, catx2)
        x = self.Exp3(x, catx2)
        # x = self.Exp4(x, catx1)
        x = self.FinalConv(x)
        '''
            classifier out of it
        '''

        return x
