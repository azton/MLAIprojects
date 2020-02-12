import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import torchvision, os, h5py, copy
from my_utils import *


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, filter_size, stride=1, \
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
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        # x = self.relu(x)
        return x

class ReductionBlock(nn.Module):
    def __init__(self, in_channels, l1_channels, out_channels, filter_size=3, padding=0):
        super(ReductionBlock, self).__init__()
        self.conv1 = BasicConv(in_channels, l1_channels, filter_size)
        self.bn1 = nn.BatchNorm2d(l1_channels)
        self.conv2 = BasicConv(l1_channels, out_channels, filter_size)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        y = x
        return x, y

class ExpansionBlock(nn.Module):
    def __init__(self, in_channels, l1_channels, out_channels, filter_size=3):
        super(ExpansionBlock, self).__init__()
        self.conv1 = BasicConv(in_channels, l1_channels, filter_size)
        self.bn1 = nn.BatchNorm2d(l1_channels)
        self.conv2 = BasicConv(l1_channels, out_channels, filter_size)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, xcat):
        x = torch.cat((x,xcat), 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        '''
            define the reductions
        '''

        self.Reduce1 = ReductionBlock(in_channels, 32, 64)
        self.bnr1 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.Reduce2 = ReductionBlock(64, 64, 128)
        self.bnr2 = nn.BatchNorm2d(128)
        self.Reduce3 = ReductionBlock(128, 128, 256)
        self.bnr3 = nn.BatchNorm2d(256)
        self.Reduce4 = ReductionBlock(256, 256, 512)
        self.bnr4 = nn.BatchNorm2d(512)

        '''
            Expansion Blocks
        '''
        self.max_bypass = nn.MaxPool2d(kernel_size=9, stride=1, padding=0)
        self.max_half = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Exp1 = ExpansionBlock(512, 256, 256)
        self.bne1 = nn.BatchNorm2d(256)
        self.Exp2 = ExpansionBlock(256, 128, 128)
        self.bne2 = nn.BatchNorm2d(128)
        self.Exp3 = ExpansionBlock(128, 64, 64)
        self.bne3 = nn.BatchNorm2d(64)
        ## final 1x1x1 convolution to get right number of outputs
        self.FinalConv = BasicConv(64, out_channels, filter_size=3)

        self.upConv1 = UpConv(512, 256, filter_size=2, stride=2)
        self.upConv2 = UpConv(256, 128, filter_size=2, stride=2)
        self.upConv3 = UpConv(128, 64, filter_size=2, stride=2)

        #self.fcOutput = nn.Linear(5832, 1000)
        # self.output = nn.Softmax()



    def crop (self, x, bypass):
        c = (bypass.size()[2] - x.size()[2])//2
        y = F.pad(bypass, (-c, -c, -c, -c))
        return y
    def forward(self, x):
        '''
            contract
        '''
        x, catx1 = self.Reduce1(x)
        x = self.bnr1(x)
        x = self.max_pool(x)
        x , catx2 = self.Reduce2(x)
        x = self.bnr2(x)
        x = self.max_pool(x)
        x, catx3 = self.Reduce3(x)
        x = self.bnr3(x)
        # print(x.size())
        x = self.max_pool(x)
        x, _ = self.Reduce4(x)
        x = self.bnr4(x)
        '''
            expand
        '''
        x = self.upConv1(x)
        x = self.Exp1(x,catx3)
        x = self.bne1(x)
        x = self.upConv2(x)
        catx2 = self.crop(x, catx2)
        # print(x.size(), catx2.size())
        x = self.Exp2(x, catx2)
        x = self.bne2(x)
        x = self.upConv3(x)
        catx1= self.crop(x, catx1)
        # print(x.size(), catx1.size())
        x = self.Exp3(x, catx1)
        x = self.bne3(x)
        x = self.FinalConv(x)
        '''
            classifier out of it
        '''

        return x
