import mxnet as mx

from mxnet import np, npx, gluon, init, autograd, gpu
from mxnet.gluon.data import DataLoader
from dataloader import *
from u_net import Unet3d

print(mx.context.num_gpus())


testarray = np.zeros((100,6,128,128,128))