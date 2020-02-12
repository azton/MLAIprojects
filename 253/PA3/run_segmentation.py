from torchvision import utils
from basic_fcn import *
from dataloader import *
from utils import *
from verify_network import *
from LiftedUnet import *
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

model = 'unet'
reload = False
transform = ['crop']
batch_size=4
train_dataset = CityScapesDataset(csv_file='train.csv', transforms = transform)
test_dataset = CityScapesDataset(csv_file='test.csv', transforms = transform)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          num_workers=0,
                          shuffle=True)
val_dataset = CityScapesDataset(csv_file='val.csv', transforms=transform)
val_loader = DataLoader(dataset=val_dataset,
                        batch_size=1,
                        num_workers=0,
                        shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          num_workers=0,
                          shuffle=True)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.xavier_uniform(m.bias.data)

epochs     = 100
criterion = jaccard_loss # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
if model=='fcn':
    seg_model = FCN(n_class=n_class)
if model == 'unet':
    seg_model = UNet(3,n_class)
#fcn_model.apply(init_weights)
if reload:
    print('Reloading prior best model for %s'%model)
    seg_model.load_state_dict(torch.load('%s_best_model'%model))
lr = 5e-1
optimizer = optim.Adam(seg_model.parameters(), lr=lr)


use_gpu = torch.cuda.is_available()
print('Using cuda =', use_gpu)
if use_gpu:
    seg_model = seg_model.cuda()

best_loss = 1e10
last_loss = 0.0
for epoch in range(epochs):
    epoch_loss = 0
    ts = time.time()
    for iter, (X, tar, Y) in enumerate(train_loader):
        optimizer.zero_grad()

        if use_gpu:
            inputs = X.cuda()
            labels = Y.cuda()
        else:
            inputs, labels = X, Y
        outputs = seg_model(inputs)
        loss = criterion(outputs, labels)
        record_loss = loss.item()
        epoch_loss += record_loss
        loss.backward()
        optimizer.step()

        if iter % 10 == 0:
            print("epoch{}, iter{}, loss: {}".format(epoch, iter, epoch_loss/(iter+1)))
        if (record_loss < best_loss):
            print("Updating best model")
            torch.save(seg_model.state_dict(), '%s_best_model'%model)
            best_loss = record_loss
    if epoch_loss-last_loss < 1e-3:
        lr = 0.1*lr
        print("Reducing learning rate for further training")
        optimizer = optim.Adam(seg_model.parameters(), lr =lr)
    last_loss = epoch_loss
    print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
    torch.save(seg_model.state_dict(), "%s_EndEpoch_model"%model)
    verify(seg_model, val_loader, epoch)
    seg_model.cuda()