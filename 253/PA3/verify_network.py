from torchvision import utils
from basic_fcn import *
from dataloader import *
from LiftedUnet import *
from PIL import Image
from utils import *
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time, sys
import matplotlib.pyplot as plt

def plot_image(predict, y, pixel_accuracy, iter):
    print("Saving Image")
    _, classes = torch.max(predict, 1)
    classes = classes.to('cpu')
    y = y.to('cpu')
    # tar = tar.to('cpu')
    # print(np.shape(y), np.shape(classes), tar.size())
    plt.figure()    
    im = plt.imshow(classes[0])
    plt.savefig('classes_%f_%d.png'%(pixel_accuracy/iter, epoch))
    plt.close()
    plt.figure()
    plt.imshow(y[0])
    plt.savefig('label_%f_%d.png'%(pixel_accuracy/iter, epoch))
    plt.close()
def verify(model, ver_loader, epoch):
    model.eval()
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    iou_total = 0
    loss = jaccard_loss
    pixel_accuracy = 0
    model.to(device)
    pa_best = 0
    for iter, (x, tar, y) in enumerate(ver_loader):
        x = x.to(device)
        y = y.to(device)

        predict = model(x)

        ## generating classes on gpu runs oom
        ## get array of class labels for each pixel
        # print(classes.size(), y.size())
        IoU = loss(predict, y)
        PAcc = pixel_acc(predict,y)

        iou_total+= IoU.item()
        pixel_accuracy += PAcc
        if IoU.item() > pa_best:
            plot_image(predict, y, PAcc, iter+1)
        if (iter+1)%10 == 0:
            print('Intermediate Accuracy: IoU=%f PA = %f'%(iou_total/iter, pixel_accuracy/iter))
        if iter > 100:
            break
    
    # plt.imsave('tgt%f.png'%(pixel_accuracy/iter), tar)
    iou_avg = iou_total/len(ver_loader)
    acc_avg = pixel_accuracy/len(ver_loader)

    print('Verification Accuracy IoU = %f PA = %f'%(iou_avg, acc_avg))
    model.train()
    return iou_avg, acc_avg

if __name__ == '__main__':
    model = sys.argv[1]
    epoch = int(sys.argv[2])
    batch_size=1
    if model == 'fcn':
        seg_model = FCN(n_class=n_class)
    if model == 'unet':
        seg_model = UNet(3, n_class)
    seg_model.load_state_dict(torch.load('%s_EndEpoch_model'%model))
    transform = ['downsample']
    val_dataset = CityScapesDataset(csv_file='val.csv', transforms=transform)
    val_loader = DataLoader(dataset=val_dataset,
                        batch_size=batch_size,
                        num_workers=4,
                        shuffle=True)

    verify(seg_model, val_loader, epoch)