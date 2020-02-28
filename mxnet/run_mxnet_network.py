import mxnet as mx
import glob
from progressbar import *
from mxnet import np, npx
from mxnet import gluon, init, autograd, gpu
from mxnet.gluon.data import DataLoader
from helpers.dataloader import *
from helpers.utils import *
from mxnet_networks.u_net_3d import Unet3d


npx.set_np()
device = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
print ('Training on', device)
batch_size=4.0
dim = [64,64,64]
data_filepath = '/data/FormationTrainingSet'
filenames = glob.glob('%s/*.h5' % data_filepath)
for f in glob.glob('%s/extra/*.h5' % data_filepath):
    filenames.append(f)
filenames = filenames[:-1]
field_list = ('density', 'total_energy', 'SN_Colour', 'velocity_divergence')
train_dataset = CosmologySingleDataset(filenames, 2, field_list, \
                                star_blur=1, transforms=['crop'],\
                                crop_dim=dim, \
                                class_type='segmentation',\
                                scale_file = 'Scaling.torch')
val_loader = DataLoader(train_dataset, batch_size=batch_size, \
                                shuffle=True, num_workers=6)

net = Unet3d(len(field_list), 2)
net.initialize(ctx = device)
# net.collect_params().initialize(force_reinit=True, ctx = device)
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False) # sparse label is for 1-hot output
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001})

recorder = Recorder('unet_3d', './')
for epoch in range(10):
    widgets = ['[%d]'%epoch, Bar(), Counter(), '/%d'%(len(filenames)//batch_size)]
    bar = ProgressBar(widgets=widgets, maxval=len(filenames)//batch_size)
    bar.start()
    losses, pix_acc, iou_acc = 0, 0, 0
    acc = mx.metric.Accuracy()
    for i, (x, y) in enumerate(val_loader):
        x=x.copyto(device)
        y=y.copyto(device)
        with autograd.record():
            output = net(x)
            loss = loss_fn(output, y)
        loss.backward()
        trainer.step(batch_size=batch_size)  # include batch size in step
        losses += loss.mean()
        intersect = 0
        union = 0
        # acc.update(pred=output, labels=y)
        # pix_acc += acc.get()
        # union = float((output > 0.5).sum() + y.sum()) - acc.get()*y.sum()
        # iou_acc += acc.get()*y.sum()/(union+1e-7)
        bar.update(i)
    iou_acc /= float(i + 1) * batch_size
    pix_acc /= float(i + 1) * batch_size
    losses /= float(i + 1) * batch_size
    recorder.record(['train_loss','train_accuracy'], [losses, iou_acc], epoch)
    bar.finish()
    print('[%d] loss = %f\tPA = %e\tIOU = %e\t' %
          (epoch, losses, pix_acc, iou_acc))
recorder.plot_and_save_record(epoch, val=False)
