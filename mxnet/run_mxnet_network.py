import mxnet as mx
import glob, d2l
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
batch_size=1.0
dim = [64, 64, 64]
data_filepath = '/data/FormationTrainingSet'
filenames = glob.glob('%s/*.h5' % data_filepath)
for f in glob.glob('%s/extra/*.h5' % data_filepath):
    filenames.append(f)
for i in sorted(range(len(filenames)), reverse=True):
    try:
        file = h5py.File(filenames[i], 'r')
        file.close()
    except:
        filenames.pop(i)
# filenames = filenames[:100]
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
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss(axis=1, from_logits=False, sparse_label=True) # sparse label is for 1-hot output
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001})

recorder = Recorder('unet_3d', './')
for epoch in range(2):
    widgets = ['[%d]'%epoch, Bar(), Counter(), '/%d'%(len(filenames)//batch_size)]
    bar = ProgressBar(widgets=widgets, maxval=len(filenames)//batch_size)
    bar.start()
    losses, pix_acc, iou_acc = 0, 0, 0
    for i, (x, y) in enumerate(val_loader):
        x=x.copyto(device)
        y=y.copyto(device).reshape(x.shape[0],1, x.shape[2], x.shape[3], x.shape[4])
        with autograd.record():
            output = net(x)
            output.wait_to_read()
            loss = loss_fn(output, y)
            loss.wait_to_read()
        loss.reshape(x.shape[0])
        loss.backward()
        trainer.step(x.shape[0])  # include batch size in step
        
        '''
            record metrics...
        '''

        losses += np.sum(loss)
        pix_acc += pixel_accuracy(output, y)
        iou_acc += IoU(output, y)
        bar.update(i)


    pix_acc /= float(i + 1) * batch_size
    losses = losses/float(i + 1) / batch_size
    iou_acc = iou_acc/float(i+1) / batch_size
    recorder.record(['train_loss', 'train_acc'], [losses, iou_acc/(i+1)/batch_size], epoch)
    bar.finish()
    print(epoch, losses)
    print('[%d] loss = %f\tPA = %e\tIOU = %e\t' %
          (epoch, losses, pix_acc, iou_acc))
recorder.plot_and_save_record(epoch, val=False)
