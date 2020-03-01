import matplotlib, torch, os, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from mxnet import np, npx

class Recorder():
    def __init__(self, model, out_dir, segmentation=True):
        self.model = model
        self.filepath = '%s/%s'%(out_dir, model)
        self.records = defaultdict(list)
        self.segment = segmentation
        for rec in ['train_loss','train_accuracy', \
                    'val_accuracy', 'val_loss', 'epoch']:
            self.records[rec] = []
        ## per-epoch values
        self.epoch_measures = {}
        self.epoch_measures['correct'] = 0
        self.epoch_measures['total'] = 0
        self.epoch_measures['pixel_acc_mean'] = 0
        self.epoch_measures['iou'] = 0
        self.epoch_measures['n_pos'] = 0
        self.epoch_measures['n_true_pos'] = 0
        self.epoch_measures['n_false_pos'] = 0
        self.epoch_measures['n_false_neg'] = 0
        self.epoch_measures['loss'] = 0

        # global training values
        self.global_records = {}
        self.global_records['Best_pixel_acc'] = 0
        self.global_records['Best_loss'] = 10000

    def clear_epoch(self):
        for k in self.epoch_measures:
            self.epoch_measures[k] = 0

    def add_record(self, rec_name):
        self.records[rec_name] = []

    def record(self, key, value, epoch):
        '''
            keys are the losses/accuracies to update.
            values are listed corresponding values
        '''
        for i,k in enumerate(key):
            self.records[k].append(value[i])
        if epoch not in self.records['epoch']:
            self.records['epoch'].append(epoch)

    def check_dir(self):
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)

    def record_reload(self):
        reload = torch.load('%s/records.torch'%self.filepath)
        self.global_records = reload['global_records']
        self.records = reload['records']

    def save_record(self, val):
        self.check_dir()
        save = {'records':self.records,\
                    'global_records':self.global_records}
        if not val:
            torch.save(save, '%s/records_train.torch'%self.filepath)
        if val:
            torch.save(save, '%s/records_val.torch'%self.filepath)
    def plot_and_save_record(self, epoch, val = True):
        self.check_dir()
        fig, ax = plt.subplots(1,2, figsize=(8,4))
        ax[0].plot(self.records['epoch'], self.records['train_loss'], label='Train')
        if val:
            ax[0].plot(self.records['epoch'], self.records['val_loss'], label='Val')
        ax[1].plot(self.records['epoch'], self.records['train_accuracy'], label='Train')
        if val:
            ax[1].plot(self.records['epoch'], self.records['val_accuracy'], label='Val')
        ax[0].set_title('Loss')
        if self.segment:
            ax[1].set_title('IoU')
        else:
            ax[1].set_title('Accuracy')
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        ax[0].legend()
        if not val:
            plt.savefig('%s/record_progress.png'%(self.filepath))
        else:
            plt.savefig('%s/val_train_record_progress.png'%(self.filepath))
            
        ## save the record that made the plot
        self.save_record(val)


class Plotter():
    def __init__(self, prefix, model):
        self.prefix = prefix
        self.model = model
        self.filepath = '%s/%s'%(self.prefix, self.model)

    def create_folders(self, direct):
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
        if not os.path.exists('%s/%s'%(self.filepath, direct)):
            os.makedirs('%s/%s'%(self.filepath, direct))


    def flatten(self,classes, y, img):
        classes = torch.sum(classes, 3)
        y = torch.sum(y, 3)
        img = torch.sum(img, 3)
        return classes, y, img

    def plot_image(self,predict, y, img, pixel_accuracy, iter, epoch, direct):
        print("Saving Image")
        ## check and create directories if needed
        self.create_folders(direct)

        _, classes = torch.max(predict.cpu(), 1)
        classes = classes.to('cpu')
        y = y.to('cpu')
        if len(classes.size()) == 4:  #(B, w, h, d)
            ## sum over z-axis to plot on x-y plane
            classes, y, img = self.flatten(classes, y, img)

        ## make multiframe image
        for b in range(predict.size()[0]):
            fig, ax = plt.subplots(1,3, figsize=(12,4))
            ax[2].imshow(classes[b])
            ax[2].set_title('positive: %d'%classes[b].sum())
            ax[1].imshow(y[b])
            ax[1].set_title('True pos: %d'%y[b].sum())
            ax[0].imshow(img[b, 0])
            ax[0].set_title('Projected density(column)')
            plt.savefig('%s/%s/input_%03d_%d_%0.6f.png' %
                        (self.filepath, direct, epoch, b, pixel_accuracy))
            plt.close()
        del (classes)
        del (y)


def save_state(filepath, model, epoch, optimizer, loss=None):
    torch.save({'epoch': epoch,\
            'model_state': model.state_dict(),\
            'optimizer_state': optimizer.state_dict(),\
            'loss': loss},\
            filepath)



def pixel_accuracy(output, y):
    '''
        binary class prediction accuracy. 
        output is dim(B, 1, W, H, {D})
        target is dim(B,  W, H, {D})
    '''
    true_pos = np.sum(y)
    if output.shape[1] == 1:
        classes = (output > 0.5).astype('float32')
    if output.shape[1] == 2:
        classes = np.argmax(output, axis=1)
    acc = (classes.astype('bool') * y.astype('bool')).sum()
    # print('Acc:',acc)
    if true_pos > 0:
        pix_acc = acc/y.sum().astype('float32')
    if true_pos == 0 and acc == 0:
        pix_acc = 1.0
    if true_pos == 0 and acc != 0:
        pix_acc = 0.0

    return pix_acc


def IoU(pred, target):
    eps = 1e-7
    '''
        iou or jaccard loss for binary classes
    '''
    intersect = pixel_accuracy(pred, target)*target.sum()
    if pred.shape[1] == 1:
        classes = (pred > 0.5).astype('int32')
    else:
        classes = np.argmax(pred, axis=1)
    union = pred.sum()+target.sum().astype('int32')-intersect.astype('int32')


    if union > 0:
        sum_ious = intersect/(union+eps)
    elif union <= 0:
        sum_ious = 0

    '''
        Autograd wants to minimize a loss, so negate the IOU
    '''
    return 1.- sum_ious








"""Common image segmentation losses.
"""

import torch

from torch.nn import functional as F


def bce_loss(true, logits, pos_weight=None):
    """Computes the weighted binary cross-entropy loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, 1, H, W]. Corresponds to
            the raw output or logits of the model.
        pos_weight: a scalar representing the weight attributed
            to the positive class. This is especially useful for
            an imbalanced dataset.
    Returns:
        bce_loss: the weighted binary cross-entropy loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(
        logits.float(),
        true.float(),
        pos_weight=pos_weight,
    )
    return bce_loss


def ce_loss(true, logits, weights, ignore=255):
    """Computes the weighted multi-class cross-entropy loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        weight: a tensor of shape [C,]. The weights attributed
            to each class.
        ignore: the class index to ignore.
    Returns:
        ce_loss: the weighted multi-class cross-entropy loss.
    """
    ce_loss = F.cross_entropy(
        logits.float(),
        true.long(),
        ignore_index=ignore,
        weight=weights,
    )
    return ce_loss


def dice_loss(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0, ) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

def DiceLoss3d(predict, true, eps=1e-7):
    n_class = predict.shape[1]
    onehot_true = torch.eye(n_class)[true.squeeze(1)]
    onehot_true = onehot_true.permute(0,4,1,2,3).float()
    probs = F.softmax(predict)
    onehot_true = onehot_true.type(predict.type())
    dims = (0,)+tuple(range(2, true.ndimension()))
    intersection = torch.sum(probs * onehot_true, dims)
    cardinality = torch.sum(probs + onehot_true , dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def jaccard_loss(logits, true, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0, ) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims[0]).sum(dims[1])
    cardinality = torch.sum(probas + true_1_hot, dims[0]).sum(dims[1])
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    if union == 0:
        jacc_loss = 0
    return (1 - jacc_loss)


def tversky_loss(logits, true, alpha, beta, eps=1e-7):
    """Computes the Tversky loss [1].
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.
    Returns:
        tversky_loss: the Tversky loss.
    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0, ) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()
    return (1 - tversky_loss)


from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return 1.0 - s / (i + 1)