'''
    convenience functions to help processing NN data
'''
from torch.utils import data
import torch.nn as nn
import h5py, os, torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class DataSet(data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

class Conv():
    def __init__(self, in_channel, out_channel, kernel_size):
        super(Conv).__init__()
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size, bias=False)
    def forward(self, x):
        x = self.conv(x)
        return x

def load_h5(fp):
    '''
        load a h5 file, return samples, X, and labels, Y
        also return info, # channels, dimensions, and # samples
    '''
    info = {}
    if os.path.exists(fp):
        f = h5py.File(fp, 'r')
        info["n_channels"] = int(f['samples'].attrs['n_channels'])
        info['cube_dims'] = int(f['samples'].attrs['cube_dims'][0])
        info['n_samples'] = int(f['samples'].attrs['n_samples'])
        X = torch.Tensor(f['samples'])
        X = X.reshape(info['n_samples'], info['n_channels'],\
                 info['cube_dims'], info['cube_dims'], info['cube_dims'])
        Y = torch.Tensor(f['labels']).long()
        return X, Y, info
    else:
        print('%s does not exist!!'%fp)
        return False

'''
    take in x-y, and generate a semi-balanced sample set 
        probably just for network testing
'''
def balanced_subset(X, Y):
    npos = Y.sum()
    ind_pos = np.where(Y==1)[0]
    for n in range(len(ind_pos)):
        r_ind = np.random.randint(len(Y))
        while r_ind in ind_pos:
            r_ind = np.random.randint(len(Y))
        ind_pos = np.append(ind_pos, r_ind)
    return X[ind_pos], Y[ind_pos]