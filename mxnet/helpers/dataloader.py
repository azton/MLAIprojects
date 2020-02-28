from mxnet.gluon.data import Dataset# For custom data-sets
from mxnet import np
import torch.nn.functional as F
import numpy
import torch, h5py, glob



class CosmologySingleDataset(Dataset):

    def __init__(self, filenames, n_class=2, \
            field_list = ('density'), class_type = 'segmentation',\
            transforms=None, crop_dim = None, star_blur = 1, scale_file='Scaling.torch'):
        self.data      = filenames
        self.transforms = transforms
        self.fields  = field_list
        self.crop_dim  = crop_dim
        self.classifier = class_type # segmentation or classifier
        ## star blur is the range (in grid cells) to allow star formation;
        ##       just one cell in a 128^3 may be too rare of an event, this way we
        ##       can use star_blur^3 cells to increase the region, but decrease the
        ##       precision
        self.star_blur = star_blur
        self.scaling = torch.load(scale_file)

    def __len__(self):
        return len(self.data)


    def transform(self, img, label, dx, le):
        for t in self.transforms:
            if t == 'crop':
                csize = self.crop_dim
                s = img.size()
                xstart = numpy.random.randint(0, s[-3]-csize[0])
                ystart = numpy.random.randint(0, s[-2]-csize[1])
                zstart = numpy.random.randint(0, s[-1]-csize[2])

                img = img[:, xstart:xstart+csize[0], \
                            ystart:ystart+csize[1], \
                            zstart:zstart+csize[2]]
                label = label[xstart:xstart+csize[0], \
                            ystart:ystart+csize[1], \
                            zstart:zstart+csize[2]]
                le = [xstart*dx+le[0], le[1]+ystart*dx, le[2]+zstart*dx ]
        return img, label, le


    def __getitem__(self, idx):
        file = h5py.File(self.data[idx], 'r')
        vol = torch.zeros((len(self.fields), 128,128,128))
        ## assign requested fields to data category
        for i, field in enumerate(self.fields):
            if 'velocity' not in field:
                shp = file[field][:].shape
                vol[i] = F.pad(torch.from_numpy(file[field][:]), \
                            (0, 128-shp[2], 0, 128-shp[1], 0, 128-shp[0]))
                vol[i] = (vol[i]-self.scaling[field]['mean']) \
                                /self.scaling[field]['std']
            if "velocity_divergence" in field:
                div = torch.zeros_like(vol[0])
                for ii,ax in enumerate('xyz'):
                    fld = 'velocity_%s'%ax
                    shp = file[fld].shape
                    vel_field = F.pad(torch.from_numpy(file[fld][:]), \
                            (0, 128-shp[2], 0, 128-shp[1], 0, 128-shp[0]))
                                    
                    div += numpy.gradient(\
                                    (vel_field-self.scaling[fld]['mean']\
                                        /self.scaling[fld]['std']))[ii]
                vol[i] = div

        le = file.attrs['cover_grid_left_edge']
        dx = file.attrs['dx']
        star_inds = []
        for c in file['new_stars_position'][:]:
            rel_pos = c - le
            star_inds.append((rel_pos)//dx+1)
        star_inds = numpy.array(star_inds)
        ### return matrix of pixel labels
        label = torch.zeros((128,128,128))

        offset = range(-self.star_blur, (self.star_blur)+1)

        ## set all cells hosting stars (and those within radius star_blur) to 1
        ## for segmentations
        for s in star_inds:
            for dx in offset:
                for dy in offset:
                    for dz in offset:
                        ind = (s + [dx, dy, dz]).astype(int)
                        if not numpy.any(ind >= 128):
                            label[ind[0],ind[1],ind[2]] = 1
        if self.transforms != None:
            vol, label, le = self.transform(vol, label, dx, le)

        ## if cropping, have to check the new volumee for stars
        if self.classifier == 'classification':
            label = (label.sum() >= 1).long()
        ## apply transforms to data
        file.close()
        return np.array(vol.numpy()), np.array(label.long().numpy())



class CosmologyMultiloadDataset(Dataset):
    def __init__():
        pass
    def __len__():
        pass
    def __getitem__():
        pass