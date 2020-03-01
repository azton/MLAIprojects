from mxnet import np
from mxnet.gluon import nn
import mxnet.initializer as init

class BasicConv(nn.Block):
    ''' 
        basic convolution with activation and batchnorm built into it
    '''
    def __init__(self,c2, kernel_size=3, strides=1, padding=1, activation='relu',\
                    bias=False):
        super(BasicConv, self).__init__()
        self.c1 = nn.Conv3D(c2, kernel_size=kernel_size, \
                        strides=strides, padding=padding, \
                        activation='relu', weight_initializer=init.Xavier(), 
                        use_bias=bias)    
        self.bn = nn.BatchNorm()

    def forward(self, x):
        x = self.c1(x)
        x = self.bn(x)
        return x

class Reduction(nn.Block):
    def __init__(self, c1, c2):
        super(Reduction, self).__init__()
        '''
            3 convolutions, which go from channels=c1->c2
        '''
        self.conv1 = BasicConv(c1)
        self.conv2 = BasicConv(c2)
        self.conv3 = BasicConv(c2)
        self.pool = nn.MaxPool3D(pool_size=2, strides=1)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        y = self.pool(x)
        return y, x

class Expansion(nn.Block):
    def __init__(self, c1, c2):
        super(Expansion, self).__init__()
        '''
            Upconvolution, concatenation, then 3 convolutions 
                channels = c1 ->c2
        '''

        self.upconv = nn.Conv3DTranspose(c1, kernel_size=2, strides=1, padding=0,\
                        use_bias=False, weight_initializer=init.Bilinear())
        self.upconv.collect_params().setattr('grad_req','null')
        # self.upBN = nn.BatchNorm()

        self.conv1 = BasicConv(c1)
        self.conv2 = BasicConv(c2)
        self.conv3 = BasicConv(c2)

    def forward(self, x, y):
        x = self.upconv(x)
        # x = self.upBN(x)
        x = np.concatenate([x,y], axis=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class BottleNeck(nn.Block):
    def __init__(self, c1, c2):
        super(BottleNeck,self).__init__()

        self.conv1 = BasicConv(c1)
        self.conv2 = BasicConv(c2)
        self.conv3 = BasicConv(c2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class Unet3d(nn.Block):
    def __init__(self, cin, cout):
        super(Unet3d, self).__init__()

        # self.preprocess = nn.Sequential()
        # self.preprocess.add(BasicConv(16, kernel_size=7, strides=2, padding=3))
        # self.preprocess.add(nn.MaxPool3d(pool_size=2, stride=2))
        # self.preprocess.add(nn.BasicConv(32))

        self.reduce1 = Reduction(16,32)
        self.reduce2 = Reduction(32,48)
        self.reduce3 = Reduction(48, 64)
        # self.reduce4 = Reduction(64, 128)

        self.neck = BottleNeck(64, 128)

        # self.expand4 = Expansion(128, 64)
        self.expand3 = Expansion(64, 48)
        self.expand2 = Expansion(48, 32)
        self.expand1 = Expansion(32,16)

        self.finalout = BasicConv(1, kernel_size=1, strides=1, padding=0, 
                    activation = 'sigmoid')
    
    def forward(self, x):
       # x= self.preprocess(x)

        x, cat1 = self.reduce1(x)
        x, cat2 = self.reduce2(x)
        x, cat3 = self.reduce3(x)
        # x, cat4 = self.reduce4(x)

        x = self.neck(x)

        # x = self.expand4(x, cat4)
        x = self.expand3(x, cat3)
        x = self.expand2(x, cat2)
        x = self.expand1(x, cat1)
        x = self.finalout(x)
        return x