from mxnet import np, npx
from mxnet.gluon import nn

npx.set_np()

class Inception(nn.Block):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super (Inception, self).__init__(**kwargs)

        self.path1 = nn.Conv2d(c1, kernel_size=1, activation='relu')

        self.p2_1 = nn.Conv2d(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2d(c2[1], kernel_size=3, padding=1, activation='relu')

        self.p3_1 = nn.Conv2d(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2d(c3[1], kernel_size=5, padding=2, activation='relu')

        self.p4_1 = nn.MaxPool2d(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2d(c4, kernel_size=1, activation='relu')

    def forward(self,x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_1(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))

        return np.concatenate((p1,p2,p3,p4), axis=1)

class GoogleNet()