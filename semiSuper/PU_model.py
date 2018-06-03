import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, cuda
import utils, Config

class MyClassifier(Chain):

    def __call__(self, x, t, loss_func):
        self.clear()
        h = self.calculate(x)
        self.loss = loss_func(h, t)
        chainer.reporter.report({'loss': self.loss}, self)
        return self.loss

    def clear(self):
        self.loss = None

    def calculate(self, x):
        return None

    def error(self, x, t):
        xp = cuda.get_array_module(x, False)
        h = utils.get_output_by_activation(self, x)
        size = len(t)
        if isinstance(h, chainer.Variable):
            h = h.data
        if isinstance(t, chainer.Variable):
            t = t.data
        result = (h != t).sum() / size
        precision, recall, _ = utils.get_model_stats(h, t)
        chainer.reporter.report({'error': result,'precision':precision,'recall':recall}, self)
        return cuda.to_cpu(result) if xp != np else result


class MultiLayerPerceptron(MyClassifier, Chain):
    def __init__(self, prior, dim, channels, loss_func_name):
        super(MultiLayerPerceptron, self).__init__(
            # input size of each layer will be inferred when set `None`
            l1=L.Linear(None, 300),  # n_in -> n_units
            l2=L.Linear(None, 300),  # n_units -> n_units
            l3=L.Linear(None, 1),  # n_units -> n_out
        )
        self.prior = prior
        self.threshold = 0.5
        self.auc = None
        self.loss_func_name = loss_func_name

    def calculate(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h = self.l3(h2)
        return h


class CNN(MyClassifier, Chain):
    def __init__(self, prior, dim, channels, loss_func_name):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(3, 96, 3, pad=1),
            conv2=L.Convolution2D(96, 96, 3, pad=1),
            conv3=L.Convolution2D(96, 96, 3, pad=1, stride=2),
            conv4=L.Convolution2D(96, 192, 3, pad=1),
            conv5=L.Convolution2D(192, 192, 3, pad=1),
            conv6=L.Convolution2D(192, 192, 3, pad=1, stride=2),
            conv7=L.Convolution2D(192, 192, 3, pad=1),
            conv8=L.Convolution2D(192, 192, 1),
            conv9=L.Convolution2D(192, 10, 1),
            b1=L.BatchNormalization(96),
            b2=L.BatchNormalization(96),
            b3=L.BatchNormalization(96),
            b4=L.BatchNormalization(192),
            b5=L.BatchNormalization(192),
            b6=L.BatchNormalization(192),
            b7=L.BatchNormalization(192),
            b8=L.BatchNormalization(192),
            b9=L.BatchNormalization(10),
            fc1=L.Linear(None, 1000),
            fc2=L.Linear(1000, 1000),
            fc3=L.Linear(1000, 1),
        )
        self.af = F.relu
        self.prior = prior
        self.threshold = 0.5
        self.loss_func_name = loss_func_name
        self.auc = None

    def calculate(self, x):

        h = self.conv1(x)
        h = self.b1(h)
        h = self.af(h)
        h = self.conv2(h)
        h = self.b2(h)
        h = self.af(h)
        h = self.conv3(h)
        h = self.b3(h)
        h = self.af(h)
        h = self.conv4(h)
        h = self.b4(h)
        h = self.af(h)
        h = self.conv5(h)
        h = self.b5(h)
        h = self.af(h)
        h = self.conv6(h)
        h = self.b6(h)
        h = self.af(h)
        h = self.conv7(h)
        h = self.b7(h)
        h = self.af(h)
        h = self.conv8(h)
        h = self.b8(h)
        h = self.af(h)
        h = self.conv9(h)
        h = self.b9(h)
        h = self.af(h)
        h = self.fc1(h)
        h = self.af(h)
        h = self.fc2(h)
        h = self.af(h)
        h = self.fc3(h)
        return h

class BassNet(MyClassifier, Chain):
    def __init__(self, prior, dim, channels, loss_func_name):
        self.block1_nfilters = channels
        self.block1_patch_size = 1
        self.af_block1 = F.relu

        self.image_length = 3
        self.image_width = 3
        self.block2_nfilters_1 = 20
        self.block2_patch_size_1 = 3
        self.af_block2_1 = F.relu

        self.block2_nfilters_2 = 20
        self.block2_patch_size_2 = 3
        self.af_block2_2 = F.relu

        self.block2_nfilters_3 = 10
        self.block2_patch_size_3 = 3
        self.af_block2_3 = F.relu

        self.block2_nfilters_4 = 5
        self.block2_patch_size_4 = 5
        self.af_block2_4 = F.relu

        self.af_block3_1 = F.relu
        self.af_block3_2 = F.dropout

        self.nbands = Config.nbands
        self.prior = prior
        self.input_channels = channels
        self.band_size = [(a + 1) * (channels // self.nbands) for a in range(0, self.nbands - 1)]
        # self.band_size = self.block1_nfilters/self.nbands
        self.threshold = 0.5
        self.auc = None
        self.loss_func_name = loss_func_name

        super(BassNet, self).__init__(
            l1 = L.Convolution2D(channels, self.block1_nfilters, self.block1_patch_size),
            l2 = L.Convolution2D(self.image_length,self.block2_nfilters_1,(self.image_width,self.block2_patch_size_1)),
            l3 = L.Convolution2D(self.block2_nfilters_1,self.block2_nfilters_2,(1,self.block2_patch_size_2)),
            l4 = L.Convolution2D(self.block2_nfilters_2,self.block2_nfilters_3,(1,self.block2_patch_size_3)),
            l5 = L.Convolution2D(self.block2_nfilters_3,self.block2_nfilters_4,(1,self.block2_patch_size_4)),
            l6=L.Linear(None, 100),
            l7=L.Linear(100, 1)
        )


    def calculate(self, x):
        # print(x.dtype)
        h = self.l1(x)
        h = self.af_block1(h)
        split_h = F.split_axis(h, self.band_size, axis=1, force_tuple=True)
        h = ()
        for h1 in split_h:
            h1 = F.swapaxes(h1, axis1=1, axis2=3)
            h1 = F.flip(h1, 3)
            h1 = self.l2(h1)
            h1 = self.af_block2_1(h1)
            h1 = self.l3(h1)
            h1 = self.af_block2_2(h1)
            h1 = self.l4(h1)
            h1 = self.af_block2_3(h1)
            h1 = self.l5(h1)
            h1 = self.af_block2_4(h1)
            h1 = F.reshape(h1, (h1.shape[0], h1.shape[1] * h1.shape[3], h1.shape[2]))
            h += (h1,)
        h = F.dstack(h)
        h = F.reshape(h, (h.shape[0], h.shape[1] * h.shape[2]))
        h = self.l6(h)
        h = self.af_block3_1(h)
        h = self.af_block3_2(h)
        h = self.l7(h)
        return h