import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, cuda
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


class MyClassifier(Chain):
    prior = 0

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
        size = len(t)
        with chainer.no_backprop_mode():
            with chainer.using_config("train", False):
                if self.loss_func_name == "sigmoid_cross_entropy":
                    h = xp.reshape(F.sigmoid(self.calculate(x)).data, size)
                else:
                    h = xp.reshape(xp.sign(self.calculate(x).data), size)


        if isinstance(h, chainer.Variable):
            h = h.data
        if self.loss_func_name == "sigmoid_cross_entropy":
            h[xp.where(h >= 0.5)] = 1  # For binary
            h[xp.where(h < 0.5)] = 0  # For binary
        # print(np.unique(h))
        if isinstance(t, chainer.Variable):
            t = t.data
        result = (h != t).sum() / size
        precision, recall, _, _ = precision_recall_fscore_support(t, h, pos_label=1, average='binary')
        # print("precision, recall", precision, recall)
        chainer.reporter.report({'error': result,'precision':precision,'recall':recall}, self)
        return cuda.to_cpu(result) if xp != np else result


class LinearClassifier(MyClassifier, Chain):
    def __init__(self, prior, dim, channels, loss_func_name):
        super(LinearClassifier, self).__init__(
            l = L.Linear(dim, 2),
            l2 = L.Linear(2,1)
        )
        self.prior = prior
        self.loss_func_name = loss_func_name

    def calculate(self, x):
        h = self.l(x)
        h = self.l2(h)
        print("model params", self.l.W,self.l.b)
        return h

class ThreeLayerPerceptron(MyClassifier, Chain):
    def __init__(self, prior, dim, channels, loss_func_name):
        super(ThreeLayerPerceptron, self).__init__(l1=L.Linear(dim, 100),
                                                   l2=L.Linear(100, 1))
        self.af = F.relu
        self.prior = prior
        self.loss_func_name = loss_func_name

    def calculate(self, x):
        h = self.l1(x)
        h = self.af(h)
        h = self.l2(h)
        return h

class MultiLayerPerceptron(MyClassifier, Chain):
    def __init__(self, prior, dim, channels, loss_func_name):
        super(MultiLayerPerceptron, self).__init__(
            # input size of each layer will be inferred when set `None`
            l1=L.Linear(None, 300),  # n_in -> n_units
            l2=L.Linear(None, 300),  # n_units -> n_units
            l3=L.Linear(None, 1),  # n_units -> n_out
        )
        self.prior = prior
        self.loss_func_name = loss_func_name


    # def __init__(self, prior, dim):
    #     super(MultiLayerPerceptron, self).__init__(l1=L.Linear(dim, 300, nobias=True),
    #                                                b1=L.BatchNormalization(300),
    #                                                l2=L.Linear(300, 300, nobias=True),
    #                                                b2=L.BatchNormalization(300),
    #                                                l3=L.Linear(300, 300, nobias=True),
    #                                                b3=L.BatchNormalization(300),
    #                                                l4=L.Linear(300, 300, nobias=True),
    #                                                b4=L.BatchNormalization(300),
    #                                                l5=L.Linear(300, 1))
    #     self.af = F.relu
    #     self.prior = prior

    def calculate(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h = self.l3(h2)
        # h = self.l1(x)
        # h = self.b1(h)
        # h = self.af(h)
        # h = self.l2(h)
        # h = self.b2(h)
        # h = self.af(h)
        # h = self.l3(h)
        # h = self.b3(h)
        # h = self.af(h)
        # h = self.l4(h)
        # h = self.b4(h)
        # h = self.af(h)
        # h = self.l5(h)
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
        self.loss_func_name = loss_func_name

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

        self.nbands = 10
        self.prior = prior
        self.input_channels = channels
        self.band_size = self.block1_nfilters/self.nbands
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
        h1, h2, h3, h4, h5, h6, h7, h8, h9, h10 = F.split_axis(h,self.nbands,axis=1)
        h1 = F.swapaxes(h1, axis1=1, axis2=3)
        h1 = F.flip(h1, 3)
        h2 = F.swapaxes(h2, axis1=1, axis2=3)
        h2 = F.flip(h2, 3)
        h3 = F.swapaxes(h3, axis1=1, axis2=3)
        h3 = F.flip(h3, 3)
        h4 = F.swapaxes(h4, axis1=1, axis2=3)
        h4 = F.flip(h4, 3)
        h5 = F.swapaxes(h5, axis1=1, axis2=3)
        h5 = F.flip(h5, 3)
        h6 = F.swapaxes(h6, axis1=1, axis2=3)
        h6 = F.flip(h6, 3)
        h7 = F.swapaxes(h7, axis1=1, axis2=3)
        h7 = F.flip(h7, 3)
        h8 = F.swapaxes(h8, axis1=1, axis2=3)
        h8 = F.flip(h8, 3)
        h9 = F.swapaxes(h9, axis1=1, axis2=3)
        h9 = F.flip(h9, 3)
        h10 = F.swapaxes(h10, axis1=1, axis2=3)
        h10 = F.flip(h10, 3)
        h1 = self.l2(h1)
        h2 = self.l2(h2)
        h3 = self.l2(h3)
        h4 = self.l2(h4)
        h5 = self.l2(h5)
        h6 = self.l2(h6)
        h7 = self.l2(h7)
        h8 = self.l2(h8)
        h9 = self.l2(h9)
        h10 = self.l2(h10)
        h1 = self.af_block2_1(h1)
        h2 = self.af_block2_1(h2)
        h3 = self.af_block2_1(h3)
        h4 = self.af_block2_1(h4)
        h5 = self.af_block2_1(h5)
        h6 = self.af_block2_1(h6)
        h7 = self.af_block2_1(h7)
        h8 = self.af_block2_1(h8)
        h9 = self.af_block2_1(h9)
        h10 = self.af_block2_1(h10)
        h1 = self.l3(h1)
        h2 = self.l3(h2)
        h3 = self.l3(h3)
        h4 = self.l3(h4)
        h5 = self.l3(h5)
        h6 = self.l3(h6)
        h7 = self.l3(h7)
        h8 = self.l3(h8)
        h9 = self.l3(h9)
        h10 = self.l3(h10)
        h1 = self.af_block2_2(h1)
        h2 = self.af_block2_2(h2)
        h3 = self.af_block2_2(h3)
        h4 = self.af_block2_2(h4)
        h5 = self.af_block2_2(h5)
        h6 = self.af_block2_2(h6)
        h7 = self.af_block2_2(h7)
        h8 = self.af_block2_2(h8)
        h9 = self.af_block2_2(h9)
        h10 = self.af_block2_2(h10)
        h1 = self.l4(h1)
        h2 = self.l4(h2)
        h3 = self.l4(h3)
        h4 = self.l4(h4)
        h5 = self.l4(h5)
        h6 = self.l4(h6)
        h7 = self.l4(h7)
        h8 = self.l4(h8)
        h9 = self.l4(h9)
        h10 = self.l4(h10)
        h1 = self.af_block2_3(h1)
        h2 = self.af_block2_3(h2)
        h3 = self.af_block2_3(h3)
        h4 = self.af_block2_3(h4)
        h5 = self.af_block2_3(h5)
        h6 = self.af_block2_3(h6)
        h7 = self.af_block2_3(h7)
        h8 = self.af_block2_3(h8)
        h9 = self.af_block2_3(h9)
        h10 = self.af_block2_3(h10)
        h1 = self.l5(h1)
        h2 = self.l5(h2)
        h3 = self.l5(h3)
        h4 = self.l5(h4)
        h5 = self.l5(h5)
        h6 = self.l5(h6)
        h7 = self.l5(h7)
        h8 = self.l5(h8)
        h9 = self.l5(h9)
        h10 = self.l5(h10)
        h1 = self.af_block2_4(h1)
        h2 = self.af_block2_4(h2)
        h3 = self.af_block2_4(h3)
        h4 = self.af_block2_4(h4)
        h5 = self.af_block2_4(h5)
        h6 = self.af_block2_4(h6)
        h7 = self.af_block2_4(h7)
        h8 = self.af_block2_4(h8)
        h9 = self.af_block2_4(h9)
        h10 = self.af_block2_4(h10)
        h1 = F.reshape(h1, (h1.shape[0], h1.shape[1] * h1.shape[3], h1.shape[2]))
        h2 = F.reshape(h2, (h2.shape[0], h2.shape[1] * h2.shape[3], h2.shape[2]))
        h3 = F.reshape(h3, (h3.shape[0], h3.shape[1] * h3.shape[3], h3.shape[2]))
        h4 = F.reshape(h4, (h4.shape[0], h4.shape[1] * h4.shape[3], h4.shape[2]))
        h5 = F.reshape(h5, (h5.shape[0], h5.shape[1] * h5.shape[3], h5.shape[2]))
        h6 = F.reshape(h6, (h6.shape[0], h6.shape[1] * h6.shape[3], h6.shape[2]))
        h7 = F.reshape(h7, (h7.shape[0], h7.shape[1] * h7.shape[3], h7.shape[2]))
        h8 = F.reshape(h8, (h8.shape[0], h8.shape[1] * h8.shape[3], h8.shape[2]))
        h9 = F.reshape(h9, (h9.shape[0], h9.shape[1] * h9.shape[3], h9.shape[2]))
        h10 = F.reshape(h10, (h10.shape[0], h10.shape[1] * h10.shape[3], h10.shape[2]))
        h = F.dstack((h1, h2, h3, h4, h5, h6, h7, h8, h9, h10))
        h = F.reshape(h, (h.shape[0], h.shape[1] * h.shape[2]))
        h = self.l6(h)
        h = self.af_block3_1(h)
        h = self.af_block3_2(h)
        h = self.l7(h)
        return h