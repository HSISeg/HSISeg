import numpy as np
import six
import time, utils, Config, copy
from chainer import Variable, optimizers, serializers, initializers, cuda
import scipy.io
import chainer
import chainer.functions as F
from chainer import computational_graph
import chainer.links as L
from chainer import Chain, cuda
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from semiSuper.tanh_cross_entropy import tanh_cross_entropy

class BassNet(Chain):
    def __init__(self, channels, n_classes = 1):
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
        self.input_channels = channels
        # self.band_size = self.block1_nfilters // self.nbands
        self.band_size = [(a + 1) * (channels // self.nbands) for a in range(0, self.nbands - 1)]
        self.threshold = 0.5
        self.auc = None
        super(BassNet, self).__init__(
            l1=L.Convolution2D(channels, self.block1_nfilters, self.block1_patch_size),
            l2=L.Convolution2D(self.image_length, self.block2_nfilters_1, (self.image_width, self.block2_patch_size_1)),
            l3=L.Convolution2D(self.block2_nfilters_1, self.block2_nfilters_2, (1, self.block2_patch_size_2)),
            l4=L.Convolution2D(self.block2_nfilters_2, self.block2_nfilters_3, (1, self.block2_patch_size_3)),
            l5=L.Convolution2D(self.block2_nfilters_3, self.block2_nfilters_4, (1, self.block2_patch_size_4)),
            l6=L.Linear(None, 100),
            l7=L.Linear(100, n_classes)
        )

    def __call__(self, x):
        # print(x.shape)
        h = self.calculate_common_links(x)
        h = self.calculate_remaining_links(h)
        return h

    def calculate(self, x):
        return self.__call__(x)

    def calculate_common_links(self, x):
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
        return h

    def calculate_remaining_links(self, h):
        h = self.l6(h)
        h = self.af_block3_1(h)
        h = self.af_block3_2(h)
        h = self.l7(h)
        # print(h,h.shape,'h')
        # h = F.reshape(h, (h.shape[0],))
        return h


class Configuration2(Chain):
    def __init__(self, model1):
        self.model1 = copy.deepcopy(model1)
        self.af_block3_1 = model1.af_block3_1
        self.af_block3_2 = model1.af_block3_2
        self.threshold = 0.5
        self.auc = None
        super(Configuration2, self).__init__(
            l6=L.Linear(None, 100),
            l7=L.Linear(100, 1)
        )

    def __call__(self, x):
        with chainer.configuration.using_config('enable_backprop', False):
            h_fixed = self.model1.calculate_common_links(x)
        h = Variable(np.array(h_fixed.data))
        with chainer.configuration.using_config('enable_backprop', True):
            h = self.calculate_remaining_links(h)
        return h

    def calculate(self, x):
        return self.__call__(x)

    def calculate_remaining_links(self, h):
        h = self.l6(h)
        h = self.af_block3_1(h)
        h = self.af_block3_2(h)
        h = self.l7(h)
        h = F.reshape(h, (h.shape[0],))
        return h



def shuffle_data(X_tr, Y_tr, X_te, Y_te):
    perm = np.random.permutation(Y_tr.shape[0])
    X_tr, Y_tr = X_tr[perm], Y_tr[perm]
    perm = np.random.permutation(Y_te.shape[0])
    X_te, Y_te = X_te[perm], Y_te[perm]
    return (X_tr, Y_tr), (X_te, Y_te)

class SoftmaxClassifier(chainer.Chain):
    """Classifier is for calculating loss, from predictor's output.
    predictor is a model that predicts the probability of each label.
    """
    def __init__(self, predictor):
        super(SoftmaxClassifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t):
        y = self.predictor(x)
        # print(y.shape, "y_shape",t.shape,"t_shape")
        # print(y,t)
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)
        return self.loss

class TanhClassifier(chainer.Chain):
    def __init__(self, predictor):
        super(TanhClassifier, self).__init__()
        with self.init_scope():
            print(predictor)
            self.predictor = predictor 

    def __call__(self, x, t):
        y = self.predictor(x)
        # print(y.shape, "y_shape",t.shape,"t_shape")
        # print(y,t)
        self.loss = tanh_cross_entropy(y, t)
        self.accuracy = F.binary_accuracy(y, t)
        return self.loss

class SigmoidClassifier(chainer.Chain):
    """Classifier is for calculating loss, from predictor's output.
    predictor is a model that predicts the probability of each label.
    """
    def __init__(self, predictor):
        super(SigmoidClassifier, self).__init__()
        with self.init_scope():
            # print(predictor)
            self.predictor = predictor

    def __call__(self, x, t):
        y = self.predictor(x)
        # print(y.shape, "y_shape",t.shape,"t_shape")
        # print(y,t)
        self.loss = F.sigmoid_cross_entropy(y, t)
        self.accuracy = F.binary_accuracy(y, t)
        return self.loss


def get_accuracy(model, x, t):
    size = len(t)
    x = np.asarray(x, dtype=np.float32)
    h = F.sigmoid(model.calculate(x))
    # h = np.reshape(np.sign(model.calculate(x).data), size)
    if isinstance(h, chainer.Variable):
        h = h.data
    if isinstance(t, chainer.Variable):
        t = t.data
    negative, positive = np.unique(t)
    h[np.where(h >= 0.5)] = 1
    h[np.where(h < 0.5)] = 0

    try:
        precision, recall, _, _ = precision_recall_fscore_support(t, h, pos_label=1, average='binary')
    except Exception as e:
        precision, recall = 0.0, 0.0
    tn, fp, fn, tp = confusion_matrix(t, h).ravel()
    return precision, recall, (tn, fp, fn, tp)

def train(X_tr2, Y_tr2, X_te, Y_te, X_cluster, Y_cluster ):
    # print(X_cluster.shape)
    channels = X_cluster.shape[1]
    n_classes = np.max(Y_cluster) + 1
    model = BassNet(channels, n_classes=n_classes)

    train = (X_cluster, Y_cluster)
    test = (X_te, Y_te)

    batchsize = 1000
    n_epoch = Config.epoch

    N = len(train[1])  # training data size
    N_test = len(test[1])
    classifier_model = SoftmaxClassifier(model) 
    optimizer = optimizers.Adam()
    optimizer.setup(classifier_model)
    out = Config.out
    # Learning loop
    for epoch in six.moves.range(1, n_epoch + 1):
        print('epoch', epoch)

        # training
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        start = time.time()
        for i in six.moves.range(0, N, batchsize):
            x = chainer.Variable(np.asarray(train[0][perm[i:i + batchsize]], dtype=np.float32))
            t = chainer.Variable(np.asarray(train[1][perm[i:i + batchsize]], dtype=np.int32))
            # Pass the loss function (Classifier defines it) and its arguments
            optimizer.update(classifier_model, x, t)

            if epoch == 1 and i == 0:
                with open('{}graph.dot'.format(out), 'w') as o:
                    g = computational_graph.build_computational_graph(
                        (classifier_model.loss,))
                    o.write(g.dump())
                print('graph generated')

            sum_loss += float(classifier_model.loss.data) * len(t.data)
            sum_accuracy += float(classifier_model.accuracy.data) * len(t.data)
        # do cross validation and thresholding here
        end = time.time()
        elapsed_time = end - start
        throughput = N / elapsed_time
        print('train mean loss={}, accuracy={}, throughput={} images/sec'.format(
            sum_loss / N, sum_accuracy / N, throughput))

    batchsize = Config.batchsize
    train = (X_tr2, Y_tr2)
    N = len(train[1])
    model_2nd_stage = Configuration2(model)
    if Config.loss == "tanh_cross_entropy":
        classifier_model_2nd_stage = TanhClassifier(model_2nd_stage)
    else:
        classifier_model_2nd_stage = SigmoidClassifier(model_2nd_stage)
    
    optimizer_2nd_stage = optimizers.Adam()
    optimizer_2nd_stage.setup(classifier_model_2nd_stage)
    # Learning loop
    for epoch in six.moves.range(1, n_epoch + 1):
        print('epoch', epoch)

        # training
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        start = time.time()
        for i in six.moves.range(0, N, batchsize):
            x = chainer.Variable(np.asarray(train[0][perm[i:i + batchsize]], dtype=np.float32))
            t = chainer.Variable(np.asarray(train[1][perm[i:i + batchsize]], dtype=np.int32))
            # Pass the loss function (Classifier defines it) and its arguments
            optimizer_2nd_stage.update(classifier_model_2nd_stage, x, t)

            if epoch == 1 and i == 0:
                with open('{}graph.dot'.format(out), 'w') as o:
                    g = computational_graph.build_computational_graph(
                        (classifier_model_2nd_stage.loss,))
                    o.write(g.dump())
                print('graph generated')

            sum_loss += float(classifier_model_2nd_stage.loss.data) * len(t.data)
            sum_accuracy += float(classifier_model_2nd_stage.accuracy.data) * len(t.data)
        # do cross validation and thresholding here
        end = time.time()
        elapsed_time = end - start
        throughput = N / elapsed_time
        print('train mean loss={}, accuracy={}, throughput={} images/sec'.format(
            sum_loss / N, sum_accuracy / N, throughput))

        # evaluation
        sum_accuracy = 0
        sum_loss = 0
        for i in six.moves.range(0, N_test, batchsize):
            index = np.asarray(list(range(i, i + batchsize)))
            x = chainer.Variable(np.asarray(test[0][i:i + batchsize], dtype=np.float32))
            t = chainer.Variable(np.asarray(test[1][i:i + batchsize], dtype=np.int32))
            with chainer.no_backprop_mode():
                # When back propagation is not necessary,
                # we can omit constructing graph path for better performance.
                # `no_backprop_mode()` is introduced from chainer v2,
                # while `volatile` flag was used in chainer v1.
                loss = classifier_model_2nd_stage(x, t)
            sum_loss += float(loss.data) * len(t.data)
            sum_accuracy += float(classifier_model_2nd_stage.accuracy.data) * len(t.data)

        print('test  mean loss={}, accuracy={}'.format(
            sum_loss / N_test, sum_accuracy / N_test))
        predicted_output  = utils.get_output_by_activation(model_2nd_stage, X_te)
        precision, recall, _ = utils.get_model_stats(predicted_output, Y_te)
        print('test precision={}, recall={}'. format(precision, recall))

    return model_2nd_stage

# run_classification()
