import numpy as np
import six
import time, utils, Config
from chainer import Variable, optimizers, serializers, initializers, cuda
import scipy.io
import chainer
import chainer.functions as F
from chainer import computational_graph
import chainer.links as L
from chainer import Chain, cuda
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

class MultiLayerPerceptron(Chain):
    def __init__(self):
        super(MultiLayerPerceptron, self).__init__(
            # input size of each layer will be inferred when set `None`
            l1=L.Linear(None, 300),  # n_in -> n_units
            l2=L.Linear(None, 300),  # n_units -> n_units
            l3=L.Linear(None, 1),  # n_units -> n_out
        )

    # def __init__(self):
        # super(MultiLayerPerceptron, self).__init__(l1=L.Linear(None, 300, nobias=True),
        #                                            b1=L.BatchNormalization(300),
        #                                            l2=L.Linear(300, 300, nobias=True),
        #                                            b2=L.BatchNormalization(300),
        #                                            l3=L.Linear(300, 300, nobias=True),
        #                                            b3=L.BatchNormalization(300),
        #                                            l4=L.Linear(300, 300, nobias=True),
        #                                            b4=L.BatchNormalization(300),
        #                                            l5=L.Linear(300, 1))
        # self.af = F.relu


    def calculate(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h = self.l3(h2)
        h = F.sigmoid(h)
        h = F.reshape(h, (h.shape[0],))
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

    def __call__(self, x):
        # print(x.shape)
        h = self.calculate(x)
        print(h.shape)
        return h


class Configuration1(Chain):
    def __init__(self, channels):
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
        self.band_size = self.block1_nfilters // self.nbands
        self.threshold = 0.5
        self.auc = None
        super(Configuration1, self).__init__(
            l1=L.Convolution2D(channels, self.block1_nfilters, self.block1_patch_size),
            l2=L.Convolution2D(self.image_length, self.block2_nfilters_1, (self.image_width, self.block2_patch_size_1)),
            l3=L.Convolution2D(self.block2_nfilters_1, self.block2_nfilters_2, (1, self.block2_patch_size_2)),
            l4=L.Convolution2D(self.block2_nfilters_2, self.block2_nfilters_3, (1, self.block2_patch_size_3)),
            l5=L.Convolution2D(self.block2_nfilters_3, self.block2_nfilters_4, (1, self.block2_patch_size_4)),
            l6=L.Linear(None, 100),
            l7=L.Linear(100, 1)
        )

    def __call__(self, x):
        # print(x.shape)
        h = self.calculate(x)
        return h

    def calculate(self, x):
        h = self.l1(x)
        h = self.af_block1(h)
        split_h = F.split_axis(h, self.nbands, axis=1, force_tuple=True)
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
        h = F.reshape(h, (h.shape[0],))
        return h


def shuffle_data(X_tr, Y_tr, X_te, Y_te):
    perm = np.random.permutation(Y_tr.shape[0])
    X_tr, Y_tr = X_tr[perm], Y_tr[perm]
    perm = np.random.permutation(Y_te.shape[0])
    X_te, Y_te = X_te[perm], Y_te[perm]
    return (X_tr, Y_tr), (X_te, Y_te)

class SigmoidClassifier(chainer.Chain):
    """Classifier is for calculating loss, from predictor's output.
    predictor is a model that predicts the probability of each label.
    """
    def __init__(self, predictor):
        super(SigmoidClassifier, self).__init__()
        with self.init_scope():
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


def load_saved_data():
    mat = scipy.io.loadmat("mldata/Indian_pines_Binary_Full_Train_patch_3.mat")
    X_tr = mat["train_patch"]
    Y_tr = mat["train_labels"]
    mat = scipy.io.loadmat("mldata/Indian_pines_Binary_Test_patch_3.mat")
    X_te = mat["test_patch"]
    Y_te = mat["test_labels"]
    Y_te = np.reshape(Y_te, (Y_te.shape[0] * Y_te.shape[1]))
    Y_tr = np.reshape(Y_tr, (Y_tr.shape[0] * Y_tr.shape[1]))
    return (X_tr, Y_tr), (X_te, Y_te)

def run_classification():
    (X_tr, Y_tr), (X_te, Y_te) = load_saved_data()
    (X_tr, Y_tr), (X_te, Y_te) = shuffle_data(X_tr, Y_tr, X_te, Y_te)

    channels = X_tr.shape[1]
    model = Configuration1(channels)
    # model = MultiLayerPerceptron()

    train = (X_tr, Y_tr)
    test = (X_te, Y_te)

    batchsize = Config.batchsize
    n_epoch = Config.epoch

    N = len(train[1])  # training data size
    N_test = len(test[1])
    classifier_model = SigmoidClassifier(model)
    optimizer = optimizers.Adam()
    optimizer.setup(classifier_model)
    out = 'result'
    # Learning loop
    for epoch in six.moves.range(1, n_epoch + 1):
        print('epoch', epoch)

        # training
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        start = time.time()
        for i in six.moves.range(0, N, batchsize):
            x = chainer.Variable(np.asarray(train[0][perm[i:i + batchsize]],dtype=np.float32))
            t = chainer.Variable(np.asarray(train[1][perm[i:i + batchsize]],dtype=np.int32))
            # Pass the loss function (Classifier defines it) and its arguments
            optimizer.update(classifier_model, x, t)

            if epoch == 1 and i == 0:
                with open('{}/graph.dot'.format(out), 'w') as o:
                    g = computational_graph.build_computational_graph(
                        (classifier_model.loss,))
                    o.write(g.dump())
                print('graph generated')

            sum_loss += float(classifier_model.loss.data) * len(t.data)
            sum_accuracy += float(classifier_model.accuracy.data) * len(t.data)
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
            x = chainer.Variable(np.asarray(test[0][i:i + batchsize],dtype=np.float32))
            t = chainer.Variable(np.asarray(test[1][i:i + batchsize],dtype=np.int32))
            with chainer.no_backprop_mode():
                # When back propagation is not necessary,
                # we can omit constructing graph path for better performance.
                # `no_backprop_mode()` is introduced from chainer v2,
                # while `volatile` flag was used in chainer v1.
                loss = classifier_model(x, t)
            sum_loss += float(loss.data) * len(t.data)
            sum_accuracy += float(classifier_model.accuracy.data) * len(t.data)

        print('test  mean loss={}, accuracy={}'.format(
            sum_loss / N_test, sum_accuracy / N_test))

    precision, recall, (tn, fp, fn, tp) = get_accuracy(model, X_te, Y_te)
    # print("accuracy", accuracy)
    # print("precision", precision, "recall", recall, "tn", tn, "fp", fp, "fn", fn, "tp", tp)
    # test data size
    return precision, recall, (tn, fp, fn, tp)


def train(X_tr, Y_tr, X_te, Y_te):
    channels = X_tr.shape[1]
    model = Configuration1(channels)
    # model = MultiLayerPerceptron()

    train = (X_tr, Y_tr)
    test = (X_te, Y_te)

    batchsize = Config.batchsize
    n_epoch = Config.epoch

    N = len(train[1])  # training data size
    N_test = len(test[1])
    classifier_model = SigmoidClassifier(model)
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
                loss = classifier_model(x, t)
            sum_loss += float(loss.data) * len(t.data)
            sum_accuracy += float(classifier_model.accuracy.data) * len(t.data)

        print('test  mean loss={}, accuracy={}'.format(
            sum_loss / N_test, sum_accuracy / N_test))
        predicted_output  = utils.get_output_by_activation(model, X_te)
        precision, recall, _ = utils.get_model_stats(predicted_output, Y_te)
        print('test precision={}, recall={}'. format(precision, recall))

    # precision, recall, (tn, fp, fn, tp) = get_accuracy(model, X_te, Y_te)
    # print("accuracy", accuracy)
    # print("precision", precision, "recall", recall, "tn", tn, "fp", fp, "fn", fn, "tp", tp)
    # test data size
    return model

# run_classification()
