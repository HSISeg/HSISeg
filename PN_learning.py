import numpy as np
import six
import time
from chainer import Variable, optimizers, serializers, initializers, cuda
import pickle
import scipy.io
import chainer
import chainer.functions as F
from chainer import computational_graph
import chainer.links as L
from chainer import Chain, cuda

half_patch = 1

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

        self.nbands = 10
        self.input_channels = channels
        self.band_size = self.block1_nfilters // self.nbands
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
        h1, h2, h3, h4, h5, h6, h7, h8, h9, h10 = F.split_axis(h, self.nbands, axis=1)
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
        h = F.sigmoid(h)
        h = F.reshape(h, (h.shape[0],))
        return h


def shuffle_data(X_tr, Y_tr, X_te, Y_te):
    perm = np.random.permutation(Y_tr.shape[0])
    X_tr, Y_tr = X_tr[perm], Y_tr[perm]
    perm = np.random.permutation(Y_te.shape[0])
    X_te, Y_te = X_te[perm], Y_te[perm]
    return (X_tr, Y_tr), (X_te, Y_te)


def get_pos_class_index(data_gt):
    num_points = []
    for i in range(0, data_gt.max() + 1):
        num_points.append(len(np.where(data_gt == i)[0]))
    num_points = np.argsort(num_points)
    class_indx = num_points[-2]
    return class_indx


def preprocess_data(data):
    data = normalization(data)
    input_mat = np.transpose(data, (2, 0, 1))
    MEAN_ARRAY = get_band_mean_arr(input_mat)
    for i in range(0,MEAN_ARRAY.shape[0]):
        input_mat[i, :, :] = input_mat[i, :, :] - MEAN_ARRAY[i]
    data = np.transpose(input_mat, (2, 1, 0))
    return data


def binarize_indian_pines(labels, class_indx):
    labels = labels.astype(np.int32)
    labels[np.where(labels != class_indx)] = -1
    labels[np.where(labels != -1)] = 1
    labels[np.where(labels == -1)] = 0
    return labels


def get_unlabelled_data(data, data_gt, selected_data, n_train_unlabelled, is_random, postive_percentage, pos_class_indx):
    if is_random and postive_percentage is None:
        indx = np.where(selected_data == False)
        unlbl_indx = np.random.choice(indx[0], n_train_unlabelled, replace=False)
        unlbl_indx = (indx[0][unlbl_indx], indx[1][unlbl_indx])
        X_tr_unlbl = get_data_from_indices(unlbl_indx, data)
        Y_tr_unlbl = data_gt[unlbl_indx]
        selected_data[unlbl_indx] = True
    else:
        # select pos_class_indx % positive class pixels from unselected data for unlabelled data
        n_pos = int(n_train_unlabelled * (pos_class_indx/100))
        pos_indx = np.zeros(data_gt.shape, dtype=np.bool)
        pos_indx[np.where(data_gt == pos_class_indx)] = True
        pos_indx[np.where(selected_data == True)] = False
        indx = np.where(pos_indx == True)
        if n_pos > indx[0].shape[0]:
            n_pos = indx[0].shape[0]
        pos_indx_sel = np.random.choice(indx[0], n_pos, replace=False)
        pos_indx_sel = (indx[0][pos_indx_sel], indx[1][pos_indx_sel])
        X_tr_pos_unlbl = get_data_from_indices(pos_indx_sel, data)
        Y_tr_pos_unlbl = data_gt[pos_indx_sel]
        selected_data[pos_indx_sel] = True
        # rest pixels select randomly from unselected data
        indx = np.where(selected_data == False)
        n_rest_unbl = n_train_unlabelled - n_pos
        rest_unlbl_indx = np.random.choice(indx[0], n_rest_unbl, replace=False)
        rest_unlbl_indx = (indx[0][rest_unlbl_indx], indx[1][rest_unlbl_indx])
        X_tr_rest_unlbl = get_data_from_indices(rest_unlbl_indx, data)
        Y_tr_rest_unlbl = data_gt[rest_unlbl_indx]
        selected_data[rest_unlbl_indx] = True
        #concatenate both train data
        X_tr_unlbl = np.concatenate((X_tr_pos_unlbl, X_tr_rest_unlbl), axis=0)
        Y_tr_unlbl = np.concatenate((Y_tr_pos_unlbl, Y_tr_rest_unlbl), axis=0)
    return X_tr_unlbl, Y_tr_unlbl, selected_data


def get_indian_pines(n_train_labelled, n_train_unlabelled):
    data, data_gt = load_data()
    data = preprocess_data(data)
    class_indx = get_pos_class_index(data_gt)
    # get positive labelled data
    X_tr_pos, Y_tr_pos, patch_indices = get_positive_data(n_train_labelled, data, data_gt, class_indx)

    data_sel = np.zeros(data_gt.shape, dtype=np.bool)
    data_sel[patch_indices] = True

    # select n_train_unlabelled data from the rest
    X_tr_unlbl, Y_tr_unlbl, data_sel = get_unlabelled_data(data, data_gt, data_sel, n_train_unlabelled, True, None, class_indx)

    X_tr = np.concatenate((X_tr_pos, X_tr_unlbl), axis=0)
    Y_tr = np.concatenate((Y_tr_pos, Y_tr_unlbl), axis=0)

    #rest testing
    test_indx = np.where(data_sel == False)
    X_te = get_data_from_indices(test_indx,data)
    Y_te = data_gt[test_indx]

    return (X_tr, Y_tr), (X_te, Y_te), class_indx


def normalization(input_mat):
    input_mat -= np.min(input_mat)
    input_mat /= np.max(input_mat)
    return input_mat


def load_data():
    mat = scipy.io.loadmat("Indian_pines_gt.mat")
    Y = mat["indian_pines_gt"]
    Y = Y.astype(np.int32)
    mat = scipy.io.loadmat("Indian_pines.mat")
    X = mat['indian_pines']
    X = X.astype(np.float32)
    return X, Y


def convert_patch_data_points(patch_end_points):
    if patch_end_points is None:
        return None
    row_size = patch_end_points[1] + 1 - patch_end_points[0]
    col_size = patch_end_points[3] + 1 - patch_end_points[2]
    total_points = row_size * col_size
    rows = np.zeros(total_points, dtype='int64')
    cols = np.zeros(total_points, dtype='int64')
    for k in range(patch_end_points[0], patch_end_points[1] + 1):
        for l in range(patch_end_points[2], patch_end_points[3] + 1):
            index = (k - patch_end_points[0]) * col_size + l - patch_end_points[2]
            rows[index] = k
            cols[index] = l

    patch_indices = (rows, cols)
    return patch_indices


def get_positive_data(n_train_labelled, data, data_gt, class_indx):
    patch_end_points = None
    found = False
    row_len = int(np.sqrt(n_train_labelled))
    row = row_len - 1
    col = row_len - 1
    for i in range(half_patch, data_gt.shape[0] - row - half_patch):
        for j in range(half_patch, data_gt.shape[1] - col - half_patch):
            if found == False:
                end_row = i + row
                end_col = j + col
                if end_row > data_gt.shape[0] - 1 or end_col > data_gt.shape[1] - 1:
                    continue
                my_patch = data_gt[i:end_row + 1, :][:, j:end_col + 1]
                class_pixels = np.where(my_patch == class_indx)
                purity = (class_pixels[1].shape[0] * 100) / (my_patch.shape[0] * my_patch.shape[1] * 1.0)
                if (purity >= 95 and purity < 100):
                    patch_end_points = [i, end_row, j, end_col]
                    found = True
            else:
                break
                break
    patch_indices = convert_patch_data_points(patch_end_points)
    if patch_indices is None:
        Y = None
    else:
        Y = np.zeros((len(patch_indices[0]),), dtype=np.int32)
        Y.fill(class_indx)
    X = get_data_from_indices(patch_indices, data)
    return X, Y, patch_indices

def get_band_mean_arr(input_mat):
    BAND = input_mat.shape[0]
    MEAN_ARRAY = np.ndarray(shape=(BAND,), dtype=np.float32)
    for i in range(BAND):
        MEAN_ARRAY[i] = np.mean(input_mat[i, :, :])
    return MEAN_ARRAY

def get_data_from_indices(indices, data):
    if indices is None:
        return None
    train_X = np.zeros((len(indices[0]), data.shape[2], 3, 3))
    train_X.fill(-1)
    rows = indices[0]
    cols = indices[1]
    for i in range(0, len(indices[0])):
        row_start = rows[i] - half_patch
        row_end = rows[i] + half_patch
        col_start = cols[i] - half_patch
        col_end = cols[i] + half_patch
        # print("prev", row_start, row_end, col_start, col_end, rows[i], cols[i])
        if row_start < 0:
            row_start = 0
            row_end = row_end + (-(rows[i] - half_patch)) if row_end + (-(rows[i] - half_patch)) <= data.shape[
                0] - 1 else row_end
        elif row_end > data.shape[0] - 1:
            row_end = data.shape[0] - 1
            row_start = row_start - (rows[i] + half_patch - data.shape[0] + 1) if row_start - (
                    rows[i] + half_patch - data.shape[0] + 1) >= 0 else row_start
        if col_start < 0:
            col_start = 0
            col_end = col_end + (-(cols[i] - half_patch)) if col_end + (-(cols[i] - half_patch)) <= data.shape[
                1] - 1 else col_end
        elif col_end > data.shape[1] - 1:
            col_end = data.shape[1] - 1
            col_start = col_start - (cols[i] + half_patch - data.shape[1] + 1) if col_start - (
                    cols[i] + half_patch - data.shape[1] + 1) >= 0 else col_start
        block = data[row_start: row_end + 1][:, col_start:col_end + 1]
        block = np.rot90(block, 1, (0, 2))
        # print (block.shape,row_start,row_end,col_start,col_end,rows[i],cols[i])
        train_X[i] = block
    return train_X


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
        self.loss = F.sigmoid_cross_entropy(y, t)
        self.accuracy = F.binary_accuracy(y, t)
        return self.loss


def get_accuracy(model, x, t):
    size = len(t)
    x = np.asarray(x, dtype=np.float32)
    h = model.calculate(x)
    # h = np.reshape(np.sign(model.calculate(x).data), size)
    if isinstance(h, chainer.Variable):
        h = h.data
    if isinstance(t, chainer.Variable):
        t = t.data
    negative, positive = np.unique(t)
    positive_data = t == positive
    n_positive = positive_data.sum()
    n_negative = size - n_positive
    n_positive_match = (h[positive_data] == t[positive_data]).sum()
    n_negative_match = (h[np.logical_not(positive_data)] == t[np.logical_not(positive_data)]).sum()
    print("n_positive_test", n_positive, "n_negative_test", n_negative, n_positive_match, n_negative_match)
    accuracy = (h == t).sum() / size
    return accuracy


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


# (X_tr, Y_tr), (X_te, Y_te), class_indx = get_indian_pines(121, 7000)
(X_tr, Y_tr), (X_te, Y_te) = load_saved_data()
# print (class_indx,"class_indx")
# Y_tr = binarize_indian_pines(Y_tr, class_indx)
# Y_te = binarize_indian_pines(Y_te, class_indx)
# full_train = {}
# full_train["train_patch"] = X_tr
# full_train["train_labels"] = Y_tr
# scipy.io.savemat("mldata/Indian_Pines_Binary_Full_Train_patch_my" + str(half_patch * 2 + 1) + ".mat", full_train)
# test = {}
# test["test_patch"] = X_te
# test["test_labels"] = Y_te
# scipy.io.savemat("mldata/Indian_Pines_Binary_Test_patch_my" + str(half_patch * 2 + 1) + ".mat", test)
print(np.unique(Y_tr),"Y_tr_labels",np.unique(Y_te),"Y_te_labels")
(X_tr, Y_tr), (X_te, Y_te) = shuffle_data(X_tr, Y_tr, X_te, Y_te)

channels = X_tr.shape[1]
model = Configuration1(channels)

train = (X_tr, Y_tr)
test = (X_te, Y_te)

batchsize = 1000
n_epoch = 100

N = len(train[1])  # training data size
N_test = len(test[1])
classifier_model = SoftmaxClassifier(model)
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

accuracy = get_accuracy(model, X_te, Y_te)
print("accuracy", accuracy)

# test data size
