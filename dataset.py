import numpy as np
import urllib.request
import os
import tarfile
import pickle
import scipy.io
from sklearn.datasets import fetch_mldata
half_patch = 1

def get_mnist():
    mnist = fetch_mldata('MNIST original', data_home=".")
    x = mnist.data
    y = mnist.target
    print(x.shape,y.shape)
    # reshape to (#data, #channel, width, height)
    x = np.reshape(x, (x.shape[0], 1, 28, 28)) / 255.
    x_tr = np.asarray(x[:60000], dtype=np.float32)
    y_tr = np.asarray(y[:60000], dtype=np.int32)
    x_te = np.asarray(x[60000:], dtype=np.float32)
    y_te = np.asarray(y[60000:], dtype=np.int32)
    return (x_tr, y_tr), (x_te, y_te)

def get_custom_data(n_labeled,n_unlabeled):
    N_tr = n_labeled + n_unlabeled
    N_pos = (N_tr)//2  # number of points per class
    D = 2  # dimensionality
    X_tr = np.zeros((N_tr , D))  # data matrix (each row = single example)
    Y_tr = np.zeros(N_tr , dtype=np.int32)  # class labels
    x = 30 * np.random.rand(N_pos).astype(np.float32)
    y = 5*x + 9
    indx = range(0,N_pos)
    X_tr[indx] = np.transpose(np.vstack((x,y)))
    Y_tr[indx] = 1
    N_neg = N_tr - N_pos
    x = 30 * np.random.rand(N_neg).astype(np.float32)
    y = 5*x - 9
    indx = range(N_pos,N_tr)
    X_tr[indx] = np.transpose(np.vstack((x,y)))
    Y_tr[indx] = -1

    N_te = 2 * (n_labeled + n_unlabeled)
    N_pos = (N_te) // 2  # number of points per class
    D = 2  # dimensionality
    X_te = np.zeros((N_te, D))  # data matrix (each row = single example)
    Y_te = np.zeros(N_te, dtype=np.int32)  # class labels
    x = 30 * np.random.rand(N_pos).astype(np.float32)
    y = 5 * x + 9
    indx = range(0, N_pos)
    X_te[indx] = np.transpose(np.vstack((x,y)))
    Y_te[indx] = 1
    N_neg = N_tr - N_pos
    x = 30 * np.random.rand(N_neg).astype(np.float32)
    y = 5 * x - 9
    indx = range(N_pos, N_tr)
    X_te[indx] = np.transpose(np.vstack((x,y)))
    Y_te[indx] = -1
    return (X_tr,Y_tr), (X_te,Y_te)


# def get_indian_pines(n_unlabeled):
#     data,data_gt = load_data()
#     num_points = []
#     for i in range(0, data_gt.max() + 1):
#         num_points.append(len(np.where(data_gt == i)[0]))
#     num_points = np.argsort(num_points)
#     class_indx = num_points[-1]
#     X_tr_pos,Y_tr_pos,patch_indices = get_positive_data(data,data_gt,class_indx)
#     data_sel = np.zeros(data_gt.shape, dtype=np.bool)
#     data_sel[patch_indices] = True
#     indx = np.where(data_sel == False)
#     if (n_unlabeled == -1):
#         n_unlabeled = indx[0].shape[0]
#     neg_indx = np.random.choice(indx[0], n_unlabeled, replace=False)
#     neg_indx = (indx[0][neg_indx],indx[1][neg_indx])
#     X_tr_neg = get_data_from_indices(neg_indx,data)
#     Y_tr_neg = data_gt[neg_indx]
#     Y_tr_neg = Y_tr_neg.astype(np.int32)
#     Y_tr_neg[np.where(Y_tr_neg != class_indx)] = -1
#     Y_tr_neg[np.where(Y_tr_neg != -1)] = 1
#     X_tr = np.concatenate((X_tr_pos, X_tr_neg), axis = 0)
#     Y_tr = np.concatenate((Y_tr_pos,Y_tr_neg), axis = 0)
#     # data_sel[neg_indx] = True
#     # test_indx = np.where(data_sel == False)
#     # rand_indx = np.random.choice(test_indx[0], 15000, replace=False)
#     # test_indx = (test_indx[0][rand_indx],test_indx[1][rand_indx])
#     # X_te = get_data_from_indices(test_indx,data)
#     # Y_te = data_gt[test_indx]
#     # Y_te = Y_te.astype(np.int32)
#     # Y_te[np.where(Y_te != class_indx)] = -1
#     # Y_te[np.where(Y_te != -1)] = 1
#     print("n_unlabeled",n_unlabeled)
#     print("X_tr",X_tr.shape)
#     return (X_tr,Y_tr) , (X_tr,Y_tr), n_unlabeled


#
# def get_positive_data(data,data_gt,class_indx):
#     patch_end_points = None
#     found = False
#     row = 12
#     col = 12
#     for i in range(row, data_gt.shape[0]):
#         for j in range(col, data_gt.shape[1]):
#             if found == False:
#                 end_row = i + row
#                 end_col = j + col
#                 if end_row > data_gt.shape[0] - 1 or end_col > data_gt.shape[1] - 1:
#                    continue
#                 my_patch = data_gt[i:end_row + 1, :][:, j:end_col + 1]
#                 class_pixels = np.where(my_patch == class_indx)
#                 purity = (class_pixels[1].shape[0] * 100) / (my_patch.shape[0] * my_patch.shape[1] * 1.0)
#                 if (purity >= 95 and purity < 100):
#                     patch_end_points = [i, end_row, j, end_col]
#                     found = True
#             else:
#                 break
#                 break
#     patch_indices = convert_patch_data_points(patch_end_points)
#     if patch_indices is None:
#         Y = None
#     else:
#         Y = np.zeros((len(patch_indices[0]),),dtype=np.int32)
#         Y.fill(1)
#     X = get_data_from_indices(patch_indices,data)
#     return X,Y,patch_indices
#
# def get_data_from_indices(indices,data):
#     if indices is None:
#         return None
#     train_X = np.zeros((len(indices[0]),data.shape[2],3,3))
#     train_X.fill(-1)
#     rows = indices[0]
#     cols = indices[1]
#     half_patch = 1
#     for i in range(0,len(indices[0])):
#         row_start = rows[i] - half_patch
#         row_end = rows[i] + half_patch
#         col_start = cols[i] - half_patch
#         col_end = cols[i] + half_patch
#         # print("prev", row_start, row_end, col_start, col_end, rows[i], cols[i])
#         if row_start < 0 :
#             row_start = 0
#             row_end = row_end + (-(rows[i] - half_patch)) if row_end + (-(rows[i] - half_patch)) <= data.shape[0] - 1 else row_end
#         elif row_end > data.shape[0] - 1:
#             row_end = data.shape[0]-1
#             row_start = row_start - (rows[i]+half_patch - data.shape[0]+1) if row_start - (rows[i]+half_patch - data.shape[0]+1)>=0 else row_start
#         if col_start < 0:
#             col_start = 0
#             col_end = col_end + (-(cols[i]-half_patch)) if col_end + (-(cols[i]-half_patch))<= data.shape[1]-1 else col_end
#         elif col_end > data.shape[1] -1:
#             col_end = data.shape[1] - 1
#             col_start = col_start - (cols[i]+half_patch-data.shape[1]+1) if col_start - (cols[i]+half_patch-data.shape[1]+1)>=0 else col_start
#         block = data[row_start: row_end+1][:,col_start:col_end+1]
#         block = np.rot90(block, 1, (0, 2))
#         # print (block.shape,row_start,row_end,col_start,col_end,rows[i],cols[i])
#         train_X[i] = block
#     return train_X

# def convert_patch_data_points(patch_end_points):
#     if patch_end_points is None:
#         return None
#     row_size = patch_end_points[1] + 1 - patch_end_points[0]
#     col_size = patch_end_points[3] + 1 - patch_end_points[2]
#     total_points = row_size * col_size
#     rows = np.zeros(total_points, dtype='int64')
#     cols = np.zeros(total_points, dtype='int64')
#     for k in range(patch_end_points[0], patch_end_points[1] + 1):
#         for l in range(patch_end_points[2], patch_end_points[3] + 1):
#             index = (k - patch_end_points[0]) * col_size + l - patch_end_points[2]
#             rows[index] = k
#             cols[index] = l
#
#     patch_indices = (rows, cols)
#     return patch_indices


# NEW CODE
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
    # labels[np.where(labels == -1)] = 0
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



def load_saved_data():
    mat = scipy.io.loadmat("mldata/Indian_pines_Binary_Full_Train_patch_3.mat")
    X_tr = mat["train_patch"]
    Y_tr = mat["train_labels"]
    mat = scipy.io.loadmat("mldata/Indian_pines_Binary_Test_patch_3.mat")
    X_te = mat["test_patch"]
    Y_te = mat["test_labels"]
    Y_tr[np.where(Y_tr == 0)] = -1
    Y_te[np.where(Y_te == 0)] = -1
    Y_te = np.reshape(Y_te, (Y_te.shape[0] * Y_te.shape[1]))
    Y_tr = np.reshape(Y_tr, (Y_tr.shape[0] * Y_tr.shape[1]))
    return (X_tr, Y_tr), (X_te, Y_te)



def binarize_mnist_class(_trainY, _testY):
    trainY = np.ones(len(_trainY), dtype=np.int32)
    trainY[_trainY % 2 == 1] = -1
    testY = np.ones(len(_testY), dtype=np.int32)
    testY[_testY % 2 == 1] = -1
    return trainY, testY


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def conv_data2image(data):
    return np.rollaxis(data.reshape((3, 32, 32)), 0, 3)


def get_cifar10(path="./mldata"):

    if not os.path.isdir(path):
        os.mkdir(path)
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    file_name = os.path.basename(url)
    full_path = os.path.join(path, file_name)
    folder = os.path.join(path, "cifar-10-batches-py")
    # if cifar-10-batches-py folder doesn't exists, download from website
    if not os.path.isdir(folder):
        print("download the dataset from {} to {}".format(url, path))
        urllib.request.urlretrieve(url, full_path)
        with tarfile.open(full_path) as f:
            f.extractall(path=path)
        urllib.request.urlcleanup()

    x_tr = np.empty((0,32*32*3))
    y_tr = np.empty(1)
    for i in range(1,6):
        fname = os.path.join(folder, "%s%d" % ("data_batch_", i))
        data_dict = unpickle(fname)
        if i == 1:
            x_tr = data_dict['data']
            y_tr = data_dict['labels']
        else:
            x_tr = np.vstack((x_tr, data_dict['data']))
            y_tr = np.hstack((y_tr, data_dict['labels']))

    data_dict = unpickle(os.path.join(folder, 'test_batch'))
    x_te = data_dict['data']
    y_te = np.array(data_dict['labels'])

    bm = unpickle(os.path.join(folder, 'batches.meta'))
    # label_names = bm['label_names']
    # rehape to (#data, #channel, width, height)
    x_tr = np.transpose(np.reshape(x_tr,(np.shape(x_tr)[0],32,32,3)),(0,3,1,2))
    x_te = np.transpose(np.reshape(x_te,(np.shape(x_te)[0],32,32,3)),(0,3,1,2))
    return (x_tr, y_tr), (x_te, y_te)  # , label_names


def binarize_cifar10_class(_trainY, _testY):
    trainY = np.ones(len(_trainY), dtype=np.int32)
    trainY[(_trainY==2)|(_trainY==3)|(_trainY==4)|(_trainY==5)|(_trainY==6)|(_trainY==7)] = -1
    testY = np.ones(len(_testY), dtype=np.int32)
    testY[(_testY==2)|(_testY==3)|(_testY==4)|(_testY==5)|(_testY==6)|(_testY==7)] = -1
    return trainY, testY


def make_dataset(dataset, n_labeled, n_unlabeled):
    def make_PU_dataset_from_binary_dataset(x, y, labeled=n_labeled, unlabeled=n_unlabeled):
        # print("PU dataset",labeled,unlabeled)
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        # print(X.shape,Y.shape)
        assert(len(X) == len(Y))
        perm = np.random.permutation(len(Y))
        X, Y = X[perm], Y[perm]
        n_p = (Y == positive).sum()
        n_lp = labeled
        n_n = (Y == negative).sum()
        n_u = unlabeled
        if labeled + unlabeled == len(X):
            n_up = n_p - n_lp
            # print (n_up,"labeled+unlabeled")
        elif unlabeled == len(X):
            n_up = n_p
        else:
            raise ValueError("Only support |P|+|U|=|X| or |U|=|X|.")
        prior = float(n_up) / float(n_u)
        # print(prior)
        # prior = 0.5906844741235392
        Xlp = X[Y == positive][:n_lp]
        Xup = np.concatenate((X[Y == positive][n_lp:], Xlp), axis=0)[:n_up]
        Xun = X[Y == negative]
        X = np.asarray(np.concatenate((Xlp, Xup, Xun), axis=0), dtype=np.float32)
        # print(X.shape)
        Y = np.asarray(np.concatenate((np.ones(n_lp), -np.ones(n_u))), dtype=np.int32)
        perm = np.random.permutation(len(Y))
        X, Y = X[perm], Y[perm]
        return X, Y, prior

    def make_PN_dataset_from_binary_dataset(x, y):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        n_p = (Y == positive).sum()
        n_n = (Y == negative).sum()
        Xp = X[Y == positive][:n_p]
        Xn = X[Y == negative][:n_n]
        X = np.asarray(np.concatenate((Xp, Xn)), dtype=np.float32)
        Y = np.asarray(np.concatenate((np.ones(n_p), -np.ones(n_n))), dtype=np.int32)
        perm = np.random.permutation(len(Y))
        X, Y = X[perm], Y[perm]
        return X, Y

    (_trainX, _trainY), (_testX, _testY) = dataset
    trainX, trainY, prior = make_PU_dataset_from_binary_dataset(_trainX, _trainY)
    testX, testY = make_PN_dataset_from_binary_dataset(_testX, _testY)
    print("training:{}".format(trainX.shape))
    print("test:{}".format(testX.shape))
    return list(zip(trainX, trainY)), list(zip(testX, testY)), prior, testX, testY,trainX, trainY


def load_dataset(dataset_name, n_labeled, n_unlabeled):
    # print(dataset_name,n_labeled,n_unlabeled)
    if dataset_name == "mnist":
        (trainX, trainY), (testX, testY) = get_mnist()
        trainY, testY = binarize_mnist_class(trainY, testY)
    elif dataset_name == "cifar10":
        (trainX, trainY), (testX, testY) = get_cifar10()
        trainY, testY = binarize_cifar10_class(trainY, testY)
    elif dataset_name == 'indian_pines':
        # (trainX, trainY), (testX, testY), class_indx = get_indian_pines(n_labeled, n_unlabeled)
        # # print("trainX.shape", trainX.shape)
        # trainY = binarize_indian_pines(trainY, class_indx)
        # testY = binarize_indian_pines(testY, class_indx)
        (trainX, trainY), (testX, testY) = load_saved_data()
        # (trainX, trainY), (testX,testY), n_unlabeled = get_indian_pines(n_unlabeled)
    elif dataset_name == 'custom':
        (trainX,trainY), (testX,testY) = get_custom_data(n_labeled,n_unlabeled)
    else:
        raise ValueError("dataset name {} is unknown.".format(dataset_name))
    XYtrain, XYtest, prior,testX, testY,trainX, trainY = make_dataset(((trainX, trainY), (testX, testY)), n_labeled, n_unlabeled)
    return XYtrain, XYtest, prior, testX, testY,trainX, trainY
