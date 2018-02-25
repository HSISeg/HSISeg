import numpy as np
import scipy.io as io


def load_data():
    input_mat = io.loadmat("mldata/Indian_pines_Preprocessed_patch_3.mat")['preprocessed_img']
    target_mat = io.loadmat("mldata/Indian_pines_Preprocessed_patch_3.mat")['preprocessed_gt']
    target_mat = np.asarray(target_mat, dtype=np.int32)
    input_mat = np.asarray(input_mat, dtype=np.float32)
    return input_mat, target_mat


def get_classwise_pixel(target_mat):
    OUTPUT_CLASSES = np.max(target_mat) + 1
    CLASSES = []
    for i in range(OUTPUT_CLASSES):
        CLASSES.append(np.where(target_mat == i))

    return CLASSES

def get_test_train_PN_pixel(CLASSES, pos_label , neg_labels_list, train_pos_percentage, train_neg_percentage, is_random_neg):
    if len(CLASSES[pos_label][0]) == 0 or len(neg_labels_list) == 0:
        return None, None, None, None
    n_pos_pixels = len(CLASSES[pos_label][0])
    train_n_pos_pixels = (n_pos_pixels * train_pos_percentage) // 100
    if train_n_pos_pixels == 0:
        return None, None, None, None
    if pos_label in neg_labels_list:
        neg_labels_list.remove(pos_label)
    neg_pixels = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    indx = np.random.permutation(len(CLASSES[pos_label][0]))
    train_pos_pixels = (CLASSES[pos_label][0][indx[:train_n_pos_pixels]], CLASSES[pos_label][1][indx[:train_n_pos_pixels]])
    test_pos_pixels = (CLASSES[pos_label][0][indx[train_n_pos_pixels:]], CLASSES[pos_label][1][indx[train_n_pos_pixels:]])

    if is_random_neg:
        for i in range(len(neg_labels_list)):
            if len(CLASSES[neg_labels_list[i]][0]) > 0:
                neg_pixels = (np.concatenate((CLASSES[neg_labels_list[i]][0], neg_pixels[0])), np.concatenate((CLASSES[neg_labels_list[i]][1], neg_pixels[1])))
        if len(neg_pixels[0]) == 0:
            return  None, None, None, None
        train_n_neg_pixels = (len(neg_pixels[0]) * train_neg_percentage) // 100
        if train_n_neg_pixels == 0:
            return None, None, None, None
        indx = np.random.permutation(len(neg_pixels[0]))
        train_neg_pixels = (neg_pixels[0][indx[:train_n_neg_pixels]], neg_pixels[1][indx[:train_n_neg_pixels]])
        test_neg_pixels = (neg_pixels[0][indx[train_n_neg_pixels:]], neg_pixels[1][indx[train_n_neg_pixels:]])
    # If train_neg_percentage no of pixels taken for each class
    else:
        test_neg_pixels = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        train_neg_pixels = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        for i in range(len(neg_labels_list)):
            if len(CLASSES[neg_labels_list[i]][0]) > 0:
                train_n_neg_pixels_class = (len(CLASSES[neg_labels_list[i]][0]) *  train_neg_percentage) // 100
                if train_n_neg_pixels_class > 0:
                    indx = np.random.permutation(len(CLASSES[neg_labels_list[i]][0]))
                    train_neg_pixels = (np.concatenate((train_neg_pixels[0], CLASSES[neg_labels_list[i]][0][indx[:train_n_neg_pixels_class]] )),
                                        np.concatenate((train_neg_pixels[1], CLASSES[neg_labels_list[i]][1][indx[:train_n_neg_pixels_class]])))
                    test_neg_pixels = (np.concatenate((test_neg_pixels[0], CLASSES[neg_labels_list[i]][0][indx[train_n_neg_pixels_class:]] )),
                                        np.concatenate((test_neg_pixels[1], CLASSES[neg_labels_list[i]][1][indx[train_n_neg_pixels_class:]])))
        if len(train_neg_pixels[0]) == 0 or len(test_neg_pixels[0]) == 0:
            return None, None, None, None
    return train_pos_pixels, train_neg_pixels, test_pos_pixels, test_neg_pixels

def get_PN_data(pos_pixels, neg_pixels, input_mat, pos_label, neg_label):
    X = np.concatenate((input_mat[pos_pixels], input_mat[neg_pixels]))
    Y = np.asarray(np.concatenate((np.full(len(pos_pixels[0]), pos_label), np.full(len(neg_pixels[0]), neg_label))), dtype=np.int32)
    return X, Y

def shuffle_data(X, Y):
    perm = np.random.permutation(len(Y))
    X, Y = X[perm], Y[perm]
    return X, Y

def make_dataset(dataset,pixels, n_labeled, n_unlabeled, unlabeled_tag):
    def make_PU_dataset_from_binary_dataset(x, y, train_pos_pixels, train_neg_pixels, labeled=n_labeled, unlabeled=n_unlabeled,
                                                unlabeled_tag=unlabeled_tag):
        labels = np.unique(y)
        if labels[0] == unlabeled_tag:
            positive, negative = labels[1], labels[0]
        else:
            positive, negative = labels[0], labels[1]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)

        n_p = (Y == positive).sum()
        n_lp = labeled
        n_u = unlabeled
        if labeled + unlabeled == len(X):
            n_up = n_p - n_lp
        elif unlabeled == len(X):
            n_up = n_p
        else:
            raise ValueError("Only support |P|+|U|=|X| or |U|=|X|.")
        prior = float(n_up) / float(n_u)
        Xlp = X[Y == positive][:n_lp]
        train_lp_pos_pixels = (train_pos_pixels[0][:n_lp], train_pos_pixels[1][:n_lp])
        Xup = np.concatenate((X[Y == positive][n_lp:], Xlp), axis=0)[:n_up]
        train_up_pos_pixels = (train_pos_pixels[0][n_lp:], train_pos_pixels[1][n_lp:])
        Xun = X[Y == negative]
        X = np.asarray(np.concatenate((Xlp, Xup, Xun), axis=0), dtype=np.float32)
        Y = np.asarray(np.concatenate((np.ones(n_lp), np.full(n_u, unlabeled_tag))), dtype=np.int32)
        perm = np.random.permutation(len(Y))
        X, Y = X[perm], Y[perm]
        return X, Y, prior, train_lp_pos_pixels, train_up_pos_pixels

    def make_PN_dataset_from_binary_dataset(x, y, unlabeled_tag):
        labels = np.unique(y)
        if labels[0] == unlabeled_tag:
            positive, negative = labels[1], labels[0]
        else:
            positive, negative = labels[0], labels[1]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        n_p = (Y == positive).sum()
        n_n = (Y == negative).sum()
        Xp = X[Y == positive][:n_p]
        Xn = X[Y == negative][:n_n]
        X = np.asarray(np.concatenate((Xp, Xn)), dtype=np.float32)
        Y = np.asarray(np.concatenate((np.ones(n_p), np.full(n_n, unlabeled_tag))), dtype=np.int32)
        perm = np.random.permutation(len(Y))
        X, Y = X[perm], Y[perm]
        return X, Y

    (_trainX, _trainY), (_testX, _testY) = dataset
    (train_pos_pixels, train_neg_pixels), (test_pos_pixels, test_neg_pixels) = pixels
    trainX, trainY, prior, train_lp_pos_pixels, train_up_pos_pixels = make_PU_dataset_from_binary_dataset(_trainX, _trainY, train_pos_pixels, train_neg_pixels)
    testX, testY = make_PN_dataset_from_binary_dataset(_testX, _testY, unlabeled_tag)
    print("training:{}".format(trainX.shape))
    print("test:{}".format(testX.shape))
    return list(zip(trainX, trainY)), list(zip(testX, testY)), prior, testX, testY, trainX, trainY, train_lp_pos_pixels, train_up_pos_pixels


def get_PU_data_by_class(pos_label , neg_labels_list, train_pos_percentage, train_neg_percentage, is_random_neg):
    input_mat, target_mat = load_data()
    CLASSES = get_classwise_pixel(target_mat)
    train_pos_pixels, train_neg_pixels, test_pos_pixels, test_neg_pixels = get_test_train_PN_pixel(CLASSES, pos_label, neg_labels_list, train_pos_percentage, train_neg_percentage, is_random_neg)
    if train_pos_pixels is None or train_neg_pixels is None or test_pos_pixels is None or test_neg_pixels is None:
        return None, None, None, None, None, None, None, (train_pos_pixels, train_neg_pixels, test_pos_pixels, test_neg_pixels)
    trainX, trainY = get_PN_data(train_pos_pixels, train_neg_pixels, input_mat, 1, 0)
    testX, testY = get_PN_data(test_pos_pixels, test_neg_pixels, input_mat, 1, 0)
    # 60 % labelled out of positive samples in training
    n_labeled = (len(train_pos_pixels[0]) * 6) // 10
    n_unlabeled = len(train_pos_pixels[0]) - n_labeled + len(train_neg_pixels[0])
    unlabeled_tag = 0
    # For PU testing
    XYtrain, XYtest, prior, testX, testY, trainX, trainY, train_lp_pos_pixels, train_up_pos_pixels = make_dataset(((trainX, trainY), (testX, testY)), ((train_pos_pixels, train_neg_pixels), (test_pos_pixels, test_neg_pixels)), n_labeled,
                                                                        n_unlabeled, unlabeled_tag)

    # For PN testing will use this data if needed -- to debug
    PN_trainX, PN_trainY = shuffle_data(np.array(trainX, copy=True), np.array(trainY, copy=True))
    PN_testX, PN_testY = shuffle_data(np.array(testX, copy=True), np.array(testY, copy=True))
    PN_XYtrain, PN_XYtest = list(zip(PN_trainX, PN_trainY)), list(zip(PN_testX, PN_testY))
    return (XYtrain, XYtest, prior, testX, testY, trainX, trainY), (train_lp_pos_pixels, train_up_pos_pixels, train_neg_pixels, test_pos_pixels, test_neg_pixels)
