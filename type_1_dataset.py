import numpy as np
import Config
from utils import shuffle_data

def get_test_train_pixel(pos_class_list , neg_class_list, labelled_img, pos_neg_ratio_in_train, cross_pos_percentage, pos_neg_ratio_in_cross):
    pos_pixels = np.where(np.isin(labelled_img, pos_class_list) == True)
    neg_pixels = np.where(np.isin(labelled_img, neg_class_list) == True)
    if len(pos_pixels[0]) == 0 or len(neg_pixels[0]) == 0:
        raise ValueError("no positive pixels or negative pixels in the image.")
    if Config.type_1_train_pos_percentage + cross_pos_percentage > 100:
        raise ValueError(" pos_percentage of train and pos_percentage of cross validation together can't be greater than 100 ")
    n_pos_pixels = len(pos_pixels[0])
    n_train_pos_pixels = (n_pos_pixels * Config.type_1_train_pos_percentage) // 100
    # cross validation
    n_cross_pos_pixels = (n_pos_pixels * cross_pos_percentage) // 100
    # n_train_pos_pixels = 200
    if n_train_pos_pixels == 0:
        raise ValueError("no positive pixels for training.")
    neg_pixels = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    indx = np.random.permutation(len(pos_pixels[0]))
    train_pos_pixels = (pos_pixels[0][indx[:n_train_pos_pixels]], pos_pixels[1][indx[:n_train_pos_pixels]])
    # cross validation
    cross_pos_pixels = (pos_pixels[0][indx][n_train_pos_pixels: n_train_pos_pixels + n_cross_pos_pixels], pos_pixels[1][indx][n_train_pos_pixels: n_train_pos_pixels + n_cross_pos_pixels])
    test_pos_pixels = (pos_pixels[0][indx[n_train_pos_pixels:]], pos_pixels[1][indx[n_train_pos_pixels:]])

    if Config.is_random_neg:
        # n_train_neg_pixels = (len(neg_pixels[0]) * Config.type_1_train_neg_percentage) // 100
        n_train_neg_pixels = int(float(n_train_pos_pixels) / float(pos_neg_ratio_in_train))
        # cross validation
        n_cross_neg_pixels = int(float(n_cross_pos_pixels)/ float(pos_neg_ratio_in_cross))
        if n_train_neg_pixels + n_cross_neg_pixels > len(neg_pixels[0]):
            n_cross_neg_pixels = len(neg_pixels[0]) - n_train_neg_pixels
        if n_cross_neg_pixels < 0:
            n_cross_neg_pixels = 0
        if n_train_neg_pixels == 0:
            raise ValueError("no negative pixels for training.")
        indx = np.random.permutation(len(neg_pixels[0]))
        train_neg_pixels = (neg_pixels[0][indx[:n_train_neg_pixels]], neg_pixels[1][indx[:n_train_neg_pixels]])
        test_neg_pixels = (neg_pixels[0][indx[n_train_neg_pixels:]], neg_pixels[1][indx[n_train_neg_pixels:]])
        # cross validation
        cross_neg_pixels = (neg_pixels[0][indx[n_train_neg_pixels: n_train_neg_pixels + n_cross_neg_pixels]], neg_pixels[1][indx[n_train_neg_pixels: n_train_neg_pixels + n_cross_neg_pixels]])
    # If train_neg_percentage no of pixels taken for each class
    else:
        test_neg_pixels = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        train_neg_pixels = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        # n_train_neg_pixels = (len(neg_pixels[0]) * Config.type_1_train_neg_percentage) // 100

        n_train_neg_pixels = (int(float(n_train_pos_pixels) / float(pos_neg_ratio_in_train))) // len(neg_class_list)
        # cross validation
        n_cross_neg_pixels = (int(float(n_cross_pos_pixels) / float(pos_neg_ratio_in_cross))) // len(neg_class_list)

        for i in neg_class_list:
            neg_pixels = np.where(labelled_img == i)


            # n_train_neg_pixels = n_train_pos_pixels // len(neg_class_list)
            # n_train_neg_pixels = 200
            if n_train_neg_pixels > 0:
                indx = np.random.permutation(len(neg_pixels[0]))
                n_cross_neg_pixels_by_class = n_cross_neg_pixels
                if n_train_neg_pixels + n_cross_neg_pixels_by_class > len(neg_pixels[0]):
                    n_cross_neg_pixels_by_class = len(neg_pixels[0]) - n_train_neg_pixels
                if n_cross_neg_pixels_by_class < 0:
                    n_cross_neg_pixels_by_class = 0
                train_neg_pixels = (
                    np.concatenate((train_neg_pixels[0], neg_pixels[0][indx[:n_train_neg_pixels]])),
                    np.concatenate((train_neg_pixels[1], neg_pixels[1][indx[:n_train_neg_pixels]])))

                test_neg_pixels = (
                    np.concatenate((test_neg_pixels[0], neg_pixels[0][indx[n_train_neg_pixels:]])),
                    np.concatenate((test_neg_pixels[1], neg_pixels[1][indx[n_train_neg_pixels:]])))

                cross_neg_pixels = (
                    np.concatenate((test_neg_pixels[0], neg_pixels[0][indx[n_train_neg_pixels: n_train_neg_pixels + n_cross_neg_pixels_by_class]])),
                    np.concatenate((test_neg_pixels[1], neg_pixels[1][indx[n_train_neg_pixels: n_train_neg_pixels + n_cross_neg_pixels_by_class]])))

        if len(train_neg_pixels[0]) == 0 or len(test_neg_pixels[0]) == 0:
            raise ValueError("no negative pixels for training and/or testing.")
    return train_pos_pixels, train_neg_pixels, test_pos_pixels, test_neg_pixels, cross_pos_pixels, cross_neg_pixels

def get_binary_data(pos_pixels, neg_pixels, data_img):
    X = np.concatenate((data_img[pos_pixels], data_img[neg_pixels]))
    Y = np.asarray(np.concatenate((np.full(len(pos_pixels[0]), 1), np.full(len(neg_pixels[0]), 0))), dtype=np.int32)
    return X, Y



def make_dataset(dataset, pixels, n_labeled, n_unlabeled, unlabeled_tag):
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
        train_lp_pixels = (train_pos_pixels[0][:n_lp], train_pos_pixels[1][:n_lp])
        Xup = np.concatenate((X[Y == positive][n_lp:], Xlp), axis=0)[:n_up]
        train_up_pixels = (train_pos_pixels[0][n_lp:], train_pos_pixels[1][n_lp:])
        Xun = X[Y == negative]
        X = np.asarray(np.concatenate((Xlp, Xup, Xun), axis=0), dtype=np.float32)
        Y = np.asarray(np.concatenate((np.ones(n_lp), np.full(n_u, unlabeled_tag))), dtype=np.int32)
        perm = np.random.permutation(len(Y))
        X, Y = X[perm], Y[perm]
        return X, Y, prior, train_lp_pixels, train_up_pixels

    def make_PN_dataset_from_binary_dataset(x, y, unlabeled_tag, test_pos_pixels, test_neg_pixels):
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
        test_pixels = (np.concatenate((test_pos_pixels[0], test_neg_pixels[0]), axis=0), np.concatenate((test_pos_pixels[1], test_neg_pixels[1]), axis=0))
        perm = np.random.permutation(len(Y))
        shuffled_test_pixels = (test_pixels[0][perm], test_pixels[1][perm])
        X, Y = X[perm], Y[perm]
        return X, Y, shuffled_test_pixels

    (_trainX, _trainY), (_testX, _testY) = dataset
    (train_pos_pixels, train_neg_pixels), (test_pos_pixels, test_neg_pixels) = pixels
    trainX, trainY, prior, train_lp_pixels, train_up_pixels = make_PU_dataset_from_binary_dataset(_trainX, _trainY, train_pos_pixels, train_neg_pixels)
    testX, testY, shuffled_test_pixels = make_PN_dataset_from_binary_dataset(_testX, _testY, unlabeled_tag, test_pos_pixels, test_neg_pixels)
    return list(zip(trainX, trainY)), list(zip(testX, testY)), prior, testX, testY, trainX, trainY, train_lp_pixels, train_up_pixels, shuffled_test_pixels


def get_PU_data(pos_class_list , neg_class_list, data_img, labelled_img, pos_neg_ratio_in_train, cross_pos_percentage, pos_neg_ratio_in_cross):
    pos_class_list = list(set(pos_class_list))
    neg_class_list = list(set(neg_class_list))
    train_pos_pixels, train_neg_pixels, test_pos_pixels, test_neg_pixels, cross_pos_pixels, cross_neg_pixels = get_test_train_pixel(pos_class_list, neg_class_list, labelled_img, pos_neg_ratio_in_train, cross_pos_percentage, pos_neg_ratio_in_cross)
    trainX, trainY = get_binary_data(train_pos_pixels, train_neg_pixels, data_img)
    testX, testY = get_binary_data(test_pos_pixels, test_neg_pixels, data_img)
    crossX, crossY = get_binary_data(cross_pos_pixels, cross_neg_pixels, data_img)
    crossX, crossY = shuffle_data(crossX, crossY)
    # 60 % labelled out of positive samples in training
    n_labeled = (len(train_pos_pixels[0]) * Config.type_1_train_pos_labelled_percentage) // 100
    n_unlabeled = len(train_pos_pixels[0]) - n_labeled + len(train_neg_pixels[0])
    # For PU testing
    XYtrain, XYtest, prior, testX, testY, trainX, trainY, train_lp_pixels, train_up_pixels, shuffled_test_pixels = make_dataset(((trainX, trainY), (testX, testY)), ((train_pos_pixels, train_neg_pixels), (test_pos_pixels, test_neg_pixels)), n_labeled,
                                                                        n_unlabeled, Config.unlabeled_tag)
    train_un_pixels = train_neg_pixels

    # For PN testing will use this data if needed -- to debug
    # PN_trainX, PN_trainY = shuffle_data(np.array(trainX, copy=True), np.array(trainY, copy=True))
    # PN_testX, PN_testY = shuffle_data(np.array(testX, copy=True), np.array(testY, copy=True))
    # PN_XYtrain, PN_XYtest = list(zip(PN_trainX, PN_trainY)), list(zip(PN_testX, PN_testY))
    # ((!IS_BLANK_OR_NULL(partnerTrackingId) & & EQUALS(eventDesc, "arrived at usps delivery unit")) | | (
    #     !IS_BLANK_OR_NULL(partnerTrackingId) & & EQUALS(eventCode, "AUSPS")) | | (!IS_BLANK_OR_NULL(partnerTrackingId) & & EQUALS(
    #     eventCode, "UPROC")) | | !IS_BLANK_OR_NULL(partnerTrackingId))
    return (XYtrain, XYtest, prior, testX, testY, trainX, trainY, crossX, crossY), (train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels, shuffled_test_pixels)
