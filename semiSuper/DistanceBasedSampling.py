import numpy as np
import math
from utils import shuffle_data, get_binary_data, get_train_unlabelled_dist

def get_pos_pixels(pos_class_list, labelled_img, train_pos_percentage, cross_pos_percentage):
    pos_pixels = np.where(np.isin(labelled_img, pos_class_list) == True)
    if len(pos_pixels[0]) == 0:
        raise ValueError("no positive pixels in the image.")
    if train_pos_percentage + cross_pos_percentage > 100:
        raise ValueError(" pos_percentage of train and pos_percentage of cross validation together can't be greater than 100 ")
    n_pos_pixels = len(pos_pixels[0])
    n_train_pos_pixels = (n_pos_pixels * train_pos_percentage) // 100
    # cross validation
    n_cross_pos_pixels = (n_pos_pixels * cross_pos_percentage) // 100
    # n_train_pos_pixels = 200
    if n_train_pos_pixels == 0:
        raise ValueError("no positive pixels for training.")
    indx = np.random.permutation(len(pos_pixels[0]))
    train_lp_pixels = (pos_pixels[0][indx[:n_train_pos_pixels]], pos_pixels[1][indx[:n_train_pos_pixels]])
    # cross validation
    cross_pos_pixels = (pos_pixels[0][indx][n_train_pos_pixels: n_train_pos_pixels + n_cross_pos_pixels],
                        pos_pixels[1][indx][n_train_pos_pixels: n_train_pos_pixels + n_cross_pos_pixels])
    return train_lp_pixels, cross_pos_pixels


def get_euclidean_dist(point1, point2):
    s = math.pow((point1[1] - point2[1]), 2) + math.pow((point1[0] - point2[0]), 2)
    return s ** 0.5

def get_distance_from_positive(train_lp_pixels, cross_pos_pixels, length, width, baseline, temp):
    all_pos_pixels = (np.concatenate((train_lp_pixels[0], cross_pos_pixels[0]), axis=0), np.concatenate((train_lp_pixels[1], cross_pos_pixels[1]), axis=0))
    dist = np.zeros((length, width), dtype=np.float32)
    for i in range(length):
        for j in range(width):
            dist[i][j] = min([get_euclidean_dist((i, j), (all_pos_pixels[0][l], all_pos_pixels[1][l])) for l in range(len(all_pos_pixels[0]))])
    # print(np.mean(dist), np.std(dist), "get_distance_from_positive")
    dist = 1 / (1 + np.exp((dist - baseline)/temp))
    dist = 1 - dist
    return dist

def get_unlabelled_pixels(dist, pos_neg_ratio_in_train, pos_neg_ratio_in_cross, train_lp_pixels, cross_pos_pixels):
    n_train_pos_pixels = len(train_lp_pixels[0])
    n_cross_pos_pixels = len(cross_pos_pixels[0])
    n_unlabelled_train = int(n_train_pos_pixels // pos_neg_ratio_in_train)
    n_unlabelled_cross = int(n_cross_pos_pixels // pos_neg_ratio_in_cross)
    indx = np.zeros(dist.shape, dtype=np.bool)
    indx[train_lp_pixels] = True
    indx[cross_pos_pixels] = True
    elements = np.where(indx == False)
    if n_unlabelled_train + n_unlabelled_cross > len(elements[0]):
        raise ValueError(
            " can't sample unlabelled training and cross data from rest of the image to maintain the ratio ")
    prob = dist[elements] / np.sum(dist[elements])
    unlabelled = np.random.choice(len(elements[0]), int(n_unlabelled_train + n_unlabelled_cross), replace=False, p=prob)
    unlabelled_indx = (elements[0][unlabelled], elements[1][unlabelled])
    train_unlabelled_indx = (unlabelled_indx[0][:n_unlabelled_train], unlabelled_indx[1][:n_unlabelled_train])
    cross_unlabelled_indx = (unlabelled_indx[0][n_unlabelled_train:], unlabelled_indx[1][n_unlabelled_train:])
    return train_unlabelled_indx, cross_unlabelled_indx

def get_test_pixels(train_lp_pixels, train_unlabelled_indx, cross_pos_pixels, cross_unlabelled_indx, labelled_img, pos_class_list):
    all_pos_pixels = np.isin(labelled_img, pos_class_list)
    # not_selected = np.zeros(labelled_img.shape, dtype=np.bool)
    test_pos_pixels = np.copy(all_pos_pixels)
    test_pos_pixels[train_lp_pixels] = False
    test_pos_pixels[train_unlabelled_indx] = False
    test_pos_pixels[cross_pos_pixels] = False
    test_pos_pixels[cross_unlabelled_indx] = False
    test_pos_pixels = np.where(test_pos_pixels == True)
    test_neg_pixels = np.logical_not(all_pos_pixels)
    test_neg_pixels[train_lp_pixels] = False
    test_neg_pixels[train_unlabelled_indx] = False
    test_neg_pixels[cross_pos_pixels] = False
    test_neg_pixels[cross_unlabelled_indx] = False
    test_neg_pixels = np.where(test_neg_pixels == True)
    return test_pos_pixels, test_neg_pixels

def shuffle_test_data(X, Y, test_pos_pixels, test_neg_pixels):
    test_pixels = (np.concatenate((test_pos_pixels[0], test_neg_pixels[0]), axis=0),
                   np.concatenate((test_pos_pixels[1], test_neg_pixels[1]), axis=0))
    perm = np.random.permutation(len(Y))
    shuffled_test_pixels = (test_pixels[0][perm], test_pixels[1][perm])
    X, Y = X[perm], Y[perm]
    return X, Y, shuffled_test_pixels

def get_PN_data(pos_class_list, neg_class_list, data_img, labelled_img, clust_labelled_img, train_pos_percentage, pos_neg_ratio_in_train, cross_pos_percentage, pos_neg_ratio_in_cross, is_dist_based, baseline, temp):
    pos_class_list = list(set(pos_class_list))
    train_lp_pixels, cross_pos_pixels = get_pos_pixels(pos_class_list, labelled_img, train_pos_percentage, cross_pos_percentage)
    dist = get_distance_from_positive(train_lp_pixels, cross_pos_pixels, labelled_img.shape[0], labelled_img.shape[1], baseline, temp)
    train_unlabelled_indx, cross_unlabelled_indx = get_unlabelled_pixels(dist, pos_neg_ratio_in_train, pos_neg_ratio_in_cross, train_lp_pixels, cross_pos_pixels)
    train_up_pixels, train_un_pixels = get_train_unlabelled_dist(labelled_img, pos_class_list, train_unlabelled_indx)
    trainX, trainY = get_binary_data(train_lp_pixels, train_unlabelled_indx, data_img)
    crossX, crossY = get_binary_data(cross_pos_pixels, cross_unlabelled_indx, data_img)
    trainX, trainY = shuffle_data(trainX, trainY)
    crossX, crossY = shuffle_data(crossX, crossY)
    test_pos_pixels, test_neg_pixels = get_test_pixels(train_lp_pixels, train_unlabelled_indx, cross_pos_pixels, cross_unlabelled_indx, labelled_img, pos_class_list)
    testX, testY = get_binary_data(test_pos_pixels, test_neg_pixels, data_img)
    XYtrain = list(zip(trainX, trainY))
    prior = 0.33
    testX, testY, shuffled_test_pixels = shuffle_test_data(testX, testY, test_pos_pixels, test_neg_pixels)
    XYtest = list(zip(testX, testY))
    return (XYtrain, XYtest, prior, testX, testY, trainX, trainY, crossX, crossY), \
           (train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels, shuffled_test_pixels)

