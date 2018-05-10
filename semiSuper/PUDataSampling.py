import numpy as np
import math, Config
from utils import get_binary_data, shuffle_data, get_train_unlabelled_dist

def get_prob_cluster(clust_labelled_img):
    n_classes = np.max(clust_labelled_img) + 1
    size = clust_labelled_img.size
    clust_prob = np.zeros(n_classes, dtype=np.float32)
    for i in range(n_classes):
        clust_prob[i] = len(np.where(clust_labelled_img == i)[0]) / size
    return clust_prob

def get_clust_given_pos_prob(clust_labelled_img, train_lp_pixels, cross_pos_pixels):
    pos_sel = np.zeros(clust_labelled_img.shape, dtype=np.bool)
    pos_sel[train_lp_pixels] = True
    pos_sel[cross_pos_pixels] = True
    n_pos = len(train_lp_pixels[0]) + len(cross_pos_pixels[0])
    n_classes = np.max(clust_labelled_img) + 1
    clust_pos_prob = np.zeros(n_classes, dtype=np.float32)
    for i in range(n_classes):
        clust_pos_prob[i] = (np.logical_and(clust_labelled_img == i, pos_sel)).sum() / n_pos
    return clust_pos_prob

def get_euclidean_dist(point1, point2):
    s = math.pow((point1[1] - point2[1]), 2) + math.pow((point1[0] - point2[0]), 2)
    return s ** 0.5

def get_distance_from_positive(train_lp_pixels, cross_pos_pixels, length, width):
    all_pos_pixels = (np.concatenate((train_lp_pixels[0], cross_pos_pixels[0]), axis=0), np.concatenate((train_lp_pixels[1], cross_pos_pixels[1]), axis=0))
    dist = np.zeros((length, width), dtype=np.float32)
    for i in range(length):
        for j in range(width):
            dist[i][j] = min([get_euclidean_dist((i, j), (all_pos_pixels[0][l], all_pos_pixels[1][l])) for l in range(len(all_pos_pixels[0]))])
    return dist

def get_point_wise_prob(gt_labelled_img, clust_labelled_img, train_lp_pixels, cross_pos_pixels, is_dist_based):
    if is_dist_based:
        dist = get_distance_from_positive(train_lp_pixels, cross_pos_pixels, gt_labelled_img.shape[0], gt_labelled_img.shape[1])
    else:
        dist = np.ones(clust_labelled_img.shape, dtype=np.float32)
    clust_prob = get_prob_cluster(clust_labelled_img)
    clust_pos_prob = get_clust_given_pos_prob(clust_labelled_img, train_lp_pixels, cross_pos_pixels)
    clust_sel_prob = clust_prob / clust_pos_prob
    clust_sel_prob[np.isnan(clust_sel_prob)] = 0
    # setting the infinite value pixels to max prob + 1
    inf_pixels = clust_sel_prob == np.inf
    clust_sel_prob[inf_pixels] = -1
    max_prob = np.max(clust_sel_prob)
    clust_sel_prob[inf_pixels] = max_prob + 1
    n_classes = np.max(clust_labelled_img) + 1
    final_prob = np.zeros(clust_labelled_img.shape, dtype=np.float32)
    for i in range(n_classes):
        class_pixels = clust_labelled_img == i
        final_prob[class_pixels] = dist[class_pixels] * clust_sel_prob[i]
        # final_prob[class_pixels] += clust_sel_prob[i]
    return final_prob

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

def get_exclude_pixels(pos_class_list, neg_class_list, gt_labelled_img):
    exclude_pixels = np.logical_not(np.isin(gt_labelled_img, list(set(pos_class_list).union(set(neg_class_list)))))
    return exclude_pixels

def get_unlabelled_pixels(exclude_pixels, weight, pos_neg_ratio_in_cross, train_lp_pixels, cross_pos_pixels):
    n_cross_pos_pixels = len(cross_pos_pixels[0])
    n_unlabelled_cross = int(n_cross_pos_pixels // pos_neg_ratio_in_cross)
    indx = np.zeros(weight.shape, dtype=np.bool)
    indx[exclude_pixels] = True
    indx[train_lp_pixels] = True
    indx[cross_pos_pixels] = True
    elements = np.where(indx == False)
    if n_unlabelled_cross > len(elements[0]):
        raise ValueError(
            " can't sample unlabelled training and cross data from rest of the image to maintain the ratio ")
    prob = weight[elements] / np.sum(weight[elements])
    unlabelled = np.random.choice(len(elements[0]), len(elements[0]), replace=False, p=prob)
    unlabelled_indx = (elements[0][unlabelled], elements[1][unlabelled])
    train_unlabelled_indx = unlabelled_indx
    cross_unlabelled_indx = (unlabelled_indx[0][:n_unlabelled_cross], unlabelled_indx[1][:n_unlabelled_cross])
    return train_unlabelled_indx, cross_unlabelled_indx

def get_test_pixels(exclude_pixels, train_lp_pixels, cross_pos_pixels, labelled_img, pos_class_list):
    all_pos_pixels = np.isin(labelled_img, pos_class_list)
    test_pos_pixels = np.copy(all_pos_pixels)
    test_pos_pixels[train_lp_pixels] = False
    test_pos_pixels[cross_pos_pixels] = False
    test_pos_pixels = np.where(test_pos_pixels == True)
    test_neg_pixels = np.logical_not(all_pos_pixels)
    test_neg_pixels[train_lp_pixels] = False
    test_neg_pixels[exclude_pixels] = False
    test_neg_pixels[cross_pos_pixels] = False
    test_neg_pixels = np.where(test_neg_pixels == True)
    return test_pos_pixels, test_neg_pixels

def shuffle_test_data(X, Y, test_pos_pixels, test_neg_pixels):
    test_pixels = (np.concatenate((test_pos_pixels[0], test_neg_pixels[0]), axis=0),
                   np.concatenate((test_pos_pixels[1], test_neg_pixels[1]), axis=0))
    perm = np.random.permutation(len(Y))
    shuffled_test_pixels = (test_pixels[0][perm], test_pixels[1][perm])
    X, Y = X[perm], Y[perm]
    return X, Y, shuffled_test_pixels

def get_PU_data(pos_class_list, neg_class_list, data_img, gt_labelled_img, clust_labelled_img, train_pos_percentage, pos_neg_ratio_in_train, cross_pos_percentage, pos_neg_ratio_in_cross, is_dist_based):
    pos_class_list = list(set(pos_class_list))
    train_lp_pixels, cross_pos_pixels = get_pos_pixels(pos_class_list, gt_labelled_img, train_pos_percentage,
                                                       cross_pos_percentage)
    final_weight = get_point_wise_prob(gt_labelled_img, clust_labelled_img, train_lp_pixels, cross_pos_pixels, is_dist_based)
    exclude_pixels = get_exclude_pixels(pos_class_list, neg_class_list, gt_labelled_img)
    train_unlabelled_indx, cross_unlabelled_indx = get_unlabelled_pixels(exclude_pixels, final_weight,
                                                                         pos_neg_ratio_in_cross, train_lp_pixels,
                                                                         cross_pos_pixels)
    train_up_pixels, train_un_pixels = get_train_unlabelled_dist(gt_labelled_img, pos_class_list, train_unlabelled_indx)
    trainX, trainY = get_binary_data(train_lp_pixels, train_unlabelled_indx, data_img)
    crossX, crossY = get_binary_data(cross_pos_pixels, cross_unlabelled_indx, data_img)
    trainX, trainY = shuffle_data(trainX, trainY)
    crossX, crossY = shuffle_data(crossX, crossY)
    test_pos_pixels, test_neg_pixels = get_test_pixels(exclude_pixels, train_lp_pixels, cross_pos_pixels, gt_labelled_img, pos_class_list)
    testX, testY = get_binary_data(test_pos_pixels, test_neg_pixels, data_img)
    XYtrain = list(zip(trainX, trainY))
    prior = Config.default_positive_prior
    testX, testY, shuffled_test_pixels = shuffle_test_data(testX, testY, test_pos_pixels, test_neg_pixels)
    XYtest = list(zip(testX, testY))
    return (XYtrain, XYtest, prior, testX, testY, trainX, trainY, crossX, crossY), \
           (train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels, shuffled_test_pixels)

