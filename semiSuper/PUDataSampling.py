import numpy as np
import math, Config
from utils import get_binary_data, shuffle_data, get_train_unlabelled_dist
from semiSuper.sampling import get_point_wise_prob, get_pos_pixels, get_exclude_pixels

def get_unlabelled_pixels(exclude_pixels, weight, train_lp_pixels, cross_pos_pixels, gt_labelled_img, pos_class_list, neg_pos_ratio_in_unlabelled, calculate_prior):
    n_cross_pos_pixels = len(cross_pos_pixels[0])
    n_train_pos_pixels = len(train_lp_pixels[0])
    indx = np.zeros(weight.shape, dtype=np.bool)
    indx[exclude_pixels] = True
    indx[train_lp_pixels] = True
    indx[cross_pos_pixels] = True
    elements = np.where(indx == False)
    prob = weight[elements] / np.sum(weight[elements])
    unlabelled = np.random.choice(len(elements[0]), len(elements[0]), replace=False, p=prob)
    unlabelled_indx = (elements[0][unlabelled], elements[1][unlabelled])
    train_unlabelled_indx = unlabelled_indx

    train_up_pixels, train_un_pixels = get_train_unlabelled_dist(gt_labelled_img, pos_class_list, train_unlabelled_indx)

    if calculate_prior:
        prior = len(train_up_pixels[0]) / (len(train_un_pixels[0]) + len(train_up_pixels[0])) # should provide this value
        neg_pos_ratio_in_unlabelled = len(train_un_pixels[0])/(len(train_up_pixels[0]) + n_train_pos_pixels)
    else:
        prior = 1/(1 + neg_pos_ratio_in_unlabelled)
    
    n_unlabelled_cross = int(n_cross_pos_pixels * neg_pos_ratio_in_unlabelled)

    if n_unlabelled_cross > len(elements[0]):
        raise ValueError(
            " can't sample unlabelled training and cross data from rest of the image to maintain the ratio ")

    cross_unlabelled_indx = (unlabelled_indx[0][:n_unlabelled_cross], unlabelled_indx[1][:n_unlabelled_cross])
    return train_unlabelled_indx, cross_unlabelled_indx, train_up_pixels, train_un_pixels, prior

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

def get_PU_data(pos_class_list, neg_class_list, data_img, gt_labelled_img, clust_labelled_img, train_pos_percentage, 
                    neg_pos_ratio_in_unlabelled, cross_pos_percentage, is_dist_based, baseline, temp, calculate_prior = False):
    pos_class_list = list(set(pos_class_list))
    train_lp_pixels, cross_pos_pixels = get_pos_pixels(pos_class_list, gt_labelled_img, train_pos_percentage,
                                                       cross_pos_percentage)
    # probability of a point being negative
    final_weight = get_point_wise_prob(clust_labelled_img, train_lp_pixels, cross_pos_pixels, is_dist_based,  baseline, temp, is_uniform_sampling=True)
    exclude_pixels = get_exclude_pixels(pos_class_list, neg_class_list, gt_labelled_img)
    train_unlabelled_indx, cross_unlabelled_indx, train_up_pixels, train_un_pixels, prior = get_unlabelled_pixels(exclude_pixels, final_weight, train_lp_pixels, 
                                                                         cross_pos_pixels, gt_labelled_img, pos_class_list, 
                                                                         neg_pos_ratio_in_unlabelled, calculate_prior)
    
    trainX, trainY = get_binary_data(train_lp_pixels, train_unlabelled_indx, data_img)
    crossX, crossY = get_binary_data(cross_pos_pixels, cross_unlabelled_indx, data_img)
    trainX, trainY = shuffle_data(trainX, trainY)
    crossX, crossY = shuffle_data(crossX, crossY)
    test_pos_pixels, test_neg_pixels = get_test_pixels(exclude_pixels, train_lp_pixels, cross_pos_pixels, gt_labelled_img, pos_class_list)
    testX, testY = get_binary_data(test_pos_pixels, test_neg_pixels, data_img)
    XYtrain = list(zip(trainX, trainY))
    # prior = Config.default_positive_prior
    
    testX, testY, shuffled_test_pixels = shuffle_test_data(testX, testY, test_pos_pixels, test_neg_pixels)
    XYtest = list(zip(testX, testY))
    return (XYtrain, XYtest, prior, testX, testY, trainX, trainY, crossX, crossY), \
           (train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels, shuffled_test_pixels)

# a - x /a + b -x