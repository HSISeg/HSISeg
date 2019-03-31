import numpy as np
from  semiSuper.ClusterDistBasedSampling import get_PN_data

def get_train_test_data(pos_class_list, neg_class_list, data_img, labelled_img, clust_labelled_img, train_pos_percentage, neg_pos_ratio_in_train, cross_pos_percentage, is_dist_based, baseline, temp):
    train1_pixels = np.where(clust_labelled_img >= 0)
    train1X = data_img[train1_pixels]
    train1Y = clust_labelled_img[train1_pixels]
    (XY2train, XYtest, prior, testX, testY, train2X, train2Y, crossX, crossY), \
           (train2_lp_pixels, train2_up_pixels, train2_un_pixels, test_pos_pixels, test_neg_pixels, shuffled_test_pixels) \
    = get_PN_data(pos_class_list, neg_class_list, data_img, labelled_img, clust_labelled_img, train_pos_percentage, neg_pos_ratio_in_train, cross_pos_percentage, is_dist_based, baseline, temp)
    return (train1X, train1Y, XY2train, XYtest, prior, testX, testY, train2X, train2Y, crossX, crossY), \
           (train1_pixels, train2_lp_pixels, train2_up_pixels, train2_un_pixels, test_pos_pixels, test_neg_pixels,
            shuffled_test_pixels)