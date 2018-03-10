import numpy as np
import utils

def convert_patch_data_points(patch_end_points):
    if patch_end_points is None:
        return None
    patch_indices = (np.array([], dtype='int64'),np.array([],dtype='int64'))
    if patch_end_points[0] != -1 and patch_end_points[1] != -1 and patch_end_points[2] != -1 and patch_end_points[3] != -1:
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

def get_train_PU_data(patch_indices, data_img, labelled_img, pos_class_list, neg_class_list):
    # selecting positive indices in patch
    indx = np.zeros(labelled_img.shape, dtype=np.bool)
    indx[patch_indices] = True
    # all the negative index
    neg_indx = np.isin(labelled_img, neg_class_list)
    if len(neg_indx[0]):
        raise ValueError("no negative pixels for training in image")
    # this will also set negative index in patch
    indx[neg_indx] = False
    # only positive index in patch
    train_lp_pixels = np.where(indx == True)
    train_pos_X = data_img[train_lp_pixels]
    train_pos_Y = np.full(len(train_lp_pixels[0]), 1, dtype=np.int32)

    # training unlabelled positive pixels
    indx = np.zeros(labelled_img.shape, dtype=np.bool)
    indx[np.isin(labelled_img, pos_class_list)] = True
    indx[train_lp_pixels] = False

    train_up_pixels = np.where(indx == True)
    # training unlabelled negative pixels
    train_un_pixels = neg_indx
    # unlabelled pixels = unlabelled positive pixels + unlabelled negative pixels
    unlb_indx = (np.concatenate((train_up_pixels[0], train_un_pixels[0]), axis=0), np.concatenate((train_up_pixels[1], train_un_pixels[1]), axis=0))

    train_unlb_X = data_img[unlb_indx]
    train_unlb_Y = np.full(len(unlb_indx[0]), 0, dtype=np.int32)
    train_X = np.concatenate((train_pos_X, train_unlb_X), axis=0)
    train_Y = np.concatenate((train_pos_Y, train_unlb_Y), axis=0)
    perm = np.random.permutation(len(train_Y))
    train_X, train_Y = train_X[perm], train_Y[perm]
    XYtrain = list(zip(train_X, train_Y))
    prior = float(len(train_up_pixels[0])) / (float(len(train_up_pixels[0])) + float(len(train_un_pixels[0])))
    if prior == 0:
        prior = 0.5
    print("prior", prior)
    return XYtrain, prior, train_X, train_Y, (train_lp_pixels, train_up_pixels, train_un_pixels)


def get_test_PU_data(train_lp_pixels, train_up_pixels, train_un_pixels, data_img):
    test_pos_pixels = (np.concatenate((train_lp_pixels[0], train_up_pixels[0]), axis=0), np.concatenate((train_lp_pixels[1], train_up_pixels[1]), axis=0))
    test_neg_pixels = train_un_pixels
    test_X = np.concatenate((data_img[test_pos_pixels], data_img[test_neg_pixels]), axis=0)
    test_Y = np.concatenate((np.full(len(test_pos_pixels[0]), 1), np.full(len(test_neg_pixels[0]), 0)), axis=0)
    test_pixels = (np.concatenate((test_pos_pixels[0], test_neg_pixels[0]), axis=0),
                    np.concatenate((test_pos_pixels[1], test_neg_pixels[1]), axis=0))
    perm = np.random.permutation(len(test_Y))
    test_X, test_Y = test_X[perm], test_Y[perm]
    shuffled_test_pixels = (test_pixels[0][perm], test_pixels[1][perm])
    XYtest = list(zip(test_X, test_Y))
    return XYtest, test_X, test_Y, (test_pos_pixels, test_neg_pixels, shuffled_test_pixels)




def get_PU_data(pos_class_list, neg_class_list, data_img, labelled_img):
                # (pos_label, exclude_indices):
    # exclude_indices = utils.get_indices_from_list(labelled_img, exclude_list)
    pos_class_list = list(set(pos_class_list))
    neg_class_list = list(set(neg_class_list))
    patch_end_points_list = []
    for pos_class in pos_class_list:
        patch_end_points = utils.get_patch_points_by_class(pos_class)
        if patch_end_points is None:
            raise ValueError("no positive pixels for training in image for class " + str(pos_class) )
        patch_end_points_list.append(patch_end_points)
    patch_indices = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    for patch_end_points in patch_end_points_list:
        temp_patch_indices = convert_patch_data_points(patch_end_points)
        patch_indices = (np.concatenate((patch_indices[0], temp_patch_indices[0]), axis=0), np.concatenate((patch_indices[0], temp_patch_indices[0]), axis=0))
    XYtrain, prior, train_X, train_Y, (train_lp_pixels, train_up_pixels, train_un_pixels) = get_train_PU_data(patch_indices, data_img, labelled_img, pos_class_list, neg_class_list)
    XYtest, test_X, test_Y, (test_pos_pixels, test_neg_pixels, shuffled_test_pixels) = get_test_PU_data(train_lp_pixels, train_up_pixels, train_un_pixels, data_img)
    return (XYtrain, XYtest, prior, test_X, test_Y, train_X, train_Y), (train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels, shuffled_test_pixels)
