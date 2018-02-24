import numpy as np
import scipy.io as io
import sqlite3

def load_data():
    input_mat = io.loadmat("mldata/Indian_pines_Preprocessed_patch_3.mat")['preprocessed_img']
    target_mat = io.loadmat("mldata/Indian_pines_Preprocessed_patch_3.mat")['preprocessed_gt']
    target_mat = np.asarray(target_mat, dtype=np.int32)
    input_mat = np.asarray(input_mat, dtype=np.float32)
    return input_mat, target_mat


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

def get_train_PU_data(patch_indices, exclude_indices, target_mat, input_mat, pos_class):
    indx = np.zeros(target_mat.shape, dtype=np.bool)
    indx[patch_indices] = True
    indx[np.where(target_mat != pos_class)] = False
    train_lp_pos_pixels = np.where(indx == True)
    train_pos_X = input_mat[train_lp_pos_pixels]
    train_pos_Y = np.full(len(train_lp_pos_pixels[0]), 1, dtype=np.int32)
    indx[exclude_indices] = True
    unlb_indx = np.where(indx == False)
    unlb_Y = target_mat[unlb_indx]
    train_up_pos_pixels = (unlb_indx[0][np.where(unlb_Y == pos_class)[0]], unlb_indx[1][np.where(unlb_Y == pos_class)[0]])
    train_neg_pixels = (unlb_indx[0][np.where(unlb_Y != pos_class)[0]], unlb_indx[1][np.where(unlb_Y != pos_class)[0]])
    train_unlb_X = input_mat[unlb_indx]
    train_unlb_Y = np.full(len(unlb_indx[0]), 0, dtype=np.int32)
    train_X = np.concatenate((train_pos_X, train_unlb_X), axis=0)
    train_Y = np.concatenate((train_pos_Y, train_unlb_Y), axis=0)
    perm = np.random.permutation(len(train_Y))
    train_X, train_Y = train_X[perm], train_Y[perm]
    XYtrain = list(zip(train_X, train_Y))
    prior = 0.5
    return XYtrain, prior, train_X, train_Y, (train_lp_pos_pixels, train_up_pos_pixels, train_neg_pixels)


def get_test_PU_data(exclude_indices, target_mat, input_mat, pos_class):
    indx = np.ones(target_mat.shape, dtype=np.bool)
    indx[exclude_indices] = False
    test_indx = np.where(indx == True)
    test_Y = target_mat[test_indx]
    test_pos_pixels = (test_indx[0][np.where(test_Y == pos_class)[0]], test_indx[1][np.where(test_Y == pos_class)[0]])
    test_neg_pixels = (test_indx[0][np.where(test_Y != pos_class)[0]], test_indx[1][np.where(test_Y != pos_class)[0]])
    test_X = np.concatenate((input_mat[test_pos_pixels], input_mat[test_neg_pixels]), axis=0)
    test_Y = np.concatenate((np.full(len(test_pos_pixels[0]), 1), np.full(len(test_neg_pixels[0]), 0)), axis=0)
    perm = np.random.permutation(len(test_Y))
    test_X, test_Y = test_X[perm], test_Y[perm]
    XYtest = list(zip(test_X, test_Y))
    return XYtest, test_X, test_Y, (test_pos_pixels, test_neg_pixels)


def get_patch_points_by_class(class_indx):
    conn = sqlite3.connect('nnPU.db')
    c = conn.cursor()
    query = "SELECT patch_row_start, patch_row_end, patch_col_start, patch_col_end from PatchClass where class = ? order by creation_time desc limit 1"
    values = (class_indx,)
    c.execute(query, values)
    rows = c.fetchall()
    result = None
    if len(rows) > 0 and rows[0][0] != -1:
        result = rows[0]
    conn.commit()
    conn.close()
    return result

def get_PU_data_by_class(pos_label, exclude_indices):
    patch_end_points = get_patch_points_by_class(pos_label)
    if patch_end_points is None:
        return None, None, None, None, None, None, None, (None, None, None, None)
    print(patch_end_points)
    patch_indices = convert_patch_data_points(patch_end_points)
    input_mat, target_mat = load_data()
    XYtrain, prior, train_X, train_Y, (train_lp_pos_pixels, train_up_pos_pixels, train_neg_pixels) = get_train_PU_data(patch_indices, exclude_indices, target_mat, input_mat, pos_label)
    XYtest, test_X, test_Y, (test_pos_pixels, test_neg_pixels) = get_test_PU_data(exclude_indices, target_mat, input_mat, pos_label)
    return (XYtrain, XYtest, prior, test_X, test_Y, train_X, train_Y), (train_lp_pos_pixels, train_up_pos_pixels, train_neg_pixels, test_pos_pixels, test_neg_pixels)
