from type_2_test_train import  get_PU_data_by_class as get_type2_data
from train import get_PU_model
from visual_results import get_visual_results
import numpy as np
import scipy.io as io
import copy
import pickle

train_pos_percentage = 30
train_neg_percentage = 30
is_random_neg = False
gpu = -1
unlabeled_tag = 0
n_class = 17

def load_data():
    input_mat = io.loadmat("mldata/Indian_pines_Preprocessed_patch_3.mat")['preprocessed_img']
    target_mat = io.loadmat("mldata/Indian_pines_Preprocessed_patch_3.mat")['preprocessed_gt']
    target_mat = np.asarray(target_mat, dtype=np.int32)
    input_mat = np.asarray(input_mat, dtype=np.float32)
    return input_mat, target_mat

def get_indices_from_list(target_mat, indices_list):
    indx = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    for i in indices_list:
        indx_i = np.where(target_mat == i)
        indx = (np.concatenate((indx[0], indx_i[0]), axis=0), np.concatenate((indx[1], indx_i[1]), axis=0))
    return indx

def run():
    # neg_labels_list = [8]
    neg_labels_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]
    for i in range(n_class):
        neg_labels_list_i = copy.copy(neg_labels_list)
        if i in neg_labels_list_i:
            neg_labels_list_i.remove(i)
        if len(neg_labels_list_i) > 0 and i == 12:
            exclude_list = []
            for j in range(n_class):
                if j !=i and j not in neg_labels_list:
                    exclude_list.append(j)
            input_mat, target_mat = load_data()
            exclude_indices = get_indices_from_list(target_mat, exclude_list)
            (XYtrain, XYtest, prior, testX, testY, trainX, trainY), \
            (train_lp_pos_pixels, train_up_pos_pixels, train_neg_pixels, test_pos_pixels, test_neg_pixels) = get_type2_data(i , exclude_indices)
            print("training", trainX.shape)
            print("training split: labelled positive ->", len(train_lp_pos_pixels[0]), "unlabelled positive ->", len(train_up_pos_pixels[0]), "unlabelled negative ->", len(train_neg_pixels[0]))
            print("test", testX.shape)
            model = get_PU_model(XYtrain, XYtest, prior, unlabeled_tag, gpu)
            gt_img, predicted_img, train_lp_pos_pixels, train_up_pos_pixels, train_neg_pixels, test_pos_pixels, test_neg_pixels, exclude_pixels = get_visual_results(target_mat, model, input_mat, train_lp_pos_pixels, train_up_pos_pixels, train_neg_pixels,
                           test_pos_pixels, test_neg_pixels, 'type_2', i, neg_labels_list_i, exclude_list)
            pickle_data = {}
            pickle_data['gt_img'] = gt_img
            pickle_data['predicted_img'] = predicted_img
            pickle_data['train_lp_pos_pixels'] = train_lp_pos_pixels
            pickle_data['train_up_pos_pixels'] = train_up_pos_pixels
            pickle_data['train_un_pixels'] = train_neg_pixels
            pickle_data['test_pos_pixels'] = test_pos_pixels
            pickle_data['test_neg_pixels'] = test_neg_pixels
            pickle_data['exclude_pixels'] = exclude_pixels
            with open("result/type_2_test_" + str(i) + "_pos.pickle", "wb") as fp:
                pickle.dump(pickle_data, fp, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    run()


