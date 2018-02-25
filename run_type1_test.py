from type_1_test_train import  get_PU_data_by_class as get_type1_data
from train import get_PU_model
from visual_data_format import gen_visual_results_data, save_data_in_PUstats, check_if_test_done
from visual_results import generate_and_save_visualizations
import numpy as np
import scipy.io as io
import copy
import pickle, datetime

# percentage of pixels from total pixels of positive class that will be included in training, 60% of positive pixels will be labelled
train_pos_percentage = 30
# percentage of pixels from total pixels of negative class (include_class_list - positive_class) that will be included in training
train_neg_percentage = 30
# class label included in this test, negative class list = list(set(include_class_list) - set(positive_class))
# Note: Class labels should be present in groundtooth image
# include_class_list = [8, 12]
include_class_list =  [2, 3, 5, 6, 8, 10, 11, 12, 14]
""" 
boolean value to indicate the way of selection for negative unlabelled training data,
If set to true then all the data from negative class list are accumulated and randomly train_neg_percentage of the total data is selected
In the above case it might happen that training data might not have any pixel for class k in negative class list 
If set to false then train_neg_percentage pixels of data is randomly selected for each class in negative class list
The above ensures that training data has some pixels of each class in negative class list 
"""
is_random_neg = False
# Zero-origin GPU ID (negative value indicates CPU)
gpu = -1


def load_data():
    input_mat = io.loadmat("mldata/Indian_pines_Preprocessed_patch_3.mat")['preprocessed_img']
    target_mat = io.loadmat("mldata/Indian_pines_Preprocessed_patch_3.mat")['preprocessed_gt']
    target_mat = np.asarray(target_mat, dtype=np.int32)
    input_mat = np.asarray(input_mat, dtype=np.float32)
    return input_mat, target_mat


def run():
    unlabeled_tag = 0
    n_class = 17
    for pos_class in include_class_list:
        neg_labels_list = copy.copy(list(set(include_class_list)))
        neg_labels_list.remove(pos_class)
        if len(neg_labels_list) > 0:
            exclude_list = list(set([i for i in range(n_class)]) - set(include_class_list))
            if check_if_test_done(pos_class, 'type_1', neg_labels_list):
                (XYtrain, XYtest, prior, testX, testY, trainX, trainY), \
                (train_lp_pos_pixels, train_up_pos_pixels, train_neg_pixels, test_pos_pixels, test_neg_pixels) = get_type1_data(pos_class , neg_labels_list, train_pos_percentage, train_neg_percentage, is_random_neg)
                print("training", trainX.shape)
                print("training split: labelled positive ->", len(train_lp_pos_pixels[0]), "unlabelled positive ->", len(train_up_pos_pixels[0]), "unlabelled negative ->", len(train_neg_pixels[0]))
                print("test", testX.shape)
                model = get_PU_model(XYtrain, XYtest, prior, unlabeled_tag, gpu)
                input_mat, target_mat = load_data()
                gt_img, predicted_img, train_lp_pos_pixels, train_up_pos_pixels, train_neg_pixels, \
                test_pos_pixels, test_neg_pixels, exclude_pixels, (precision, recall, tp, tn, fp, fn ) = gen_visual_results_data(target_mat, model, input_mat,\
                                                                                                                                 train_lp_pos_pixels, train_up_pos_pixels, train_neg_pixels,
                               test_pos_pixels, test_neg_pixels)
                visual_result_filename = "result/type_1_test_" + str(pos_class) + "_pos_"+ str(datetime.datetime.now().timestamp() * 1000) +".png"
                generate_and_save_visualizations(gt_img, predicted_img, train_lp_pos_pixels, train_up_pos_pixels, train_neg_pixels, test_pos_pixels, test_neg_pixels, \
                                                 exclude_pixels, visual_result_filename)
                save_data_in_PUstats((
                                     str(pos_class), ",".join([str(i) for i in neg_labels_list]), precision, recall, tp,
                                     tn, fp, fn, 'type_1', ",".join([str(i) for i in exclude_list]), int(len(train_lp_pos_pixels[0])),
                                     int(len(train_up_pos_pixels[0])), int(len(train_neg_pixels[0])), visual_result_filename))
                # pickle_data = {}
                # pickle_data['gt_img'] = gt_img
                # pickle_data['predicted_img'] = predicted_img
                # pickle_data['train_lp_pos_pixels'] = train_lp_pos_pixels
                # pickle_data['train_up_pos_pixels'] = train_up_pos_pixels
                # pickle_data['train_un_pixels'] = train_neg_pixels
                # pickle_data['test_pos_pixels'] = test_pos_pixels
                # pickle_data['test_neg_pixels'] = test_neg_pixels
                # pickle_data['exclude_pixels'] = exclude_pixels
                # with open("result/type_1_test_" + str(i) + "_pos.pickle", "wb") as fp:
                #     pickle.dump(pickle_data, fp, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    run()


