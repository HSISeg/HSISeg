from ClusterBasedSampling import get_PU_data
from PN_learning import train as train_PN
import utils
from visual_results import generate_and_save_visualizations
import copy
import Config
import datetime
import numpy as np

def run():
    # cross validation ratio change
    ratio = 0.33
    include_class_list = Config.type_1_include_class_list
    for pos_class in include_class_list:
        neg_class_list = copy.copy(list(set(include_class_list)))
        neg_class_list.remove(pos_class)
        if len(neg_class_list) > 0:
            test_name = 'cluster_exclude_PN_'+str(ratio)
            if not utils.check_if_test_done(pos_class, test_name, ",".join([str(i) for i in neg_class_list])):
                data_img, labelled_img = utils.load_preprocessed_data()
                clust_labelled_img = utils.load_clustered_img()
                n_class = np.max(labelled_img) + 1
                exclude_list = list(set([i for i in range(n_class)]) - set(include_class_list))
                (XYtrain, XYtest, prior, testX, testY, trainX, trainY, crossX, crossY), \
                (train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels, shuffled_test_pixels) = get_PU_data([pos_class], neg_class_list, data_img, labelled_img, clust_labelled_img, Config.type_1_train_pos_percentage, ratio, Config.type_1_cross_pos_percentage, ratio)
                print("training", trainX.shape)
                print("training split: labelled positive ->", len(train_lp_pixels[0]), "unlabelled positive ->", len(train_up_pixels[0]), "unlabelled negative ->", len(train_un_pixels[0]))
                print("test", testX.shape)
                model = train_PN(trainX, trainY, testX, testY)
                # generate predicted and groundtooth image
                exclude_pixels = utils.get_excluded_pixels(labelled_img, train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels)
                gt_img = utils.get_binary_gt_img(labelled_img, train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels, exclude_pixels)
                threshold = utils.get_threshold(model, crossX, crossY)
                print("threshold ->", threshold)
                predicted_img, predicted_output = utils.get_binary_predicted_image(labelled_img, model, testX, train_lp_pixels, train_up_pixels, train_un_pixels, shuffled_test_pixels, exclude_pixels)

                # get model stats
                precision, recall, (tn, fp, fn, tp) = utils.get_model_stats(predicted_output, testY)
                utils.set_model_auc(model, predicted_output, testY)
                visual_result_filename = "result/" + str(test_name) + "_" + str(pos_class) + "_pos_"+ str(datetime.datetime.now().timestamp() * 1000) +".png"
                generate_and_save_visualizations(gt_img, predicted_img, train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels, \
                                                 exclude_pixels, visual_result_filename)
                utils.save_data_in_PUstats((
                                     str(pos_class), ",".join([str(i) for i in neg_class_list]), precision, recall, tp,
                                     tn, fp, fn, test_name, ",".join([str(i) for i in exclude_list]), int(len(train_lp_pixels[0])),
                                     int(len(train_up_pixels[0])), int(len(train_un_pixels[0])), visual_result_filename, ratio, model.threshold, model.auc))


if __name__ == '__main__':
    run()
