from semiSuper.ClusterDistBasedSampling import get_PN_data
from semiSuper.PN_model_train import train as train_PN
from semiSuper.DistanceBasedSampling import get_PN_data as get_PN_data_dist
import utils
import copy
import Config
import numpy as np, datetime
from algo.models import PNstats
from semiSuper.visual_results import generate_and_save_visualizations

def run_PN_on_cluster_dist_sampling(clust_labelled_img, data_img, labelled_img):
    # cross validation ratio change
    include_class_list = Config.type_1_include_class_list
    for pos_class in include_class_list:
        neg_class_list = copy.copy(list(set(include_class_list)))
        neg_class_list.remove(pos_class)
        if len(neg_class_list) > 0:
            for ratio in Config.type_1_pos_neg_ratio_in_train:
                test_name = 'cluster_dist_cross_epoch_PN_refactored'
                if not utils.check_if_test_done_models(str(pos_class), test_name, ",".join([str(i) for i in neg_class_list]), Config.data, ratio, False):
                    if clust_labelled_img is None or data_img is None or labelled_img is None:
                        data_img, labelled_img = utils.load_preprocessed_data()
                        clust_labelled_img = utils.load_clustered_img()
                    n_class = np.max(labelled_img) + 1
                    exclude_list = list(set([i for i in range(n_class)]) - set(include_class_list))
                    (XYtrain, XYtest, prior, testX, testY, trainX, trainY, crossX, crossY), \
                    (train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels, shuffled_test_pixels) = get_PN_data([pos_class], neg_class_list, data_img, labelled_img, clust_labelled_img, Config.type_1_train_pos_percentage, ratio, Config.type_1_cross_pos_percentage, ratio, True)
                    print("training", trainX.shape, "positive class", pos_class)
                    print("training split: labelled positive ->", len(train_lp_pixels[0]), "unlabelled positive ->", len(train_up_pixels[0]), "unlabelled negative ->", len(train_un_pixels[0]))
                    print("test", testX.shape)
                    model = train_PN(trainX, trainY, testX, testY)
                    # generate predicted and groundtooth image
                    exclude_pixels = utils.get_excluded_pixels(labelled_img, train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels)
                    utils.get_threshold(model, crossX, crossY)
                    gt_binary_img = utils.get_binary_gt_img(labelled_img, train_lp_pixels, train_up_pixels,
                                                            train_un_pixels, test_pos_pixels, test_neg_pixels,
                                                            exclude_pixels)
                    predicted_binary_img, predicted_output = utils.get_binary_predicted_image(labelled_img, model,
                                                                                              testX, train_lp_pixels,
                                                                                              train_up_pixels,
                                                                                              train_un_pixels,
                                                                                              shuffled_test_pixels,
                                                                                              exclude_pixels)
                    visual_result_filename = "result/" + str(test_name) + str(
                        datetime.datetime.now().timestamp() * 1000) + ".png"
                    generate_and_save_visualizations(gt_binary_img, predicted_binary_img, train_lp_pixels,
                                                     train_up_pixels,
                                                     train_un_pixels, test_pos_pixels, test_neg_pixels, \
                                                     exclude_pixels, visual_result_filename)
                    # get model stats
                    precision, recall, (tn, fp, fn, tp) = utils.get_model_stats(predicted_output, testY)
                    utils.set_model_auc(model, predicted_output, testY)
                    PNstats.objects.create(pos_class = str(pos_class), neg_class = ",".join([str(i) for i in neg_class_list]),
                        precision = precision, recall = recall, true_pos = tp, true_neg = tn, false_pos = fp, false_neg = fn, test_type = test_name,
                        exclude_class_indx = ",".join([str(i) for i in exclude_list]), no_train_pos_labelled = int(len(train_lp_pixels[0])),
                            no_train_pos_unlabelled = int(len(train_up_pixels[0])), no_train_neg_unlabelled = int(len(train_un_pixels[0])),
                            train_pos_neg_ratio = ratio, threshold = model.threshold, auc = model.auc, data_name = Config.data, visual_result_filename=visual_result_filename)


def run_PN_on_cluster_sampling(clust_labelled_img, data_img, labelled_img):
    # cross validation ratio change
    include_class_list = Config.type_1_include_class_list
    for pos_class in include_class_list:
        neg_class_list = copy.copy(list(set(include_class_list)))
        neg_class_list.remove(pos_class)
        if len(neg_class_list) > 0:
            for ratio in Config.type_1_pos_neg_ratio_in_train:
                test_name = 'cluster_cross_epoch_PN_refactored'
                if not utils.check_if_test_done_models(str(pos_class), test_name, ",".join([str(i) for i in neg_class_list]), Config.data, ratio, False):
                    if clust_labelled_img is None or data_img is None or labelled_img is None:
                        data_img, labelled_img = utils.load_preprocessed_data()
                        clust_labelled_img = utils.load_clustered_img()
                    n_class = np.max(labelled_img) + 1
                    exclude_list = list(set([i for i in range(n_class)]) - set(include_class_list))
                    (XYtrain, XYtest, prior, testX, testY, trainX, trainY, crossX, crossY), \
                    (train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels, shuffled_test_pixels) = get_PN_data([pos_class], neg_class_list, data_img, labelled_img, clust_labelled_img, Config.type_1_train_pos_percentage, ratio, Config.type_1_cross_pos_percentage, ratio, False)
                    print("training", trainX.shape)
                    print("training split: labelled positive ->", len(train_lp_pixels[0]), "unlabelled positive ->", len(train_up_pixels[0]), "unlabelled negative ->", len(train_un_pixels[0]))
                    print("test", testX.shape)
                    model = train_PN(trainX, trainY, testX, testY)
                    # generate predicted and groundtooth image
                    exclude_pixels = utils.get_excluded_pixels(labelled_img, train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels)
                    utils.get_threshold(model, crossX, crossY)
                    gt_binary_img = utils.get_binary_gt_img(labelled_img, train_lp_pixels, train_up_pixels,
                                                            train_un_pixels, test_pos_pixels, test_neg_pixels,
                                                            exclude_pixels)
                    predicted_binary_img, predicted_output = utils.get_binary_predicted_image(labelled_img, model,
                                                                                              testX, train_lp_pixels,
                                                                                              train_up_pixels,
                                                                                              train_un_pixels,
                                                                                              shuffled_test_pixels,
                                                                                              exclude_pixels)
                    visual_result_filename = "result/" + str(test_name) + str(
                        datetime.datetime.now().timestamp() * 1000) + ".png"
                    generate_and_save_visualizations(gt_binary_img, predicted_binary_img, train_lp_pixels,
                                                     train_up_pixels,
                                                     train_un_pixels, test_pos_pixels, test_neg_pixels, \
                                                     exclude_pixels, visual_result_filename)
                    predicted_img, predicted_output = utils.get_binary_predicted_image(labelled_img, model, testX, train_lp_pixels, train_up_pixels, train_un_pixels, shuffled_test_pixels, exclude_pixels)

                    # get model stats
                    precision, recall, (tn, fp, fn, tp) = utils.get_model_stats(predicted_output, testY)
                    utils.set_model_auc(model, predicted_output, testY)
                    PNstats.objects.create(pos_class = str(pos_class), neg_class = ",".join([str(i) for i in neg_class_list]),
                        precision = precision, recall = recall, true_pos = tp, true_neg = tn, false_pos = fp, false_neg = fn, test_type = test_name,
                        exclude_class_indx = ",".join([str(i) for i in exclude_list]), no_train_pos_labelled = int(len(train_lp_pixels[0])),
                            no_train_pos_unlabelled = int(len(train_up_pixels[0])), no_train_neg_unlabelled = int(len(train_un_pixels[0])),
                            train_pos_neg_ratio = ratio, threshold = model.threshold, auc = model.auc, data_name = Config.data, visual_result_filename=visual_result_filename)



def run_PN_on_dist_sampling(clust_labelled_img, data_img, labelled_img):
    # cross validation ratio change
    include_class_list = Config.type_1_include_class_list
    for pos_class in include_class_list:
        neg_class_list = copy.copy(list(set(include_class_list)))
        neg_class_list.remove(pos_class)
        if len(neg_class_list) > 0:
            for ratio in Config.type_1_pos_neg_ratio_in_train:
                test_name = 'dist_cross_epoch_PN_refactored'
                if not utils.check_if_test_done_models(str(pos_class), test_name, ",".join([str(i) for i in neg_class_list]), Config.data, ratio, False):
                    if data_img is None or labelled_img is None:
                        data_img, labelled_img = utils.load_preprocessed_data()
                    n_class = np.max(labelled_img) + 1
                    exclude_list = list(set([i for i in range(n_class)]) - set(include_class_list))
                    (XYtrain, XYtest, prior, testX, testY, trainX, trainY, crossX, crossY), \
                    (train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels, shuffled_test_pixels) = get_PN_data_dist([pos_class], neg_class_list, data_img, labelled_img, None, Config.type_1_train_pos_percentage, ratio, Config.type_1_cross_pos_percentage, ratio, True)
                    print("training", trainX.shape)
                    print("training split: labelled positive ->", len(train_lp_pixels[0]), "unlabelled positive ->", len(train_up_pixels[0]), "unlabelled negative ->", len(train_un_pixels[0]))
                    print("test", testX.shape)
                    model = train_PN(trainX, trainY, testX, testY)
                    # generate predicted and groundtooth image
                    exclude_pixels = utils.get_excluded_pixels(labelled_img, train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels)
                    utils.get_threshold(model, crossX, crossY)
                    gt_binary_img = utils.get_binary_gt_img(labelled_img, train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels, exclude_pixels)
                    predicted_binary_img, predicted_output = utils.get_binary_predicted_image(labelled_img, model, testX, train_lp_pixels, train_up_pixels, train_un_pixels, shuffled_test_pixels, exclude_pixels)
                    visual_result_filename = "result/" + str(test_name) + str(datetime.datetime.now().timestamp() * 1000) + ".png"
                    generate_and_save_visualizations(gt_binary_img, predicted_binary_img, train_lp_pixels, train_up_pixels,
                                                     train_un_pixels, test_pos_pixels, test_neg_pixels, \
                                                     exclude_pixels, visual_result_filename)
                    # get model stats
                    precision, recall, (tn, fp, fn, tp) = utils.get_model_stats(predicted_output, testY)
                    utils.set_model_auc(model, predicted_output, testY)
                    PNstats.objects.create(pos_class = str(pos_class), neg_class = ",".join([str(i) for i in neg_class_list]),
                        precision = precision, recall = recall, true_pos = tp, true_neg = tn, false_pos = fp, false_neg = fn, test_type = test_name,
                        exclude_class_indx = ",".join([str(i) for i in exclude_list]), no_train_pos_labelled = int(len(train_lp_pixels[0])),
                            no_train_pos_unlabelled = int(len(train_up_pixels[0])), no_train_neg_unlabelled = int(len(train_un_pixels[0])),
                            train_pos_neg_ratio = ratio, threshold = model.threshold, auc = model.auc, data_name = Config.data, visual_result_filename=visual_result_filename)

if __name__ == '__main__':
    run_cluster_dist(None, None, None)
