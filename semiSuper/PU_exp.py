import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "HSISeg.settings")
django.setup()
from semiSuper.PUDataSampling import get_PU_data
from semiSuper.PU_train import train as train_PU
import utils
from visual_results import generate_and_save_visualizations
import copy
import Config
import datetime
from algo.models import PUstats
import numpy as np

def run(clust_labelled_img, data_img, labelled_img):
    # cross validation ratio change
    include_class_list = Config.type_1_include_class_list
    for pos_class in include_class_list:
        neg_class_list = copy.copy(list(set(include_class_list)))
        neg_class_list.remove(pos_class)
        if len(neg_class_list) > 0:
            for ratio in Config.type_1_pos_neg_ratio_in_train:
                test_name = Config.data + 'cluster_dist_PU_'+str(ratio)
                if not utils.check_if_test_done_models(str(pos_class), test_name, ",".join([str(i) for i in neg_class_list]), Config.data, ratio, True):
                    if clust_labelled_img is None or data_img is None or labelled_img is None:
                        data_img, labelled_img = utils.load_preprocessed_data()
                        clust_labelled_img = utils.load_clustered_img()

                    n_class = np.max(labelled_img) + 1
                    exclude_list = list(set([i for i in range(n_class)]) - set(include_class_list))
                    (XYtrain, XYtest, prior, testX, testY, trainX, trainY, crossX, crossY), \
                    (train_lp_pixels, train_up_pixels, train_un_pixels, test_pos_pixels, test_neg_pixels, shuffled_test_pixels) = get_PU_data([pos_class], neg_class_list, data_img, labelled_img, clust_labelled_img, Config.type_1_train_pos_percentage, ratio, Config.type_1_cross_pos_percentage, ratio, True)
                    print("training", trainX.shape)
                    print("training split: labelled positive ->", len(train_lp_pixels[0]), "unlabelled positive ->", len(train_up_pixels[0]), "unlabelled negative ->", len(train_un_pixels[0]))
                    print("test", testX.shape)
                    model = train_PU(XYtrain, XYtest, prior)

                    # generate predicted and groundtooth image

                    exclude_pixels = utils.get_excluded_pixels(labelled_img, train_lp_pixels, train_up_pixels, train_un_pixels,
                                                           test_pos_pixels, test_neg_pixels)
                    utils.get_threshold(model, crossX, crossY)
                    gt_binary_img = utils.get_binary_gt_img(labelled_img, train_lp_pixels, train_up_pixels, train_un_pixels,
                                                        test_pos_pixels, test_neg_pixels, exclude_pixels)
                    predicted_binary_img, predicted_output = utils.get_binary_predicted_image(labelled_img, model, testX,
                                                                                          train_lp_pixels, train_up_pixels,
                                                                                          train_un_pixels,
                                                                                          shuffled_test_pixels,
                                                                                          exclude_pixels)
                    visual_result_filename = "result/" + str(test_name) + str(datetime.datetime.now().timestamp() * 1000) + ".png"
                    generate_and_save_visualizations(gt_binary_img, predicted_binary_img, train_lp_pixels, train_up_pixels,
                                                 train_un_pixels, test_pos_pixels, test_neg_pixels, \
                                                 exclude_pixels, visual_result_filename)

                    # get model stats
                    precision, recall, (tn, fp, fn, tp) = utils.get_model_stats(predicted_output, testY)
                    utils.set_model_auc(model, predicted_output, testY)
                    PUstats.objects.create(pos_class=str(pos_class), neg_class=",".join([str(i) for i in neg_class_list]),
                                           precision=precision, recall=recall, true_pos=tp, true_neg=tn, false_pos=fp,
                                           false_neg=fn, test_type=test_name,
                                           exclude_class_indx=",".join([str(i) for i in exclude_list]),
                                           no_train_pos_labelled=int(len(train_lp_pixels[0])),
                                           no_train_pos_unlabelled=int(len(train_up_pixels[0])),
                                           no_train_neg_unlabelled=int(len(train_un_pixels[0])),
                                           train_pos_neg_ratio=ratio, threshold=model.threshold, auc=model.auc,
                                           data_name=Config.data)

if __name__ == '__main__':
    run(None, None, None)
